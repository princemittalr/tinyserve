"""CUDA graphs for the decode step (M8).

M7 measured a decode step at C=32 as launch-bound: the forward pass launches 1,851
kernels and takes 34.2 ms, but the GPU is busy for only 16.7 ms of it. The rest is
Python and the CUDA driver queueing kernels one by one. A CUDA graph records the
kernels of one forward pass once (``capture``); replaying it later sends all of them to
the GPU in one call, with no Python in between.

The catch is that a replay repeats exactly what was recorded: the same kernels, the same
shapes, the same memory addresses. So:
  - the inputs live in fixed buffers (``DecodeInputs``) that each step overwrites in
    place, and the output is the tensor the captured pass wrote (overwritten by the next
    replay, so read it before then);
  - nothing in the pass may depend on a Python value that changes between steps, or wait
    for the GPU (``Qwen3.forward_decode`` is written for this);
  - one graph per batch size. A step with 5 sequences replays the graph for 8, with 3
    padding rows (the KV cache's padding block takes their writes).

Only pure decode steps (every sequence brings one token) use a graph. Prefill and mixed
steps change shape every time and run eagerly, as before.

    graphs = DecodeGraphs(model, cache, max_batch=64)   # captures (a few seconds on a T4)
    engine = Engine(model, cache, decode_graphs=graphs)

On a CPU, or with ``capture=False``, ``DecodeGraphs`` runs ``forward_decode`` eagerly on
the same padded buffers, so the bucketing and padding are tested without a GPU.
"""

import time
from collections.abc import Sequence

import torch

from tinyserve.batch import DecodeInputs, fill_decode_inputs
from tinyserve.block_table import BlockTable
from tinyserve.kv_cache import KVCache
from tinyserve.model import Qwen3


def default_buckets(max_batch: int) -> list[int]:
    """Batch sizes to capture: 1, 2, 4, 8, then every multiple of 8 up to ``max_batch``.

    A step pads up to the next bucket, so padding wastes at most 7 rows past 8 sequences;
    each bucket costs one graph and one ``[bucket, vocab]`` float32 output buffer.
    """
    if max_batch < 1:
        raise ValueError(f"max_batch must be >= 1, got {max_batch}")
    small = [b for b in (1, 2, 4) if b < max_batch]
    return [*small, *range(8, max_batch, 8), max_batch]


class DecodeGraphs:
    def __init__(
        self,
        model: Qwen3,
        cache: KVCache,
        max_batch: int = 64,
        buckets: Sequence[int] | None = None,
        max_blocks_per_seq: int | None = None,
        capture: bool | None = None,
    ) -> None:
        """Allocate the input buffers and, on a GPU, capture one graph per bucket.

        Args:
            max_batch: the largest decode batch that replays a graph; larger ones run
                eagerly. Set it to the engine's ``max_num_seqs``.
            buckets: batch sizes to capture (default: ``default_buckets(max_batch)``).
            max_blocks_per_seq: block-table width. Default: the most blocks one sequence
                can hold, ``min(num_blocks, ceil(max_position_embeddings / block_size))``.
                A sequence with more blocks than this runs eagerly.
            capture: default True on CUDA, False elsewhere.
        """
        if model.decode_launch is None:
            raise ValueError('CUDA graphs need decode_attention="triton"')
        self.model = model
        self.cache = cache
        self.buckets = sorted(set(buckets or default_buckets(max_batch)))
        if self.buckets[0] < 1:
            raise ValueError(f"bucket sizes must be >= 1: {self.buckets}")
        width = max_blocks_per_seq or min(
            cache.num_blocks, -(-model.cfg.max_position_embeddings // cache.block_size)
        )
        device = cache.k.device
        self.inputs = DecodeInputs.empty(self.buckets[-1], width, device)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.outputs: dict[int, torch.Tensor] = {}  # each graph's logits buffer
        self.capture_seconds = 0.0
        self.capture_bytes = 0  # GPU memory reserved by capturing (pool + output buffers)
        if capture if capture is not None else device.type == "cuda":
            self._capture()

    @property
    def max_batch(self) -> int:
        return self.buckets[-1]

    @property
    def max_blocks_per_seq(self) -> int:
        return self.inputs.block_tables.shape[1]

    def bucket_for(self, num_seqs: int) -> int:
        """The smallest bucket that holds ``num_seqs`` rows."""
        for b in self.buckets:
            if b >= num_seqs:
                return b
        raise ValueError(f"{num_seqs} sequences exceed the largest bucket ({self.max_batch})")

    def fits(self, tables: Sequence[BlockTable]) -> bool:
        """Can a decode step over these sequences replay a graph?"""
        width = self.max_blocks_per_seq
        return 0 < len(tables) <= self.max_batch and all(len(t.blocks) <= width for t in tables)

    def prepare(self, items: Sequence[tuple[int, BlockTable]]) -> int:
        """Fill the input buffers for one decode step; returns the bucket to replay.

        ``items`` are ``(new_token, table)`` pairs, as for ``fill_decode_inputs``.
        """
        bucket = self.bucket_for(len(items))
        fill_decode_inputs(self.inputs.head(bucket), items, self.cache.pad_block)
        return bucket

    def replay(self, bucket: int, num_seqs: int) -> torch.Tensor:
        """Run the prepared step: ``[num_seqs, vocab]`` float32 logits.

        With graphs, the result is a view of that graph's output buffer: the next replay
        overwrites it, so use it (sample) before preparing another step.
        """
        graph = self.graphs.get(bucket)
        if graph is None:  # not captured (CPU, or capture=False): same pass, eagerly
            return self.model.forward_decode(self.inputs.head(bucket), self.cache)[:num_seqs]
        graph.replay()
        return self.outputs[bucket][:num_seqs]

    def _capture(self) -> None:
        """Record one graph per bucket, largest first, all in one memory pool.

        Before capturing, the pass runs once per bucket on a side stream (the largest
        three times): the first runs compile the Triton kernel, let cuBLAS set up its
        workspace and load the GEMM kernels each batch size picks, and none of that may
        happen during capture. The captures share one memory pool, so the intermediate
        tensors of one graph's pass reuse the memory of another's (they never run at the
        same time); only each graph's output buffer stays reserved.
        """
        device = self.cache.k.device
        torch.cuda.synchronize(device)
        start, mem0 = time.perf_counter(), torch.cuda.memory_reserved(device)
        # All-padding inputs: every row writes to and reads from the padding block, so
        # capturing changes no block that a sequence owns.
        pad = self.cache.pad_block
        self.inputs.token_ids.zero_()
        self.inputs.positions.zero_()
        self.inputs.slots.fill_(pad * self.cache.block_size)
        self.inputs.block_tables.fill_(pad)
        self.inputs.context_lens.fill_(1)

        side = torch.cuda.Stream(device)  # type: ignore[no-untyped-call]
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            for b in [*self.buckets, self.max_batch, self.max_batch]:
                self.model.forward_decode(self.inputs.head(b), self.cache)
        torch.cuda.current_stream(device).wait_stream(side)

        pool = None
        for b in reversed(self.buckets):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                self.outputs[b] = self.model.forward_decode(self.inputs.head(b), self.cache)
            pool = graph.pool()
            self.graphs[b] = graph
        torch.cuda.synchronize(device)
        self.capture_seconds = time.perf_counter() - start
        self.capture_bytes = torch.cuda.memory_reserved(device) - mem0
