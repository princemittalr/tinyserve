"""The serving loop: the scheduler and the model, one step at a time.

    engine = Engine(model, cache)
    ids = [engine.add_request(p, max_new_tokens=64) for p in prompts]
    while engine.has_work():
        engine.step()          # requests may be added between steps (continuous batching)

Every step: the scheduler picks the sequences and reserves their cache slots, the model
runs all their pending tokens in one ``forward_batch``, and each sequence gets its next
token: greedy, or sampled with its own ``SamplingParams`` (M6, sampling.py). The engine
also records when each token came out, which is what the M3 benchmark turns into TTFT
and TPOT. With ``trace_steps=True`` (M7) it also records where each step's time went
(``StepRecord``); that waits for the GPU after the forward pass, so leave it off when
measuring throughput.

With ``decode_graphs`` (M8, cuda_graphs.py), every step in which all sequences decode one
token replays a captured CUDA graph instead of running ``forward_batch``; prefill and
mixed steps run eagerly as before.
"""

import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from tinyserve.batch import build_batch
from tinyserve.block_allocator import BlockAllocator
from tinyserve.cuda_graphs import DecodeGraphs
from tinyserve.kv_cache import KVCache
from tinyserve.model import Qwen3
from tinyserve.sampling import SamplingParams, sample
from tinyserve.scheduler import Scheduler
from tinyserve.sequence import SeqStatus, Sequence


@dataclass
class RequestTimes:
    """Wall-clock times (seconds, from the engine's clock) of one request."""

    arrival: float
    tokens: list[float] = field(default_factory=list)  # when each output token came out

    @property
    def ttft(self) -> float:
        """Time to first token: queueing + prefill."""
        return self.tokens[0] - self.arrival

    @property
    def tpot(self) -> float | None:
        """Mean time per output token after the first; None with a single token."""
        if len(self.tokens) < 2:
            return None
        return (self.tokens[-1] - self.tokens[0]) / (len(self.tokens) - 1)


@dataclass(frozen=True)
class StepRecord:
    """One engine step: what ran and how long each phase took (``Engine(trace_steps=True)``).

    A prefill sequence is one that brought more than one token (a prompt, the part after
    a cached prefix, or a recompute after preemption); the rest decode one token each.
    """

    num_seqs: int
    num_prefill_seqs: int
    num_tokens: int  # rows in the forward pass
    context_tokens: int  # sum of each sequence's tokens so far, new ones included
    schedule_ms: float  # schedule(): admission, preemption, prefix-cache lookups
    build_ms: float  # build_batch: block tables -> tensors on the device
    forward_ms: float  # forward_batch, until the GPU has finished it
    sample_ms: float  # sampling, plus copying the next tokens to the host
    update_ms: float  # scheduler.update(): record tokens, retire, cache blocks
    graph: bool = False  # the forward was a CUDA graph replay (M8); build_ms then fills it

    @property
    def total_ms(self) -> float:
        return self.schedule_ms + self.build_ms + self.forward_ms + self.sample_ms + self.update_ms


class Engine:
    def __init__(
        self,
        model: Qwen3,
        cache: KVCache,
        max_num_seqs: int = 64,
        max_batch_tokens: int = 2048,
        clock: Callable[[], float] = time.perf_counter,
        prefix_cache: bool = False,
        seed: int = 0,
        trace_steps: bool = False,
        decode_graphs: DecodeGraphs | None = None,
    ) -> None:
        if decode_graphs is not None and (
            decode_graphs.model is not model or decode_graphs.cache is not cache
        ):
            raise ValueError("decode_graphs must be captured for this engine's model and cache")
        self.model = model
        self.decode_graphs = decode_graphs
        self.cache = cache
        self.scheduler = Scheduler(
            BlockAllocator(cache.num_blocks),
            cache.block_size,
            max_num_seqs,
            max_batch_tokens,
            prefix_cache=prefix_cache,
        )
        self.clock = clock
        self.seqs: dict[int, Sequence] = {}
        self.times: dict[int, RequestTimes] = {}
        self.num_steps = 0
        self.kv_utilization: list[float] = []  # sampled once per step, after scheduling
        self._next_id = 0
        dev = cache.k.device
        self.sampling: dict[int, SamplingParams] = {}
        # Seeded requests get their own generator; the rest share this one (sampling.py).
        self.generators: dict[int, torch.Generator] = {}
        self.shared_generator = torch.Generator(device=dev).manual_seed(seed)
        self.step_records: list[StepRecord] | None = [] if trace_steps else None

    def add_request(
        self,
        prompt: list[int],
        max_new_tokens: int,
        eos_ids: frozenset[int] = frozenset(),
        sampling: SamplingParams | None = None,
    ) -> int:
        seq_id = self._next_id
        self._next_id += 1
        self.seqs[seq_id] = self.scheduler.add(seq_id, prompt, max_new_tokens, eos_ids)
        self.times[seq_id] = RequestTimes(arrival=self.clock())
        params = sampling or SamplingParams()
        self.sampling[seq_id] = params
        if params.seed is not None:
            g = torch.Generator(device=self.cache.k.device)
            self.generators[seq_id] = g.manual_seed(params.seed)
        return seq_id

    def has_work(self) -> bool:
        return self.scheduler.has_work()

    def abort(self, seq_id: int) -> None:
        """Stop a request between steps and free its blocks (no-op if it already finished)."""
        self.scheduler.abort(self.seqs[seq_id])
        self.generators.pop(seq_id, None)

    def release(self, seq_id: int) -> None:
        """Forget a finished request. A server calls this so its records do not grow forever;
        the benchmarks keep them to compute TTFT and TPOT afterwards."""
        if self.seqs[seq_id].status is not SeqStatus.FINISHED:
            raise ValueError(f"request {seq_id} has not finished; abort it first")
        del self.seqs[seq_id], self.times[seq_id], self.sampling[seq_id]
        self.generators.pop(seq_id, None)

    def step(self) -> list[Sequence]:
        """Run one model step. Returns the sequences that finished in it."""
        s = self.scheduler
        marks: list[float] = []  # phase boundaries, only when tracing
        self._mark(marks)
        seqs = s.schedule()
        if not seqs:
            # add() rejects requests that can never fit, so this would be a scheduler bug.
            raise RuntimeError("scheduler made no progress with requests still queued")
        self.kv_utilization.append(s.kv_utilization)
        # Each sequence's input is what schedule() just appended to its table: a whole
        # prompt, the part after a cached prefix (M5), or one decode token.
        items = [(q.tokens[-q.num_scheduled :], q.table) for q in seqs]
        # Read before update(), which releases the tables of finished sequences.
        context = sum(table.num_tokens for _, table in items)
        self._mark(marks)
        graphs = self.decode_graphs
        use_graph = (
            graphs is not None
            and all(len(t) == 1 for t, _ in items)
            and graphs.fits([table for _, table in items])
        )
        if graphs is not None and use_graph:
            bucket = graphs.prepare([(t[0], table) for t, table in items])
            self._mark(marks)
            logits = graphs.replay(bucket, len(items))
        else:
            batch = build_batch(items, device=self.cache.k.device)
            self._mark(marks)
            logits = self.model.forward_batch(batch, self.cache)
        self._mark(marks)
        params = [self.sampling[q.seq_id] for q in seqs]
        gens = [self.generators.get(q.seq_id) for q in seqs]
        next_tokens = sample(logits, params, gens, self.shared_generator).tolist()
        self._mark(marks)
        now = self.clock()
        for q in seqs:
            self.times[q.seq_id].tokens.append(now)
        self.num_steps += 1
        finished = s.update(seqs, next_tokens)
        for q in finished:
            self.generators.pop(q.seq_id, None)  # a finished request draws no more noise
        if self.step_records is not None:
            self._mark(marks)
            ms = [(b - a) * 1e3 for a, b in itertools.pairwise(marks)]
            self.step_records.append(
                StepRecord(
                    num_seqs=len(seqs),
                    num_prefill_seqs=sum(len(t) > 1 for t, _ in items),
                    num_tokens=sum(len(t) for t, _ in items),
                    context_tokens=context,
                    schedule_ms=ms[0],
                    build_ms=ms[1],
                    forward_ms=ms[2],
                    sample_ms=ms[3],
                    update_ms=ms[4],
                    graph=use_graph,
                )
            )
        return finished

    def _mark(self, marks: list[float]) -> None:
        """Tracing only: wait for queued GPU work, then note the time. CUDA calls return
        once kernels are queued, so without the wait their time would show up in whatever
        phase next waits for the GPU (the ``.tolist()`` after sampling)."""
        if self.step_records is None:
            return
        dev = self.cache.k.device
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        marks.append(self.clock())

    def generate(
        self,
        prompts: list[list[int]],
        max_new_tokens: int,
        eos_ids: frozenset[int] = frozenset(),
        sampling: SamplingParams | None = None,
    ) -> list[list[int]]:
        """Run all prompts to completion together; outputs in the order of ``prompts``."""
        ids = [self.add_request(p, max_new_tokens, eos_ids, sampling) for p in prompts]
        while self.has_work():
            self.step()
        return [self.seqs[i].output for i in ids]
