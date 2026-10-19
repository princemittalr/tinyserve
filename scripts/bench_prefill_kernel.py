"""Prefill attention alone: M8's batched SDPA vs the M9 Triton kernel's tile shapes.

One layer at Qwen3-0.6B's shapes (16 query heads, 8 kv heads, head_dim 128, fp16,
block size 16), with the k/v already in the cache, for two batches:

    prefix   32 prefix-cache hits: 1,024 cached + 64 new tokens each (M5's workload)
    prompts  8 fresh prompts of 512 tokens

It prints the median time of one call and the largest difference from the batched
result, for the batched path and for each kernel tile (rows per tile, keys per step,
warps). M9 step 8 picks ``triton_prefill.DEFAULT_TILE`` from this.

Usage:
    python scripts/bench_prefill_kernel.py              # on the T4
"""

import argparse

import torch

from tinyserve.attention import batched_prefill_attention, prefill_plan
from tinyserve.batch import build_batch
from tinyserve.block_allocator import BlockAllocator
from tinyserve.profiling import prefill_items, time_ms
from tinyserve.triton_prefill import DEFAULT_TILE, paged_prefill_attention

HQ, HKV, D, BS = 16, 8, 128, 16
TILES = [DEFAULT_TILE, (16, 16, 4), (32, 16, 4), (32, 16, 8), (64, 32, 8), (64, 32, 4)]
WORKLOADS = {"prefix": (32, 1024, 64), "prompts": (8, 0, 512)}  # (seqs, cached, new)


def run(name: str, seqs: int, cached: int, new: int, dev: torch.device, iters: int) -> None:
    g = torch.Generator().manual_seed(0)
    alloc = BlockAllocator(seqs * -(-(cached + new) // BS) + 1)
    batch = build_batch(prefill_items(alloc, BS, seqs, cached, new), dev)
    slots = alloc.num_blocks * BS
    k = torch.randn(slots, HKV, D, generator=g).half().to(dev)
    v = torch.randn(slots, HKV, D, generator=g).half().to(dev)
    q = torch.randn(seqs * new, HQ, D, generator=g).half().to(dev)
    scale = D**-0.5
    starts = [i * new for i in range(seqs)]
    plan = prefill_plan(starts, batch.query_lens, batch.seq_slots)

    def batched() -> torch.Tensor:
        out = torch.empty_like(q)
        out[plan.rows[plan.real]] = batched_prefill_attention(q, k, v, plan, scale)[plan.real]
        return out

    want = batched()
    print(f"{name}: {seqs} sequences, {cached} cached + {new} new tokens each")
    print(f"  batched SDPA (M8)     {time_ms(batched, dev, iters=iters):8.2f} ms")
    q_starts = torch.tensor(starts, dtype=torch.int32, device=dev)
    q_lens = torch.tensor(batch.query_lens, dtype=torch.int32, device=dev)
    for rows, block_n, warps in TILES:

        def kernel(r: int = rows, n: int = block_n, w: int = warps) -> torch.Tensor:
            return paged_prefill_attention(
                q, k, v, batch.block_tables, q_starts, q_lens, batch.context_lens, scale,
                BS, new, rows_per_tile=r, block_n=n, num_warps=w,
            )  # fmt: skip

        diff = (kernel() - want).abs().max().item()
        ms = time_ms(kernel, dev, iters=iters)
        mark = " (default)" if (rows, block_n, warps) == DEFAULT_TILE else ""
        print(
            f"  triton {rows:3}/{block_n:2}/{warps}       {ms:8.2f} ms, max diff {diff:.1e}{mark}"
        )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()
    for name, (seqs, cached, new) in WORKLOADS.items():
        run(name, seqs, cached, new, torch.device(args.device), args.iters)


if __name__ == "__main__":
    main()
