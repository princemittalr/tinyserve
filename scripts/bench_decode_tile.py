"""M11 step 1: the decode kernel at several tiles, against what the T4's memory really gives.

M10 run 2 measured the C=32 decode step at 14.4 ms against a 5.9 ms floor; the decode
kernel moves k/v at 95 GB/s at any split count. That floor uses the T4's spec, 320 GB/s.
Two things are measured here, in one process:

1. The ceiling: the bandwidth a plain read (``x.sum()``) and a copy (``y.copy_(x)``) of
   512 MiB reach on this GPU at its clock. Every "% of floor" so far divided by 320.
2. The decode kernel (Qwen3-0.6B shapes, fp16, one layer, the split count the engine
   picks) at each tile: cache blocks loaded per loop iteration x warps per program
   (``DECODE_TILE``, M11). 1x4 is the kernel M8-M10 measured. The tiles are timed in
   ``--rounds`` rounds, each in a rotated order, and the median is reported, since one
   point of M10's sweep varied by up to ~25% between sessions. Each tile's output is
   compared with 1x4's on the same inputs.

The tiles skip those ptxas spills heavily on sm_75 (4x1, 4x2, 8x2, 8x4: check_t4_compile.py).
M11 step 2 added 1x1 and 2x1 after run 1 found 1x2 fastest.

Usage (T4): python scripts/bench_decode_tile.py
"""

import argparse
import datetime as dt
import json
import statistics
from functools import partial
from pathlib import Path
from typing import Any

import torch

from tinyserve.batch import build_batch
from tinyserve.block_allocator import BlockAllocator
from tinyserve.profiling import GpuSampler, decode_tables, loop_ms
from tinyserve.triton_attention import launch_paged_decode_split, num_splits_for

HQ, HKV, D = 16, 8, 128  # Qwen3-0.6B
BASE = (1, 4)  # M8-M10's loop and launch


def ceiling(dev: torch.device, mib: int, iters: int) -> dict[str, float]:
    """GB/s of a read and of a copy of ``mib`` MiB (a copy moves its bytes twice)."""
    x = torch.ones(mib * 2**19, device=dev, dtype=torch.float16)
    y = torch.empty_like(x)
    gb = x.numel() * 2 / 1e9
    read = loop_ms(x.sum, dev, iters, warmup=20)
    copy = loop_ms(lambda: y.copy_(x), dev, iters, warmup=20)
    return {"read_gbps": gb / read * 1e3, "copy_gbps": 2 * gb / copy * 1e3}


def time_case(b: int, ctx: int, tiles: list[tuple[int, int]], args: argparse.Namespace
              ) -> dict[str, Any]:  # fmt: skip
    """One layer's decode kernel for ``b`` sequences of ``ctx`` cached tokens + 1 new."""
    dev, bs = torch.device(args.device), args.block_size
    num_blocks = b * -(-(ctx + 1) // bs)
    tables = decode_tables(BlockAllocator(num_blocks), bs, b, ctx)
    batch = build_batch([([0], t) for t in tables], dev)
    g = torch.Generator(device=dev).manual_seed(0)
    shape = (num_blocks * bs, HKV, D)
    k = torch.randn(shape, generator=g, device=dev, dtype=torch.float16)
    v = torch.randn(shape, generator=g, device=dev, dtype=torch.float16)
    q = torch.randn(b, HQ, D, generator=g, device=dev, dtype=torch.float16)
    splits = num_splits_for(b, HKV, batch.block_tables.shape[1])

    def launch(tile: tuple[int, int]) -> torch.Tensor:
        return launch_paged_decode_split(q, k, v, batch.block_tables, batch.context_lens,
                                         D**-0.5, bs, num_splits=splits, tile=tile)  # fmt: skip

    ref = launch(BASE)
    diff = {t: (launch(t) - ref).abs().max().item() for t in tiles}
    runs: dict[tuple[int, int], list[float]] = {t: [] for t in tiles}
    for r in range(args.rounds):
        for t in tiles[r % len(tiles) :] + tiles[: r % len(tiles)]:
            runs[t].append(loop_ms(partial(launch, t), dev, args.iters, warmup=10))
    ms = {t: statistics.median(x) for t, x in runs.items()}
    best = min(ms, key=lambda t: ms[t])
    kv_gb = b * (ctx + 1) * HKV * D * 2 * 2 / 1e9  # k and v, each read once
    cells = " ".join(f"{t[0]}x{t[1]}:{m:.3f}" for t, m in ms.items())
    print(f"B={b:<3} ctx={ctx:<5} splits={splits:<2} {cells} | best {best[0]}x{best[1]}: "
          f"{kv_gb / ms[best] * 1e3:.0f} GB/s, {ms[BASE] / ms[best]:.2f}x 1x4 | "
          f"max |diff| {max(diff.values()):.1e}", flush=True)  # fmt: skip
    return {"batch": b, "context": ctx, "splits": splits, "kv_gb": kv_gb,
            "ms": {f"{t[0]}x{t[1]}": m for t, m in ms.items()},
            "runs": {f"{t[0]}x{t[1]}": x for t, x in runs.items()},
            "max_abs_diff": {f"{t[0]}x{t[1]}": x for t, x in diff.items()}}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", default="1,8,16,32,64")
    parser.add_argument("--contexts", default="160,1024")
    parser.add_argument("--tiles", default="1x4,1x1,1x2,1x8,2x1,2x2,2x4,2x8,4x4,4x8,8x8")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--ceiling-mib", type=int, default=512)
    parser.add_argument("--out", type=Path, default=Path("results/m11_decode_tile_cuda.json"))
    args = parser.parse_args()
    dev = torch.device(args.device)
    tiles = [(int(a), int(w)) for a, w in (x.split("x") for x in args.tiles.split(","))]
    if BASE not in tiles:
        tiles.insert(0, BASE)
    with GpuSampler() as clocks:
        ceil = ceiling(dev, args.ceiling_mib, args.iters)
    print(f"ceiling: read {ceil['read_gbps']:.0f} GB/s, copy {ceil['copy_gbps']:.0f} GB/s "
          f"(spec 320) | {GpuSampler.describe(clocks.summary())}", flush=True)  # fmt: skip
    print("ms per layer at each tile (blocks per iteration x warps), median of rounds")
    with GpuSampler() as clocks:
        rows = [time_case(b, ctx, tiles, args)
                for ctx in (int(c) for c in args.contexts.split(","))
                for b in (int(x) for x in args.batches.split(","))]  # fmt: skip
    print(f"during the sweep: {GpuSampler.describe(clocks.summary())}")
    hw = torch.cuda.get_device_name() if dev.type == "cuda" else "CPU (interpreter)"
    report = {"date": dt.date.today().isoformat(), "hardware": hw, "torch": torch.__version__,
              "ceiling": ceil, "clocks": clocks.summary(), "rows": rows}  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
