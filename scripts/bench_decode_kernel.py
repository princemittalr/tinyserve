"""M10 step 2: the decode kernel alone, at each split count, batch size and context length.

At C=32 a decode step spends ~5 ms of its ~15 ms in the decode kernel (M9 run 2), which
moves the k/v at ~100 GB/s of the T4's 320. ``num_splits_for`` stops splitting at batch
10, where the batch alone gives 80 programs (2 per SM), but each program walks all of its
sequence's blocks one after another, so a few programs per SM may not keep enough loads
in flight. This times the kernel at 1 (the plain grouped kernel), 2, 4, 8 and 16 splits
for batches 1 to 64 and contexts of 192 and 1,024 tokens (Qwen3-0.6B shapes, fp16, one
layer), and marks the split count the engine picks today. Since M10 the split count is a
plain int, so every split count runs the same compiled kernel.

Usage (T4): python scripts/bench_decode_kernel.py
"""

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import torch

from tinyserve.batch import build_batch
from tinyserve.block_allocator import BlockAllocator
from tinyserve.profiling import decode_tables, loop_ms
from tinyserve.triton_attention import launch_paged_decode_split, num_splits_for

HQ, HKV, D = 16, 8, 128  # Qwen3-0.6B


def time_case(b: int, ctx: int, splits: list[int], args: argparse.Namespace) -> dict[str, Any]:
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
    pick = num_splits_for(b, HKV, batch.block_tables.shape[1])
    times = {}
    for s in sorted({*splits, pick}):

        def launch(s: int = s) -> torch.Tensor:
            return launch_paged_decode_split(q, k, v, batch.block_tables, batch.context_lens,
                                             D**-0.5, bs, num_splits=s)  # fmt: skip

        times[s] = loop_ms(launch, dev, args.iters)
    best = min(times, key=lambda s: times[s])
    kv_gb = b * (ctx + 1) * HKV * D * 2 * 2 / 1e9  # k and v, each read once
    cells = " ".join(f"{s}:{t:.3f}{'*' if s == pick else ''}" for s, t in times.items())
    print(f"B={b:<3} ctx={ctx:<5} {cells} | best {best}: {kv_gb / times[best] * 1e3:.0f} GB/s, "
          f"engine's {kv_gb / times[pick] * 1e3:.0f} GB/s", flush=True)  # fmt: skip
    return {"batch": b, "context": ctx, "ms": times, "engine_splits": pick, "best_splits": best,
            "kv_gb": kv_gb}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", default="1,8,16,32,64")
    parser.add_argument("--contexts", default="192,1024")
    parser.add_argument("--splits", default="1,2,4,8,16")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("results/m10_decode_kernel_cuda.json"))
    args = parser.parse_args()
    splits = [int(s) for s in args.splits.split(",")]
    print("ms per layer at each split count (splits:ms); * = the engine's choice "
          "(num_splits_for), 1 = the plain grouped kernel")  # fmt: skip
    rows = [time_case(b, ctx, splits, args)
            for ctx in (int(c) for c in args.contexts.split(","))
            for b in (int(x) for x in args.batches.split(","))]  # fmt: skip
    dev = torch.device(args.device)
    hw = torch.cuda.get_device_name() if dev.type == "cuda" else "CPU (interpreter)"
    report = {"date": dt.date.today().isoformat(), "hardware": hw,
              "torch": torch.__version__, "rows": rows}  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
