"""M4 step 1: where does one decode step's time go?

Every sequence decodes one token with ``--context-len`` tokens already cached (192 = a
128-token prompt halfway through 128 output tokens, the middle of the M3 benchmark).
For each concurrency C it times, on the same batch:

    build      build_batch: block tables -> slot tensors on the device (host work)
    forward    forward_batch, as the engine runs it
    no-attn    forward_batch with attention replaced by a stand-in that returns q

forward - no-attn is what attention costs as written. Then one forward at
``--profile-concurrency`` runs under torch.profiler: op calls, kernel launches, GPU busy
fraction and the top operators. The cache holds zeros; timing does not depend on values.

Usage:
    python scripts/profile_step.py --device cuda                      # T4 (Kaggle)
    python scripts/profile_step.py --concurrency 1,4 --profile-concurrency 4   # laptop
Laptop CPU numbers only show that the script works.
"""

import argparse
import datetime as dt
import json
import platform
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from tinyserve.batch import build_batch
from tinyserve.block_allocator import BlockAllocator
from tinyserve.block_table import BlockTable
from tinyserve.config import ModelConfig
from tinyserve.kv_cache import KVCache
from tinyserve.model import Qwen3
from tinyserve.profiling import attention_replaced, decode_tables, profile_summary, time_ms
from tinyserve.weights import load_weights


def hardware(device: torch.device) -> str:
    if device.type == "cuda":
        return f"{torch.cuda.get_device_name(device)}, CUDA {torch.version.cuda}"
    return f"{platform.system()} {platform.machine()} CPU"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/Qwen3-0.6B"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16"], default=None,
                        help="default: float16 on cuda, float32 on cpu")  # fmt: skip
    parser.add_argument("--concurrency", default="1,8,32,64")
    parser.add_argument("--profile-concurrency", type=int, default=64)
    parser.add_argument("--context-len", type=int, default=192)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--trace", type=Path, default=None,
                        help="also write a Chrome trace (open in ui.perfetto.dev)")  # fmt: skip
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype or ("float16" if device.type != "cpu" else "float32"))
    levels = [int(c) for c in args.concurrency.split(",")]
    out = args.out or Path(f"results/m4_profile_{device.type}.json")

    cfg = ModelConfig.from_file(args.model_dir / "config.json")
    model = Qwen3(cfg, load_weights(args.model_dir, cfg, dtype, device))
    per_seq = -(-(args.context_len + 1) // args.block_size)
    num_blocks = per_seq * max(*levels, args.profile_concurrency)
    cache = KVCache(cfg, num_blocks, args.block_size, dtype, device)
    print(f"{hardware(device)}, {dtype}, context {args.context_len} tokens, "
          f"{cfg.num_layers} layers")  # fmt: skip

    def tables_for(c: int) -> list[tuple[list[int], BlockTable]]:
        tables = decode_tables(BlockAllocator(num_blocks), args.block_size, c, args.context_len)
        return [([0], t) for t in tables]

    def measure(c: int) -> dict[str, float]:
        items = tables_for(c)
        batch = build_batch(items, device)
        build = time_ms(lambda: build_batch(items, device), device, iters=args.iters)
        fwd = time_ms(lambda: model.forward_batch(batch, cache), device, iters=args.iters)
        with attention_replaced():
            no_attn = time_ms(lambda: model.forward_batch(batch, cache), device, iters=args.iters)
        return {"concurrency": c, "build_ms": build, "forward_ms": fwd,
                "no_attention_ms": no_attn, "attention_share": (fwd - no_attn) / fwd}  # fmt: skip

    rows = []
    print(f"{'C':>4} | {'build ms':>9} | {'forward ms':>10} | {'no-attn ms':>10} | attention share")
    for c in levels:
        r = measure(c)
        rows.append(r)
        print(f"{c:>4} | {r['build_ms']:9.1f} | {r['forward_ms']:10.1f} | "
              f"{r['no_attention_ms']:10.1f} | {r['attention_share']:6.0%}")  # fmt: skip

    c = args.profile_concurrency
    batch = build_batch(tables_for(c), device)
    model.forward_batch(batch, cache)  # warm-up outside the profiler
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, acc_events=True) as prof:
        wall = time_ms(lambda: model.forward_batch(batch, cache), device, warmup=0, iters=1)
    summary = profile_summary(prof, wall)
    print(f"\nprofiled one forward at C={c}: {wall:.1f} ms wall (slower under the profiler), "
          f"{summary['aten_calls']:,.0f} aten op calls, "
          f"{summary['kernel_launches']:,.0f} kernel launches, "
          f"GPU busy {summary['gpu_kernel_ms']:.1f} ms "
          f"= {summary['gpu_busy']:.0%} of wall")  # fmt: skip
    sort = "self_device_time_total" if device.type == "cuda" else "self_cpu_time_total"
    table = prof.key_averages().table(sort_by=sort, row_limit=15)
    print(table)
    if args.trace:
        prof.export_chrome_trace(str(args.trace))
        print(f"wrote {args.trace}")

    report = {
        "date": dt.date.today().isoformat(),
        "hardware": hardware(device),
        "torch": torch.__version__,
        "dtype": str(dtype).removeprefix("torch."),
        "args": {k: str(v) for k, v in vars(args).items()},
        "timings": rows,
        "profile": {"concurrency": c, **summary},
        "top_ops": table,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
