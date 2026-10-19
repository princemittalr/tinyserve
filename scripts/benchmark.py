"""M3 benchmark: output tok/s, TTFT and TPOT p50/p99 and KV utilization vs concurrency.

Synthetic workload: random-token prompts of ``--input-len`` tokens, each generating
exactly ``--output-len`` tokens (no EOS), run closed-loop at each concurrency (see
tinyserve/bench.py). ``--num-requests`` per level defaults to 4 per concurrency slot (at
least 16). Every level uses the first N prompts of the same seeded list, and
scripts/bench_hf.py runs the same prompts through HF generate() as the baseline.

Usage:
    python scripts/benchmark.py --device cuda --dtype float16          # T4 (Kaggle)
    python scripts/benchmark.py --device cuda --attention triton       # M4 kernel
    python scripts/benchmark.py --device cuda --attention triton --cuda-graphs   # M8
    python scripts/benchmark.py --concurrency 1,4 --num-requests 8 --input-len 32 \\
        --output-len 16                                                # laptop smoke run
Laptop CPU numbers only show that the harness works; they are not benchmark results.
"""

import argparse
import datetime as dt
import json
import platform
from pathlib import Path

import torch

from tinyserve.bench import num_requests_for, run_closed_loop, synthetic_prompts
from tinyserve.config import ModelConfig
from tinyserve.cuda_graphs import DecodeGraphs
from tinyserve.engine import Engine
from tinyserve.kv_cache import KVCache
from tinyserve.model import DECODE_ATTENTIONS, PREFILL_ATTENTIONS, Qwen3
from tinyserve.weights import load_weights


def hardware(device: str) -> str:
    if device.startswith("cuda"):
        return f"{torch.cuda.get_device_name()}, CUDA {torch.version.cuda}"
    return f"{platform.system()} {platform.machine()} CPU"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/Qwen3-0.6B"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16"], default=None,
                        help="default: float16 on cuda, float32 on cpu")  # fmt: skip
    parser.add_argument("--concurrency", default="1,8,32,64")
    parser.add_argument("--num-requests", type=int, default=None,
                        help="per level; default: max(16, 4 x concurrency)")  # fmt: skip
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=None,
                        help="default: room for the largest concurrency")  # fmt: skip
    parser.add_argument("--max-batch-tokens", type=int, default=2048)
    parser.add_argument("--attention", choices=DECODE_ATTENTIONS, default="torch",
                        help="decode attention: per-sequence PyTorch or M4 kernel")  # fmt: skip
    parser.add_argument("--prefill-attention", choices=PREFILL_ATTENTIONS, default="per_seq",
                        help="per sequence, one SDPA call (M8) or the M9 kernel")  # fmt: skip
    parser.add_argument("--cuda-graphs", action="store_true",
                        help="replay CUDA graphs for decode steps (M8; needs triton)")  # fmt: skip
    parser.add_argument("--chunked-prefill", action="store_true",
                        help="split prompts into chunks of the step budget (M9)")  # fmt: skip
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--fused-kernels", action="store_true",
                        help="fused norm / RoPE / cache-write Triton kernels (M9)")  # fmt: skip
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype or ("float16" if args.device != "cpu" else "float32"))
    levels = [int(c) for c in args.concurrency.split(",")]
    milestone = "m4" if args.attention != "torch" else "m3"
    milestone = "m8" if args.cuda_graphs else milestone
    milestone = "m9" if args.chunked_prefill or args.fused_kernels else milestone
    out = args.out or Path(f"results/{milestone}_bench_{args.device.split(':')[0]}.json")

    cfg = ModelConfig.from_file(args.model_dir / "config.json")
    weights = load_weights(args.model_dir, cfg, dtype, args.device)
    model = Qwen3(cfg, weights, decode_attention=args.attention,
                  prefill_attention=args.prefill_attention,
                  fused_kernels=args.fused_kernels)  # fmt: skip
    most = max(num_requests_for(c, args.num_requests) for c in levels)
    prompts = synthetic_prompts(most, args.input_len, cfg.vocab_size, args.seed)

    per_seq = -(-(args.input_len + args.output_len) // args.block_size)
    num_blocks = args.num_blocks or per_seq * max(levels)
    cache = KVCache(cfg, num_blocks, args.block_size, dtype, args.device)  # reused by every run
    print(f"{hardware(args.device)}, {dtype}, {args.attention} decode attention, "
          f"KV cache {num_blocks} blocks x "
          f"{args.block_size} tokens = {cache.memory_bytes / 2**30:.2f} GiB")  # fmt: skip

    graphs = None
    if args.cuda_graphs:  # captured once, shared by every level's engine
        graphs = DecodeGraphs(model, cache, max_batch=max(levels))
        mib = graphs.capture_bytes / 2**20
        print(f"CUDA graphs: buckets {graphs.buckets}, captured in "
              f"{graphs.capture_seconds:.1f} s, {mib:.0f} MiB")  # fmt: skip

    def engine(max_num_seqs: int) -> Engine:
        return Engine(model, cache, max_num_seqs, args.max_batch_tokens, decode_graphs=graphs,
                      chunked_prefill=args.chunked_prefill)  # fmt: skip

    run_closed_loop(engine(2), prompts[:2], 4, concurrency=2)  # warm-up, not reported
    results = []
    for c in levels:
        n = num_requests_for(c, args.num_requests)
        r = run_closed_loop(engine(c), prompts[:n], args.output_len, concurrency=c)
        results.append(r.as_dict())
        tpot = "-"
        if r.tpot_p50_ms is not None and r.tpot_p99_ms is not None:
            tpot = f"{r.tpot_p50_ms:.1f} / {r.tpot_p99_ms:.1f}"
        print(f"concurrency {c:>3}: {r.output_tok_per_s:8.1f} tok/s | "
              f"TTFT p50/p99 {r.ttft_p50_ms:.0f} / {r.ttft_p99_ms:.0f} ms | "
              f"TPOT p50/p99 {tpot} ms | ITL p99/max {r.itl_p99_ms or 0:.0f} / "
              f"{r.itl_max_ms or 0:.0f} ms | KV util mean {r.kv_util_mean:.0%} "
              f"max {r.kv_util_max:.0%} | preemptions {r.preemptions}")  # fmt: skip

    report = {
        "date": dt.date.today().isoformat(),
        "hardware": hardware(args.device),
        "torch": torch.__version__,
        "dtype": str(dtype).removeprefix("torch."),
        "attention": args.attention,
        "prefill_attention": args.prefill_attention,
        "fused_kernels": args.fused_kernels,
        "chunked_prefill": args.chunked_prefill,
        "max_batch_tokens": args.max_batch_tokens,
        "cuda_graphs": None
        if graphs is None
        else {
            "buckets": graphs.buckets,
            "capture_seconds": graphs.capture_seconds,
            "capture_bytes": graphs.capture_bytes,
        },
        "args": {k: str(v) for k, v in vars(args).items()},
        "num_blocks": num_blocks,
        "results": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
