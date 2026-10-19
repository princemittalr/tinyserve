"""M5 benchmark: TTFT and throughput on a shared-prefix workload, prefix cache off vs on.

Workload: ``--num-prefixes`` long random prefixes of ``--prefix-len`` tokens (think: a
system prompt or a document); request i is prefix ``i % num_prefixes`` plus its own
``--suffix-len`` random tokens, and generates exactly ``--output-len`` tokens (no EOS).
Each concurrency level runs closed-loop twice on the same prompts: once without the
prefix cache, once with it. Each run starts with an empty prefix cache, so the first
request of every prefix pays for the full prefill in both runs.

Also reported: how many requests got identical tokens in the two runs. In fp32 every
one should; in fp16 a different batch shape can flip a near-tie (see M4's parity notes),
so this is a check to read, not a pass/fail.

Usage:
    python scripts/bench_prefix.py --device cuda --attention triton      # T4 (Kaggle)
    python scripts/bench_prefix.py --concurrency 1,4 --num-requests 8 --prefix-len 64 \\
        --suffix-len 8 --output-len 8                                     # laptop smoke run
"""

import argparse
import datetime as dt
import json
import platform
from pathlib import Path

import torch

from tinyserve.bench import num_requests_for, run_closed_loop, shared_prefix_prompts
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
    parser.add_argument("--concurrency", default="1,8,32")
    parser.add_argument("--num-requests", type=int, default=None,
                        help="per level; default: max(16, 4 x concurrency)")  # fmt: skip
    parser.add_argument("--num-prefixes", type=int, default=4)
    parser.add_argument("--prefix-len", type=int, default=1024)
    parser.add_argument("--suffix-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=None,
                        help="default: room for the largest concurrency")  # fmt: skip
    parser.add_argument("--max-batch-tokens", type=int, default=2048)
    parser.add_argument("--attention", choices=DECODE_ATTENTIONS, default="torch")
    parser.add_argument("--prefill-attention", choices=PREFILL_ATTENTIONS, default="per_seq",
                        help="per sequence, one SDPA call (M8) or the M9 kernel")  # fmt: skip
    parser.add_argument("--cuda-graphs", action="store_true",
                        help="replay CUDA graphs for decode steps (M8; needs triton)")  # fmt: skip
    parser.add_argument("--chunked-prefill", action="store_true",
                        help="split prompts into chunks of the step budget (M9)")  # fmt: skip
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype or ("float16" if args.device != "cpu" else "float32"))
    levels = [int(c) for c in args.concurrency.split(",")]
    out = args.out or Path(f"results/m5_prefix_bench_{args.device.split(':')[0]}.json")

    cfg = ModelConfig.from_file(args.model_dir / "config.json")
    weights = load_weights(args.model_dir, cfg, dtype, args.device)
    model = Qwen3(cfg, weights, decode_attention=args.attention,
                  prefill_attention=args.prefill_attention)  # fmt: skip
    most = max(num_requests_for(c, args.num_requests) for c in levels)
    prompts = shared_prefix_prompts(most, args.num_prefixes, args.prefix_len, args.suffix_len,
                                    cfg.vocab_size, args.seed)  # fmt: skip

    prompt_len = args.prefix_len + args.suffix_len
    per_seq = -(-(prompt_len + args.output_len) // args.block_size)
    num_blocks = args.num_blocks or per_seq * max(levels)
    cache = KVCache(cfg, num_blocks, args.block_size, dtype, args.device)  # reused by every run
    print(f"{hardware(args.device)}, {dtype}, {args.attention} decode attention, "
          f"KV cache {num_blocks} blocks x {args.block_size} tokens = "
          f"{cache.memory_bytes / 2**30:.2f} GiB; prompts {args.num_prefixes} x "
          f"{args.prefix_len}-token prefix + {args.suffix_len}-token suffix, "
          f"{args.output_len} output tokens")  # fmt: skip

    graphs = DecodeGraphs(model, cache, max_batch=max(levels)) if args.cuda_graphs else None

    def engine(max_num_seqs: int, prefix_cache: bool) -> Engine:
        return Engine(model, cache, max_num_seqs, args.max_batch_tokens,
                      prefix_cache=prefix_cache, decode_graphs=graphs,
                      chunked_prefill=args.chunked_prefill)  # fmt: skip

    run_closed_loop(engine(2, True), prompts[:2], 4, concurrency=2)  # warm-up, not reported
    results = []
    for c in levels:
        n = num_requests_for(c, args.num_requests)
        row: dict[str, object] = {"concurrency": c}
        outputs = {}
        for on in (False, True):
            e = engine(c, on)
            r = run_closed_loop(e, prompts[:n], args.output_len, concurrency=c)
            outputs[on] = [s.output for s in sorted(e.seqs.values(), key=lambda s: s.seq_id)]
            row["cache_on" if on else "cache_off"] = r.as_dict()
            print(f"C={c:>3} cache {'on ' if on else 'off'}: {r.output_tok_per_s:7.1f} tok/s | "
                  f"TTFT p50/p99 {r.ttft_p50_ms:6.0f} / {r.ttft_p99_ms:6.0f} ms | "
                  f"ITL p99/max {r.itl_p99_ms or 0:4.0f} / {r.itl_max_ms or 0:4.0f} ms | "
                  f"prefix hit {r.prefix_hit_rate:.0%} | steps {r.steps} | "
                  f"preemptions {r.preemptions}")  # fmt: skip
        same = sum(a == b for a, b in zip(outputs[False], outputs[True], strict=True))
        row["identical_outputs"] = same
        print(f"C={c:>3}: {same}/{n} requests produced identical tokens with and without")
        results.append(row)

    report = {
        "date": dt.date.today().isoformat(),
        "hardware": hardware(args.device),
        "torch": torch.__version__,
        "dtype": str(dtype).removeprefix("torch."),
        "attention": args.attention,
        "prefill_attention": args.prefill_attention,
        "cuda_graphs": args.cuda_graphs,
        "chunked_prefill": args.chunked_prefill,
        "max_batch_tokens": args.max_batch_tokens,
        "args": {k: str(v) for k, v in vars(args).items()},
        "num_blocks": num_blocks,
        "results": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
