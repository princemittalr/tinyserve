"""M6 benchmark: the M3/M4 workload sent over HTTP to scripts/serve.py, streamed as SSE.

Same prompts as scripts/benchmark.py (random token ids, ``--input-len`` 128, 128 output
tokens, EOS ignored), the same concurrency levels and request counts, closed loop: at
each level ``C`` client tasks each send their next request as soon as the last one has
finished streaming. Measured on the client: output tok/s, TTFT (request sent -> first
chunk) and TPOT. Run ``benchmark.py --out results/m6_engine_bench_<device>.json`` in the
same session to see what HTTP, JSON, SSE and detokenising cost on top of the engine.

The client runs on the same machine as the server and shares its CPU cores, so its own
parsing work is part of the measured cost.

Usage: python scripts/bench_api.py --device cuda --attention triton
       python scripts/bench_api.py --device cuda --attention triton_grouped --cuda-graphs \
           --prefill-attention batched        (M8; results/m8_api_bench_<device>.json)
"""

import argparse
import asyncio
import datetime as dt
import json
from pathlib import Path
from typing import Any

import torch

from tinyserve.bench import num_requests_for, percentile, synthetic_prompts
from tinyserve.client import StreamResult, closed_loop, running_server
from tinyserve.config import ModelConfig
from tinyserve.model import DECODE_ATTENTIONS, PREFILL_ATTENTIONS


def summarise(results: list[StreamResult], seconds: float, concurrency: int) -> dict[str, Any]:
    ttft = [r.ttft * 1e3 for r in results]
    tpot = [x * 1e3 for r in results if (x := r.tpot) is not None]
    tokens = sum(r.completion_tokens for r in results)
    return {
        "concurrency": concurrency,
        "num_requests": len(results),
        "output_tokens": tokens,
        "seconds": seconds,
        "output_tok_per_s": tokens / seconds,
        "ttft_p50_ms": percentile(ttft, 50),
        "ttft_p99_ms": percentile(ttft, 99),
        "tpot_p50_ms": percentile(tpot, 50) if tpot else None,
        "tpot_p99_ms": percentile(tpot, 99) if tpot else None,
        "finish_reasons": sorted({str(r.finish_reason) for r in results}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/Qwen3-0.6B"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attention", choices=DECODE_ATTENTIONS, default="torch")
    parser.add_argument("--prefill-attention", choices=PREFILL_ATTENTIONS, default="per_seq")
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--chunked-prefill", action="store_true")
    parser.add_argument("--max-batch-tokens", type=int, default=2048)
    parser.add_argument("--concurrency", default="1,8,32,64")
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--kv-gib", type=float, default=2.0)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    levels = [int(c) for c in args.concurrency.split(",")]
    milestone = "m8" if args.cuda_graphs else "m6"
    milestone = "m9" if args.chunked_prefill else milestone
    out = args.out or Path(f"results/{milestone}_api_bench_{args.device.split(':')[0]}.json")

    cfg = ModelConfig.from_file(args.model_dir / "config.json")
    most = max(num_requests_for(c, args.num_requests) for c in levels)
    prompts = synthetic_prompts(most, args.input_len, cfg.vocab_size, args.seed)
    serve_args = [
        "--model-dir", str(args.model_dir), "--device", args.device,
        "--attention", args.attention, "--kv-gib", str(args.kv_gib),
        "--max-num-seqs", str(max(levels)), "--ignore-eos",
        "--prefill-attention", args.prefill_attention,
        "--max-batch-tokens", str(args.max_batch_tokens),
    ] + (["--cuda-graphs"] if args.cuda_graphs else []) + (
        ["--chunked-prefill"] if args.chunked_prefill else [])  # fmt: skip

    rows = []
    with running_server(serve_args, args.port) as url:
        asyncio.run(closed_loop(url, prompts[:2], 4, 2))  # warm-up, not reported
        for c in levels:
            n = num_requests_for(c, args.num_requests)
            results, secs = asyncio.run(closed_loop(url, prompts[:n], args.output_len, c))
            row = summarise(results, secs, c)
            rows.append(row)
            tpot = "-" if row["tpot_p50_ms"] is None else (
                f"{row['tpot_p50_ms']:.1f} / {row['tpot_p99_ms']:.1f}")  # fmt: skip
            print(f"HTTP concurrency {c:>3}: {row['output_tok_per_s']:8.1f} tok/s | "
                  f"TTFT p50/p99 {row['ttft_p50_ms']:.0f} / {row['ttft_p99_ms']:.0f} ms | "
                  f"TPOT p50/p99 {tpot} ms | finish {row['finish_reasons']}",
                  flush=True)  # fmt: skip

    hw = (
        f"{torch.cuda.get_device_name()}, CUDA {torch.version.cuda}"
        if args.device != "cpu"
        else "CPU"
    )
    report = {
        "date": dt.date.today().isoformat(),
        "hardware": hw,
        "torch": torch.__version__,
        "dtype": "float16" if args.device != "cpu" else "float32",
        "attention": args.attention,
        "prefill_attention": args.prefill_attention,
        "cuda_graphs": args.cuda_graphs,
        "chunked_prefill": args.chunked_prefill,
        "max_batch_tokens": args.max_batch_tokens,
        "transport": "HTTP /v1/completions, SSE stream, token-id prompts, EOS ignored",
        "args": {k: str(v) for k, v in vars(args).items()},
        "results": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
