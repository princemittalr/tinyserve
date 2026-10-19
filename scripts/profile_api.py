"""M7 step 2: why does time to first token grow over HTTP under load?

M6 measured TTFT p99 5.6 s over HTTP at 32 streams vs 0.6 s for the engine alone. The
guess, not yet tested: the engine thread and the event loop share one interpreter (the
GIL), so the loop runs only between the engine's Python work and new requests reach the
engine late. This runs the M6 workload (bench_api.py's prompts, closed loop, streamed)
against a server started with --trace-requests and splits every request's TTFT where it
crossed between processes and threads (tinyserve/profiling.py: ``ttft_segments``):

    to_handler | parse | to_engine | engine (queue + prefill) | to_loop | to_client

It also reports the event loop's lag (how late a 5 ms sleep wakes up), the engine's
own step times while serving, and a TTFT to the first TOKEN (client send -> the event
loop has the first token), next to the client's TTFT to the first TEXT chunk: the API
sends no chunk while a token's text is empty or half a character (step 3 added this
after the T4 run showed a few requests whose first chunk came only at the very end).
``--switch-intervals 5,0.5`` repeats the runs with a shorter GIL switch interval: if GIL
contention were the cause, the loop-side segments would shrink (step 2 on the T4: they
did not).

M10: ``--prefill-attention``, ``--cuda-graphs`` and ``--fused-kernels`` profile another
setup than M7's (still the default), such as the M9 GPU default:
    python scripts/profile_api.py --device cuda --attention triton_split \
        --prefill-attention batched --cuda-graphs --fused-kernels --concurrency 32

Usage: python scripts/profile_api.py --device cuda --attention triton
"""

import argparse
import asyncio
import datetime as dt
import json
from pathlib import Path
from typing import Any

import httpx
import torch

from tinyserve.bench import num_requests_for, percentile, synthetic_prompts
from tinyserve.client import closed_loop, running_server
from tinyserve.config import ModelConfig
from tinyserve.engine import StepRecord
from tinyserve.model import DECODE_ATTENTIONS, PREFILL_ATTENTIONS
from tinyserve.profiling import TTFT_SEGMENTS, summarize_steps, ttft_segments


def stats(xs: list[float]) -> dict[str, float]:
    return {"p50": percentile(xs, 50), "p99": percentile(xs, 99), "mean": sum(xs) / len(xs),
            "max": max(xs)}  # fmt: skip


def analyse(results: list[Any], trace: dict[str, Any], seconds: float, c: int) -> dict[str, Any]:
    by_id = {r["tag"]: r for r in trace["requests"]}
    segs = [ttft_segments(r.sent, r.chunk_times[0], by_id[r.id]) for r in results]
    ttft = [r.ttft * 1e3 for r in results]
    waited = [by_id[r.id]["first_token_step"] - by_id[r.id]["added_at_step"] for r in results]
    steps = [StepRecord(**{k: v for k, v in s.items() if k != "total_ms"})
             for s in trace["steps"]]  # fmt: skip
    lag = trace["loop_lag_ms"]
    worst = sorted(range(len(results)), key=lambda i: ttft[i], reverse=True)[:3]
    # The first token, as the server's event loop had it, on the client's clock.
    token_ttft = [(by_id[r.id]["first_seen"] - r.sent) * 1e3 for r in results]
    late = [i for i, s in enumerate(segs) if s["to_client"] > 1000]
    return {
        "concurrency": c,
        "num_requests": len(results),
        "output_tok_per_s": sum(r.completion_tokens for r in results) / seconds,
        "ttft_ms": stats(ttft),
        "ttft_first_token_ms": stats(token_ttft),
        "first_text_over_1s_after_first_token": len(late),
        "segments_ms": {name: stats([s[name] for s in segs]) for name in TTFT_SEGMENTS},
        "steps_to_first_token": stats([float(w) for w in waited]),
        "loop_lag_ms": stats(lag) | {"over_10ms": sum(x > 10 for x in lag) / len(lag)},
        "engine_decode_steps": summarize_steps([s for s in steps if s.num_prefill_seqs == 0]),
        "engine_prefill_steps": summarize_steps([s for s in steps if s.num_prefill_seqs > 0]),
        "slowest_requests": [
            {
                "ttft_ms": ttft[i],
                **segs[i],
                "chunks": len(results[i].chunk_times),
                "text_head": results[i].text[:40],
            }
            for i in worst
        ],
    }


def show(row: dict[str, Any], interval: str) -> None:
    t, seg, lag = row["ttft_ms"], row["segments_ms"], row["loop_lag_ms"]
    d = row["engine_decode_steps"]
    print(f"\nGIL switch {interval} ms, C={row['concurrency']}: "
          f"{row['output_tok_per_s']:.1f} tok/s, TTFT p50/p99 {t['p50']:.0f} / {t['p99']:.0f} ms"
          f" | engine decode step {d.get('total_ms', 0):.1f} ms | loop lag p50/p99/max "
          f"{lag['p50']:.1f} / {lag['p99']:.1f} / {lag['max']:.0f} ms, "
          f"{lag['over_10ms']:.0%} of probes over 10 ms | steps from add to first token "
          f"p50/p99 {row['steps_to_first_token']['p50']:.0f} / "
          f"{row['steps_to_first_token']['p99']:.0f}")  # fmt: skip
    print("   segment ms: " + " | ".join(
        f"{n} {seg[n]['p50']:.1f} / {seg[n]['p99']:.1f}" for n in TTFT_SEGMENTS
    ) + "   (p50 / p99)")  # fmt: skip
    tt = row["ttft_first_token_ms"]
    print(f"   TTFT to the first token p50/p99 {tt['p50']:.0f} / {tt['p99']:.0f} ms; "
          f"{row['first_text_over_1s_after_first_token']} of {row['num_requests']} requests "
          f"got their first text over 1 s after their first token")  # fmt: skip
    for w in row["slowest_requests"]:
        print(f"   slow: TTFT {w['ttft_ms']:.0f} ms = " +
              " + ".join(f"{n} {w[n]:.0f}" for n in TTFT_SEGMENTS) +
              f" | {w['chunks']} chunks, text starts {w['text_head']!r}")  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/Qwen3-0.6B"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attention", choices=DECODE_ATTENTIONS, default="torch")
    parser.add_argument("--prefill-attention", choices=PREFILL_ATTENTIONS, default="per_seq")
    parser.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fused-kernels", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--concurrency", default="32,64")
    parser.add_argument("--switch-intervals", default="5",
                        help="GIL switch intervals to run, ms (Python's default is 5)")  # fmt: skip
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--kv-gib", type=float, default=2.0)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    levels = [int(c) for c in args.concurrency.split(",")]
    m7 = args.prefill_attention == "per_seq" and not (args.cuda_graphs or args.fused_kernels)
    name = "m7_api_profile" if m7 else "m10_api_profile"
    out = args.out or Path(f"results/{name}_{args.device.split(':')[0]}.json")

    cfg = ModelConfig.from_file(args.model_dir / "config.json")
    most = max(num_requests_for(c, args.num_requests) for c in levels)
    prompts = synthetic_prompts(most, args.input_len, cfg.vocab_size, args.seed)

    runs = []
    for interval in args.switch_intervals.split(","):
        serve_args = [
            "--model-dir", str(args.model_dir), "--device", args.device,
            "--attention", args.attention, "--kv-gib", str(args.kv_gib),
            "--max-num-seqs", str(max(levels)), "--ignore-eos", "--trace-requests",
            "--switch-interval-ms", interval,
            # Explicit both ways: the server's own defaults differ by device (M9).
            "--prefill-attention", args.prefill_attention,
            "--cuda-graphs" if args.cuda_graphs else "--no-cuda-graphs",
            "--fused-kernels" if args.fused_kernels else "--no-fused-kernels",
        ]  # fmt: skip
        with running_server(serve_args, args.port) as url:
            asyncio.run(closed_loop(url, prompts[:2], 4, 2))  # warm-up, not reported
            httpx.get(f"{url}/debug/trace").raise_for_status()  # drop the warm-up's trace
            for c in levels:
                n = num_requests_for(c, args.num_requests)
                results, secs = asyncio.run(closed_loop(url, prompts[:n], args.output_len, c))
                trace = httpx.get(f"{url}/debug/trace", timeout=60).json()
                row = analyse(results, trace, secs, c)
                show(row, interval)
                runs.append({"switch_interval_ms": float(interval), **row})

    report = {
        "date": dt.date.today().isoformat(),
        "hardware": torch.cuda.get_device_name() if args.device != "cpu" else "CPU",
        "torch": torch.__version__,
        "attention": args.attention,
        "prefill_attention": args.prefill_attention,
        "cuda_graphs": args.cuda_graphs,
        "fused_kernels": args.fused_kernels,
        "transport": "HTTP /v1/completions, SSE stream, token-id prompts, EOS ignored",
        "args": {k: str(v) for k, v in vars(args).items()},
        "runs": runs,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
