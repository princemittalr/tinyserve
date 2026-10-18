"""Baseline for the M3 benchmark: Hugging Face ``generate()`` on the same workload.

HF ``generate()`` has no scheduler, so "concurrency C" here means static batching: the
prompts are cut into batches of C, and each batch runs to the end before the next one
starts. All prompts have the same length, so no padding is needed, and every request
generates exactly ``--output-len`` tokens (``min_new_tokens``), like scripts/benchmark.py.
Only output tok/s is reported: generate() returns all tokens at once, so per-token
times (TTFT, TPOT) are not observable from outside.

Usage (T4): python scripts/bench_hf.py --device cuda
"""

import argparse
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from tinyserve.bench import num_requests_for, synthetic_prompts
from tinyserve.config import ModelConfig


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/Qwen3-0.6B"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16"], default=None)
    parser.add_argument("--concurrency", default="1,8,32,64")
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype or ("float16" if args.device != "cpu" else "float32"))
    levels = [int(c) for c in args.concurrency.split(",")]
    out = args.out or Path(f"results/m3_bench_hf_{args.device.split(':')[0]}.json")

    cfg = ModelConfig.from_file(args.model_dir / "config.json")
    hf: Any = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=dtype).to(args.device)
    hf.eval()
    most = max(num_requests_for(c, args.num_requests) for c in levels)
    prompts = synthetic_prompts(most, args.input_len, cfg.vocab_size, args.seed)

    def generate(batch: list[list[int]]) -> int:
        x = torch.tensor(batch, device=args.device)
        y = hf.generate(
            x,
            attention_mask=torch.ones_like(x),
            max_new_tokens=args.output_len,
            min_new_tokens=args.output_len,
            do_sample=False,
            pad_token_id=0,
        )
        return int(y.shape[0] * (y.shape[1] - x.shape[1]))

    generate(prompts[:2])  # warm-up, not reported
    results = []
    for c in levels:
        todo = prompts[: num_requests_for(c, args.num_requests)]
        sync(args.device)
        start = time.perf_counter()
        tokens = sum(generate(todo[i : i + c]) for i in range(0, len(todo), c))
        sync(args.device)
        secs = time.perf_counter() - start
        results.append({"concurrency": c, "num_requests": len(todo), "output_tokens": tokens,
                        "seconds": secs, "output_tok_per_s": tokens / secs})  # fmt: skip
        print(f"HF generate(), batch {c:>3}: {tokens / secs:8.1f} tok/s ({secs:.1f} s)")

    report = {
        "date": dt.date.today().isoformat(),
        "hardware": torch.cuda.get_device_name() if args.device != "cpu" else "CPU",
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "dtype": str(dtype).removeprefix("torch."),
        "attn_implementation": hf.config._attn_implementation,
        "args": {k: str(v) for k, v in vars(args).items()},
        "results": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
