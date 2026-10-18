"""M7 step 3: vLLM on the T4, on the same workload as benchmark.py and bench_hf.py.

The same seeded 128-token random prompts (``tinyserve.bench.synthetic_prompts``), 128
greedy output tokens each with EOS ignored, ``num_requests_for(C)`` requests per level.
All of a level's prompts go to one ``LLM.generate`` call with ``max_num_seqs=C``: vLLM's
scheduler then keeps at most C sequences running and admits the next one as soon as one
finishes, the same closed loop at concurrency C that benchmark.py runs. tok/s is output
tokens / the wall time of that call; loading the model and capturing CUDA graphs are not
counted. Only throughput is compared, as with HF: offline ``generate`` gives no
per-token times.

Two modes: "graphs" (vLLM's default: CUDA graphs and torch.compile for the decode
step) and "eager" (``enforce_eager=True``: every kernel launched from Python, like
TinyServe today). Each (C, mode) runs in its own process, because vLLM does not give
the GPU memory back when an ``LLM`` is dropped; ``--one C MODE`` is that child.

vLLM is not a TinyServe dependency. Installing it replaces Kaggle's torch, so run this
last in a session. If it does not install or run on a T4, the error is recorded.

Usage: python scripts/bench_vllm.py --model-dir checkpoints/Qwen3-0.6B
"""

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tinyserve.bench import num_requests_for, synthetic_prompts
from tinyserve.config import ModelConfig


def run_one(args: argparse.Namespace, c: int, mode: str) -> dict[str, Any]:
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    cfg = ModelConfig.from_file(args.model_dir / "config.json")
    n = num_requests_for(c, args.num_requests)
    prompts = synthetic_prompts(n, args.input_len, cfg.vocab_size, args.seed)
    llm = LLM(model=str(args.model_dir), dtype="float16", max_num_seqs=c,
              max_model_len=args.input_len + args.output_len, seed=args.seed,
              gpu_memory_utilization=args.gpu_memory_utilization,
              enforce_eager=mode == "eager")  # fmt: skip
    greedy = SamplingParams(temperature=0.0, max_tokens=args.output_len, ignore_eos=True)
    inputs = [TokensPrompt(prompt_token_ids=p) for p in prompts]
    warm = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
    llm.generate(inputs[:2], warm, use_tqdm=False)  # warm-up, not reported
    torch.cuda.synchronize()
    start = time.perf_counter()
    outs = llm.generate(inputs, greedy, use_tqdm=False)
    seconds = time.perf_counter() - start
    tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    return {"concurrency": c, "mode": mode, "num_requests": n, "output_tokens": tokens,
            "seconds": seconds, "output_tok_per_s": tokens / seconds,
            "vllm": vllm.__version__, "torch": torch.__version__,
            "hardware": torch.cuda.get_device_name()}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/Qwen3-0.6B"))
    parser.add_argument("--concurrency", default="1,8,32,64")
    parser.add_argument("--modes", default="graphs,eager")
    parser.add_argument("--eager-concurrency", default="32",
                        help="levels to run in eager mode too (it is the slow one)")  # fmt: skip
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--one", nargs=2, metavar=("C", "MODE"), default=None)
    parser.add_argument("--out", type=Path, default=Path("results/m7_vllm_bench_cuda.json"))
    args = parser.parse_args()
    if args.one:
        print("RESULT " + json.dumps(run_one(args, int(args.one[0]), args.one[1])), flush=True)
        return

    eager_levels = {int(c) for c in args.eager_concurrency.split(",")}
    jobs = [(int(c), m) for m in args.modes.split(",") for c in args.concurrency.split(",")
            if m != "eager" or int(c) in eager_levels]  # fmt: skip
    rows: list[dict[str, Any]] = []
    for c, mode in jobs:
        cmd = [sys.executable, __file__, "--one", str(c), mode, "--model-dir", str(args.model_dir),
               "--input-len", str(args.input_len), "--output-len", str(args.output_len),
               "--gpu-memory-utilization", str(args.gpu_memory_utilization),
               "--seed", str(args.seed)]  # fmt: skip
        if args.num_requests is not None:
            cmd += ["--num-requests", str(args.num_requests)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        found = [line for line in p.stdout.splitlines() if line.startswith("RESULT ")]
        if p.returncode == 0 and found:
            r = json.loads(found[-1].removeprefix("RESULT "))
            print(f"vLLM {r['vllm']} {mode:>6} C={c:>3}: {r['output_tok_per_s']:8.1f} tok/s "
                  f"({r['num_requests']} requests, {r['seconds']:.1f} s)", flush=True)  # fmt: skip
        else:
            tail = (p.stderr or p.stdout).strip().splitlines()[-15:]
            r = {"concurrency": c, "mode": mode, "error": "\n".join(tail)}
            print(f"vLLM {mode} C={c} FAILED (exit {p.returncode}):\n" + "\n".join(tail))
        rows.append(r)

    report = {"date": dt.date.today().isoformat(),
              "workload": f"{args.input_len}-token random prompts, {args.output_len} greedy "
                          "output tokens, EOS ignored, max_num_seqs = C",
              "args": {k: str(v) for k, v in vars(args).items()}, "results": rows}  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
