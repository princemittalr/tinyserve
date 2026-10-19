"""M10 step 2: was the decode kernel slower in F than in A, or was the GPU slower?

M9 run 2 profiled C=32 decode steps of A (M8: grouped decode kernel, batched prefill,
graphs) and then F (the M9 default: split-K, batched prefill, fused kernels, graphs) in
two processes, one after the other: the decode kernel took 4.4 ms per step in A and 5.2
in F. At batch 32 both launch the same grouped kernel on the same shapes (split-K picks
1 split from batch 10 up), so the code alone does not explain it. A T4 is capped at 70 W
and lowers its clock when it is hot or busy, and F, with fewer gaps between kernels,
keeps the GPU busier.

This runs both in one process, alternating A F A F ... for ``--rounds``. Each round
captures a graph for batch C, brings C requests to the decode phase, profiles
``--steps`` decode steps while ``GpuSampler`` records the SM clock, power and
temperature, then times the grouped decode kernel alone on that round's real inputs
(CUDA events). If the kernel's time follows the clock and A's and F's isolated times
agree, the difference was the GPU's state, not the code.

M10 step 4, after run 1: the difference is real and does not follow the clock. In three
alternating rounds the kernel took 4.72-4.76 ms per step in A at a mean SM clock of
1,170-1,242 MHz and 5.68-5.75 ms in F at 1,431-1,438 MHz. So this now also runs
A_fused (A plus the fused kernels) and F_unfused (F without them): A and F differ only in
split-K, which launches the same grouped kernel at batch 32, and the fused kernels, which
write q and the cache before the decode kernel reads them. The isolated time (A 11.7 ms,
F 4.4 ms in run 1, far apart while the in-step times were close) followed 50 calls right
after the profiler stopped, with no sampler running; it now has ``--warmup`` untimed
calls, ``--iters`` timed ones and its own clock samples, and prints the inputs it used
(table width and mean context) so a difference there shows. The memory clock and the
reasons the driver held the clock down are recorded too (``GpuSampler``), and the GEMM
time counts cuBLAS GEMV kernels (``is_gemm``).

Usage (T4): python scripts/decode_ab.py
"""

import argparse
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

from tinyserve.bench import synthetic_prompts
from tinyserve.config import ModelConfig
from tinyserve.cuda_graphs import DecodeGraphs
from tinyserve.engine import Engine
from tinyserve.kv_cache import KVCache
from tinyserve.model import Qwen3
from tinyserve.profiling import (
    GpuSampler,
    fill_to_decode,
    gpu_time_ms,
    is_gemm,
    loop_ms,
    profile_summary,
)
from tinyserve.triton_attention import launch_paged_decode_grouped
from tinyserve.weights import load_weights

CONFIGS: dict[str, dict[str, Any]] = {
    "A": {"decode_attention": "triton_grouped"},
    "F": {"decode_attention": "triton_split", "fused_kernels": True},
    "A_fused": {"decode_attention": "triton_grouped", "fused_kernels": True},
    "F_unfused": {"decode_attention": "triton_split"},
}  # all with batched prefill and CUDA graphs


def kernel_alone(engine: Engine, graphs: DecodeGraphs, c: int, args: argparse.Namespace
                 ) -> dict[str, Any]:  # fmt: skip
    """Mean time of one grouped-kernel launch on layer 0 with this engine's current
    block tables and lengths (the graph's input buffers after the last step), the clock
    while it ran, and what those inputs were."""
    cache, cfg = engine.cache, engine.model.cfg
    q = torch.randn(c, cfg.num_heads, cfg.head_dim, dtype=cache.k.dtype, device=cache.k.device)
    tables, lens = graphs.inputs.block_tables[:c], graphs.inputs.context_lens[:c]

    def launch() -> torch.Tensor:
        return launch_paged_decode_grouped(q, cache.k[0], cache.v[0], tables, lens,
                                           cfg.head_dim**-0.5, cache.block_size)  # fmt: skip

    with GpuSampler() as gpu:
        ms = loop_ms(launch, cache.k.device, args.iters, warmup=args.warmup)
    return {"ms_x_layers": ms * cfg.num_layers, "table_width": tables.shape[1],
            "context_mean": lens.float().mean().item(), "gpu": gpu.summary()}  # fmt: skip


def one_round(name: str, model: Qwen3, args: argparse.Namespace, prompts: list[list[int]],
              dtype: torch.dtype) -> dict[str, Any]:  # fmt: skip
    cfg, c, dev = model.cfg, args.concurrency, torch.device("cuda")
    per_seq = -(-(args.input_len + args.output_len) // args.block_size)
    cache = KVCache(cfg, per_seq * c, args.block_size, dtype, dev)
    graphs = DecodeGraphs(model, cache, c, buckets=[c])
    e = Engine(model, cache, c, 2048, decode_graphs=graphs)
    fill_to_decode(e, prompts, args.output_len)
    for _ in range(2):  # warm-up outside the profiler
        e.step()
    k = args.steps
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with GpuSampler() as gpu, profile(activities=activities, acc_events=True) as prof:
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(k):
            e.step()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - start) * 1e3
    s = profile_summary(prof, wall)
    alone = kernel_alone(e, graphs, c, args)
    row = {
        "config": name,
        "step_ms": wall / k,
        "gpu_busy_ms": s["gpu_kernel_ms"] / k,
        "gemm_ms": gpu_time_ms(prof, is_gemm) / k,
        "decode_kernel_ms": gpu_time_ms(prof, lambda n: "paged_decode" in n or "merge_splits" in n)
        / k,
        "kernel_alone": alone,
        "gpu": gpu.summary(),
    }  # fmt: skip
    print(f"{name}: step {row['step_ms']:.2f} ms, GPU busy {row['gpu_busy_ms']:.2f}, GEMMs "
          f"{row['gemm_ms']:.2f}, decode kernel {row['decode_kernel_ms']:.2f} in the step | "
          f"{GpuSampler.describe(row['gpu'])}\n    alone: {alone['ms_x_layers']:.2f} ms "
          f"(x{cfg.num_layers} layers; table width {alone['table_width']}, mean context "
          f"{alone['context_mean']:.0f}) | {GpuSampler.describe(alone['gpu'])}",
          flush=True)  # fmt: skip
    del graphs, e, cache  # free the graph pool and the cache before the next round
    torch.cuda.empty_cache()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/Qwen3-0.6B"))
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--configs", default="A,F,A_fused,F_unfused",
                        help=f"from {', '.join(CONFIGS)}")  # fmt: skip
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/m10_decode_ab_cuda.json"))
    args = parser.parse_args()
    dtype = torch.float16
    cfg = ModelConfig.from_file(args.model_dir / "config.json")
    w = load_weights(args.model_dir, cfg, dtype, "cuda")
    models = {name: Qwen3(cfg, w, prefill_attention="batched", **CONFIGS[name])
              for name in args.configs.split(",")}  # the same weights  # fmt: skip
    prompts = synthetic_prompts(args.concurrency, args.input_len, cfg.vocab_size, args.seed)
    rows = []
    with torch.inference_mode():
        for _ in range(args.rounds):
            for name, model in models.items():
                rows.append(one_round(name, model, args, prompts, dtype))
    report = {"date": dt.date.today().isoformat(), "hardware": torch.cuda.get_device_name(),
              "torch": torch.__version__, "args": {k: str(v) for k, v in vars(args).items()},
              "rounds": rows}  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
