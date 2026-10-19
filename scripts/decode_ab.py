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
    loop_ms,
    profile_summary,
)
from tinyserve.triton_attention import launch_paged_decode_grouped
from tinyserve.weights import load_weights


def kernel_alone_ms(engine: Engine, graphs: DecodeGraphs, c: int, iters: int) -> float:
    """Mean time of one grouped-kernel launch on layer 0 with this engine's current
    block tables and lengths (the graph's input buffers after the last step)."""
    cache, cfg = engine.cache, engine.model.cfg
    q = torch.randn(c, cfg.num_heads, cfg.head_dim, dtype=cache.k.dtype, device=cache.k.device)
    tables, lens = graphs.inputs.block_tables[:c], graphs.inputs.context_lens[:c]

    def launch() -> torch.Tensor:
        return launch_paged_decode_grouped(q, cache.k[0], cache.v[0], tables, lens,
                                           cfg.head_dim**-0.5, cache.block_size)  # fmt: skip

    return loop_ms(launch, cache.k.device, iters)


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
    alone = kernel_alone_ms(e, graphs, c, args.iters) * cfg.num_layers
    row = {
        "config": name,
        "step_ms": wall / k,
        "gpu_busy_ms": s["gpu_kernel_ms"] / k,
        "gemm_ms": gpu_time_ms(prof, lambda n: "gemm" in n.lower()) / k,
        "decode_kernel_ms": gpu_time_ms(prof, lambda n: "paged_decode" in n or "merge_splits" in n)
        / k,
        "kernel_alone_ms_x_layers": alone,
        "gpu": gpu.summary(),
    }  # fmt: skip
    g = row["gpu"]
    clock = "no nvidia-smi"
    if g:
        clock = (f"SM {g['sm_mhz_mean']:.0f} MHz (min {g['sm_mhz_min']:.0f}), "
                 f"{g['power_w_mean']:.1f} W, {g['temp_c_max']:.0f} C")  # fmt: skip
    print(f"{name}: step {row['step_ms']:.2f} ms, GPU busy {row['gpu_busy_ms']:.2f}, GEMMs "
          f"{row['gemm_ms']:.2f}, decode kernel {row['decode_kernel_ms']:.2f} in the step / "
          f"{alone:.2f} alone (x{cfg.num_layers} layers) | {clock}", flush=True)  # fmt: skip
    del graphs, e, cache  # free the graph pool and the cache before the next round
    torch.cuda.empty_cache()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/Qwen3-0.6B"))
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/m10_decode_ab_cuda.json"))
    args = parser.parse_args()
    dtype = torch.float16
    cfg = ModelConfig.from_file(args.model_dir / "config.json")
    w = load_weights(args.model_dir, cfg, dtype, "cuda")
    models = {  # the same weights, two configurations
        "A": Qwen3(cfg, w, decode_attention="triton_grouped", prefill_attention="batched"),
        "F": Qwen3(cfg, w, decode_attention="triton_split", prefill_attention="batched",
                   fused_kernels=True),
    }  # fmt: skip
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
