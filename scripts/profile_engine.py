"""M7 step 1: where a whole engine step's time goes, measured against the roofline.

--workload m3 (default): the M3/M4 benchmark (128-token random prompts, 128 output tokens,
    closed loop at each --concurrency) with ``Engine(trace_steps=True)``. For the steps in
    which every sequence decodes, it prints the mean time of each phase (schedule,
    build_batch, forward, sample + copy to host, update) and the step's roofline floor on
    a T4: (weights + k/v read) / 320 GB/s (see tinyserve/roofline.py). Then
    --profile-steps steady decode steps at --profile-concurrency run under torch.profiler:
    op calls, kernel launches and GPU busy time per step, and the bandwidth the GEMMs and
    the Triton kernel reach. --trace also writes a Chrome trace (ui.perfetto.dev).

--workload prefix: the M5 shared-prefix workload at --concurrency (default 32), prefix
    cache off and on, with the step trace: how many prefills each step carries and what
    those steps cost. Then one forward_batch over k cache hits at once (--prefix-len
    tokens cached, --suffix-len new each), with and without attention, to see what the
    per-sequence PyTorch prefill attention costs.

--cuda-graphs (M8): decode steps replay CUDA graphs (captured once for the run); the
    decode rows' "build" is then filling the graph's input buffers and "fwd" the replay.

Tracing waits for the GPU after every forward, so its tok/s are a little below
benchmark.py's; use benchmark.py for throughput and this for the breakdown.

Usage:
    python scripts/profile_engine.py --device cuda --attention triton           # T4
    python scripts/profile_engine.py --device cuda --attention triton --workload prefix
    python scripts/profile_engine.py --device cuda --attention triton --cuda-graphs  # M8
    python scripts/profile_engine.py --concurrency 1,4 --num-requests 8 --input-len 32 \\
        --output-len 16 --profile-concurrency 4 --profile-steps 4               # laptop
Laptop CPU numbers only show that the script works.
"""

import argparse
import datetime as dt
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

from tinyserve.batch import Batch, build_batch
from tinyserve.bench import (
    num_requests_for,
    run_closed_loop,
    shared_prefix_prompts,
    synthetic_prompts,
)
from tinyserve.block_allocator import BlockAllocator
from tinyserve.config import ModelConfig
from tinyserve.cuda_graphs import DecodeGraphs
from tinyserve.engine import Engine, StepRecord
from tinyserve.kv_cache import KVCache
from tinyserve.model import DECODE_ATTENTIONS, Qwen3
from tinyserve.profiling import (
    attention_replaced,
    fill_to_decode,
    gpu_time_ms,
    prefill_items,
    profile_summary,
    summarize_steps,
    time_ms,
)
from tinyserve.roofline import T4, decode_step_cost, kv_bytes_per_token, weight_bytes
from tinyserve.weights import load_weights


def hardware(device: torch.device) -> str:
    if device.type == "cuda":
        return f"{torch.cuda.get_device_name(device)}, CUDA {torch.version.cuda}"
    return f"{platform.system()} {platform.machine()} CPU"


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_m3(args: argparse.Namespace, cfg: ModelConfig, model: Qwen3, dev: torch.device,
           dtype: torch.dtype) -> dict[str, Any]:  # fmt: skip
    elem = dtype.itemsize
    levels = [int(c) for c in args.concurrency.split(",")]
    pc = args.profile_concurrency
    most = max(*(num_requests_for(c, args.num_requests) for c in levels), pc)
    prompts = synthetic_prompts(most, args.input_len, cfg.vocab_size, args.seed)
    per_seq = -(-(args.input_len + args.output_len) // args.block_size)
    cache = KVCache(cfg, per_seq * max(*levels, pc), args.block_size, dtype, dev)
    graphs = DecodeGraphs(model, cache, max(*levels, pc)) if args.cuda_graphs else None
    if graphs is not None:
        mib = graphs.capture_bytes / 2**20
        print(f"CUDA graphs: buckets {graphs.buckets}, captured in "
              f"{graphs.capture_seconds:.1f} s, {mib:.0f} MiB")  # fmt: skip

    def engine(c: int, trace: bool = False) -> Engine:
        return Engine(model, cache, c, args.max_batch_tokens, trace_steps=trace,
                      decode_graphs=graphs)  # fmt: skip

    run_closed_loop(engine(2), prompts[:2], 4, concurrency=2)  # warm-up, not reported
    print(f"{'C':>3} | {'tok/s':>7} | decode step ms: {'sched':>5} {'build':>5} "
          f"{'fwd':>6} {'sample':>6} {'update':>6} {'total':>6} | {'T4 floor':>8} | "
          f"floor/fwd | steps with prefill: n, mean ms, share of time")  # fmt: skip
    levels_out = []
    for c in levels:
        n = num_requests_for(c, args.num_requests)
        e = engine(c, trace=True)
        r = run_closed_loop(e, prompts[:n], args.output_len, concurrency=c)
        recs: list[StepRecord] = e.step_records or []
        d = summarize_steps([x for x in recs if x.num_prefill_seqs == 0])
        p = summarize_steps([x for x in recs if x.num_prefill_seqs > 0])
        # Share of the run's step time spent in steps that carry a prefill.
        all_ms = sum(x.total_ms for x in recs)
        share = sum(x.total_ms for x in recs if x.num_prefill_seqs > 0) / all_ms if recs else 0.0
        row: dict[str, Any] = {"concurrency": c, "output_tok_per_s": r.output_tok_per_s,
                               "decode_steps": d, "prefill_steps": p,
                               "prefill_time_share": share}  # fmt: skip
        line = f"{c:>3} | {r.output_tok_per_s:7.1f} | "
        if d["steps"]:
            cost = decode_step_cost(cfg, d["num_seqs"], d["context_tokens"], elem)
            floor = cost.floor_ms(T4, elem)
            row |= {"t4_floor_ms": floor, "floor_over_forward": floor / d["forward_ms"],
                    "intensity_flop_per_byte": cost.intensity,
                    "memory_bound": cost.memory_bound(T4, elem),
                    "weight_bytes": cost.weight_bytes, "kv_bytes": cost.kv_bytes}  # fmt: skip
            line += (f"{'':15} {d['schedule_ms']:5.1f} {d['build_ms']:5.1f} "
                     f"{d['forward_ms']:6.1f} {d['sample_ms']:6.1f} {d['update_ms']:6.1f} "
                     f"{d['total_ms']:6.1f} | {floor:8.2f} | "
                     f"{floor / d['forward_ms']:9.0%} | ")  # fmt: skip
        if p["steps"]:
            line += f"{p['steps']:.0f}, {p['total_ms']:.1f}, {share:.0%}"
        print(line)
        levels_out.append(row)

    # Steady decode under the profiler: every sequence holds about input_len tokens.
    k = args.profile_steps
    if args.output_len < k + 4:
        raise SystemExit(f"--output-len must be at least --profile-steps + 4 = {k + 4}")
    e = engine(pc)
    fill_to_decode(e, prompts[:pc], args.output_len)
    for _ in range(2):  # warm-up outside the profiler
        e.step()
    context = 0
    activities = [ProfilerActivity.CPU]
    if dev.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, acc_events=True) as prof:
        sync(dev)
        start = time.perf_counter()
        for _ in range(k):
            context += sum(q.table.num_tokens + 1 for q in e.scheduler.running)
            e.step()
        sync(dev)
        wall = (time.perf_counter() - start) * 1e3
    s = profile_summary(prof, wall)
    per_step = {key: s[key] / k for key in ("aten_calls", "kernel_launches", "graph_launches",
                                             "gpu_kernel_ms", "wall_ms")}  # fmt: skip
    gemm_ms = gpu_time_ms(prof, lambda name: "gemm" in name.lower()) / k
    attn_ms = gpu_time_ms(prof, lambda name: "paged_decode" in name) / k
    w_bytes = weight_bytes(cfg, elem)
    kv_min = context / k * kv_bytes_per_token(cfg, elem)
    # The M4 kernel loads each k/v once per query head; the M8 grouped one once per kv head.
    group = cfg.num_heads // cfg.num_kv_heads if args.attention == "triton" else 1
    prof_out: dict[str, Any] = {
        "concurrency": pc, "steps": k, "gpu_busy": s["gpu_busy"], **per_step,
        "gemm_ms": gemm_ms, "gemm_gbps": w_bytes / gemm_ms / 1e6 if gemm_ms else None,
        "decode_kernel_ms": attn_ms,
        "decode_kernel_gbps_min_bytes": kv_min / attn_ms / 1e6 if attn_ms else None,
        "decode_kernel_bytes_issued_factor": group,
    }  # fmt: skip
    print(f"\nprofiled {k} decode steps at C={pc} (slower under the profiler): per step "
          f"{per_step['wall_ms']:.1f} ms wall, {per_step['aten_calls']:,.0f} aten calls, "
          f"{per_step['kernel_launches']:,.0f} kernel launches, "
          f"{per_step['graph_launches']:.0f} graph replays, GPU busy "
          f"{per_step['gpu_kernel_ms']:.1f} ms = {s['gpu_busy']:.0%}")  # fmt: skip
    if gemm_ms:
        print(f"GEMMs: {gemm_ms:.1f} ms/step for {w_bytes / 1e9:.2f} GB of weights = "
              f"{w_bytes / gemm_ms / 1e6:.0f} GB/s (T4 spec {T4.mem_gbps:.0f})")  # fmt: skip
    if attn_ms:
        how = "one program per query head" if group > 1 else "one program per kv head"
        print(f"decode kernel: {attn_ms:.1f} ms/step for {kv_min / 1e9:.3f} GB of k/v = "
              f"{kv_min / attn_ms / 1e6:.0f} GB/s counting each k/v once; it loads each "
              f"{group}x ({how}), {group * kv_min / attn_ms / 1e6:.0f} GB/s as issued")  # fmt: skip
    sort = "self_device_time_total" if dev.type == "cuda" else "self_cpu_time_total"
    table = prof.key_averages().table(sort_by=sort, row_limit=15)
    print(table)
    if args.trace:
        prof.export_chrome_trace(str(args.trace))
        print(f"wrote {args.trace}")
    return {"levels": levels_out, "profile": prof_out, "top_ops": table}


def slowest(recs: list[StepRecord], n: int = 5) -> list[dict[str, float]]:
    top = sorted(recs, key=lambda r: r.total_ms, reverse=True)[:n]
    return [{"num_seqs": r.num_seqs, "num_prefill_seqs": r.num_prefill_seqs,
             "num_tokens": r.num_tokens, "context_tokens": r.context_tokens,
             "forward_ms": r.forward_ms, "total_ms": r.total_ms} for r in top]  # fmt: skip


def run_prefix(args: argparse.Namespace, cfg: ModelConfig, model: Qwen3, dev: torch.device,
               dtype: torch.dtype) -> dict[str, Any]:  # fmt: skip
    c = int(args.concurrency)
    n = num_requests_for(c, args.num_requests)
    prompts = shared_prefix_prompts(n, args.num_prefixes, args.prefix_len, args.suffix_len,
                                    cfg.vocab_size, args.seed)  # fmt: skip
    prompt_len = args.prefix_len + args.suffix_len
    per_seq = -(-(prompt_len + args.output_len) // args.block_size)
    cache = KVCache(cfg, per_seq * c, args.block_size, dtype, dev)

    def engine(on: bool, trace: bool = False) -> Engine:
        return Engine(model, cache, c, args.max_batch_tokens, prefix_cache=on,
                      trace_steps=trace)  # fmt: skip

    run_closed_loop(engine(True), prompts[:2], 4, concurrency=2)  # warm-up
    runs: dict[str, Any] = {}
    for on in (False, True):
        e = engine(on, trace=True)
        r = run_closed_loop(e, prompts, args.output_len, concurrency=c)
        recs: list[StepRecord] = e.step_records or []
        pre = [x for x in recs if x.num_prefill_seqs > 0]
        d, p = summarize_steps([x for x in recs if x.num_prefill_seqs == 0]), summarize_steps(pre)
        name = "cache_on" if on else "cache_off"
        runs[name] = {"bench": r.as_dict(), "decode_steps": d, "prefill_steps": p,
                      "max_prefills_in_a_step": max((x.num_prefill_seqs for x in pre), default=0),
                      "seconds_in_prefill_steps": sum(x.total_ms for x in pre) / 1e3,
                      "slowest_steps": slowest(recs)}  # fmt: skip
        print(f"C={c} cache {'on ' if on else 'off'}: {r.output_tok_per_s:6.1f} tok/s, TTFT "
              f"p50/p99 {r.ttft_p50_ms:.0f} / {r.ttft_p99_ms:.0f} ms, hit "
              f"{r.prefix_hit_rate:.0%} | {len(recs)} steps, {len(pre)} with prefill: "
              f"mean {p.get('num_prefill_seqs', 0):.1f} prefills (max "
              f"{runs[name]['max_prefills_in_a_step']}), {p.get('total_ms', 0):.0f} ms each, "
              f"{runs[name]['seconds_in_prefill_steps']:.1f} s total | decode-only steps "
              f"{d.get('total_ms', 0):.1f} ms")  # fmt: skip
        for x in runs[name]["slowest_steps"]:
            print(f"    slow step: {x['num_seqs']} seqs, {x['num_prefill_seqs']} prefilling, "
                  f"{x['num_tokens']} tokens in, {x['context_tokens']} context, forward "
                  f"{x['forward_ms']:.0f} ms, step {x['total_ms']:.0f} ms")  # fmt: skip

    # k cache hits prefilling together, as the closed loop produces them with the cache on.
    hits = []
    print(f"\nforward_batch of k hits ({args.prefix_len} cached + {args.suffix_len} new "
          f"tokens each): forward ms / without attention ms / attention share")  # fmt: skip
    for k in sorted({1, min(8, c), c}):
        alloc = BlockAllocator(cache.num_blocks)
        items = prefill_items(alloc, args.block_size, k, args.prefix_len, args.suffix_len)
        batch = build_batch(items, dev)

        def forward(b: Batch = batch) -> torch.Tensor:
            return model.forward_batch(b, cache)

        fwd = time_ms(forward, dev, iters=args.iters)
        with attention_replaced():
            bare = time_ms(forward, dev, iters=args.iters)
        hits.append({"hits": k, "forward_ms": fwd, "no_attention_ms": bare,
                     "attention_share": (fwd - bare) / fwd})  # fmt: skip
        print(f"  k={k:>3}: {fwd:7.1f} / {bare:7.1f} ms, attention {(fwd - bare) / fwd:.0%}")
    return {"concurrency": c, "runs": runs, "prefill_hits": hits}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/Qwen3-0.6B"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16"], default=None,
                        help="default: float16 on cuda, float32 on cpu")  # fmt: skip
    parser.add_argument("--attention", choices=DECODE_ATTENTIONS, default="torch")
    parser.add_argument("--prefill-attention", choices=["per_seq", "batched"], default="per_seq",
                        help="per sequence, or one call per layer (M8)")  # fmt: skip
    parser.add_argument("--workload", choices=["m3", "prefix"], default="m3")
    parser.add_argument("--cuda-graphs", action="store_true",
                        help="m3 workload: decode steps replay CUDA graphs (M8)")  # fmt: skip
    parser.add_argument("--concurrency", default=None,
                        help="m3: levels, default 1,8,32,64; prefix: one, default 32")  # fmt: skip
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=None,
                        help="default: 128 (m3), 64 (prefix)")  # fmt: skip
    parser.add_argument("--profile-concurrency", type=int, default=32)
    parser.add_argument("--profile-steps", type=int, default=10)
    parser.add_argument("--num-prefixes", type=int, default=4)
    parser.add_argument("--prefix-len", type=int, default=1024)
    parser.add_argument("--suffix-len", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-batch-tokens", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    args.concurrency = args.concurrency or ("1,8,32,64" if args.workload == "m3" else "32")
    if args.output_len is None:
        args.output_len = 128 if args.workload == "m3" else 64
    dev = torch.device(args.device)
    dtype = getattr(torch, args.dtype or ("float16" if dev.type != "cpu" else "float32"))
    name = "m7_engine_profile" if args.workload == "m3" else "m7_prefix_steps"
    name = "m8_engine_profile" if args.cuda_graphs else name
    out = args.out or Path(f"results/{name}_{dev.type}.json")

    cfg = ModelConfig.from_file(args.model_dir / "config.json")
    model = Qwen3(cfg, load_weights(args.model_dir, cfg, dtype, dev),
                  decode_attention=args.attention,
                  prefill_attention=args.prefill_attention)  # fmt: skip
    print(f"{hardware(dev)}, {dtype}, {args.attention} decode attention, workload "
          f"{args.workload}")  # fmt: skip
    with torch.inference_mode():
        run = run_m3 if args.workload == "m3" else run_prefix
        data = run(args, cfg, model, dev, dtype)

    report = {
        "date": dt.date.today().isoformat(),
        "hardware": hardware(dev),
        "torch": torch.__version__,
        "dtype": str(dtype).removeprefix("torch."),
        "attention": args.attention,
        "prefill_attention": args.prefill_attention,
        "cuda_graphs": args.cuda_graphs,
        "args": {k: str(v) for k, v in vars(args).items()},
        **data,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
