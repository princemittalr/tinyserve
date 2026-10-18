"""Measuring where a step's time goes: one forward pass (M4) and whole engine steps (M7).

The M3 T4 numbers suggest that attention dominates a large decode step: TPOT grows by
about 11 ms per extra sequence, and ``forward_batch`` runs attention once per sequence per
layer. These helpers test that guess directly instead of inferring it:

    decode_tables       B sequences that each have ``context_len`` tokens cached and one
                        new token, i.e. the shape of every step once all requests decode.
    time_ms             median wall time of a call, waiting for the GPU before and after.
    attention_replaced  swaps the model's attention for a stand-in that returns q, so
                        time(real) - time(stand-in) is the cost of attention as written.
    profile_summary     op calls, kernel launches and GPU kernel time from torch.profiler.
    gpu_time_ms         GPU time of the kernels whose name matches, e.g. only the GEMMs.
    prefill_items       batch inputs of sequences that prefill after a cached prefix (M5).
    fill_to_decode      admit requests and run their prefills, so every later step decodes.
    summarize_steps     mean phase times of an engine's ``StepRecord`` list (M7).
    ttft_segments       one HTTP request's time to first token, split where it crossed
                        from the client to the event loop to the engine thread and back.
"""

import statistics
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch.profiler import DeviceType, profile

import tinyserve.model
from tinyserve.block_allocator import BlockAllocator
from tinyserve.block_table import BlockTable
from tinyserve.engine import Engine, StepRecord


def decode_tables(
    allocator: BlockAllocator, block_size: int, num_seqs: int, context_len: int
) -> list[BlockTable]:
    """``num_seqs`` block tables, each holding ``context_len`` cached tokens plus the slot of
    one new token (already appended, as ``build_batch`` expects)."""
    tables = []
    for _ in range(num_seqs):
        t = BlockTable(allocator, block_size)
        t.append(context_len + 1)
        tables.append(t)
    return tables


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_ms(
    fn: Callable[[], object], device: torch.device, warmup: int = 2, iters: int = 5
) -> float:
    """Median wall time of ``fn()`` in milliseconds.

    CUDA kernels run asynchronously: a call returns once its kernels are queued, not done.
    Synchronising before and after each call makes the time cover the GPU work too.
    """
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times)


def _identity_attention(
    q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slots: torch.Tensor, scale: float
) -> torch.Tensor:
    return q


@contextmanager
def attention_replaced() -> Iterator[None]:
    """Inside the block, the model's paged attention returns its queries unchanged.

    Everything else in the forward pass (linears, MLP, norms, RoPE, cache writes, the
    per-sequence slicing and ``torch.cat``) still runs, so the time saved is what attention
    itself costs. The outputs are wrong inside the block; use it for timing only.
    """
    # forward_batch looks the name up in tinyserve.model's globals on every call, so
    # rebinding it there takes effect at once. vars() is that module's globals dict.
    names = vars(tinyserve.model)
    real = names["paged_gqa_attention"]
    names["paged_gqa_attention"] = _identity_attention
    try:
        yield
    finally:
        names["paged_gqa_attention"] = real


def profile_summary(prof: profile, wall_ms: float) -> dict[str, float]:
    """Totals from a finished profiler run over a span that took ``wall_ms``.

    aten_calls: PyTorch operator calls dispatched from Python (each costs CPU time).
    kernel_launches: CUDA kernels launched one by one from the host (0 on CPU).
    graph_launches: CUDA graph replays (M8); each runs a whole recorded forward pass.
    gpu_kernel_ms: time the GPU spent running kernels; gpu_busy is that over wall time.
    A low gpu_busy means the GPU mostly waits for Python to launch the next kernel.
    """
    rows = prof.key_averages()
    aten_calls = sum(r.count for r in rows if r.key.startswith("aten::"))
    launches = sum(r.count for r in rows if r.key in ("cudaLaunchKernel", "cuLaunchKernel"))
    graphs = sum(r.count for r in rows if r.key in ("cudaGraphLaunch", "cuGraphLaunch"))
    # Kernels and copies appear as rows of their own, on the GPU; CPU-side rows also carry
    # their kernels' time, so only GPU rows are summed (as the profiler's own total does).
    gpu_us = sum(r.self_device_time_total for r in rows if r.device_type != DeviceType.CPU)
    return {
        "aten_calls": aten_calls,
        "kernel_launches": launches,
        "graph_launches": graphs,
        "gpu_kernel_ms": gpu_us / 1e3,
        "wall_ms": wall_ms,
        "gpu_busy": gpu_us / 1e3 / wall_ms if wall_ms > 0 else 0.0,
    }


def gpu_time_ms(prof: profile, match: Callable[[str], bool]) -> float:
    """GPU time of the kernels whose name satisfies ``match`` (e.g. every GEMM)."""
    rows = prof.key_averages()
    gpu = [r for r in rows if r.device_type != DeviceType.CPU and match(r.key)]
    return float(sum(r.self_device_time_total for r in gpu)) / 1e3


def prefill_items(
    allocator: BlockAllocator, block_size: int, num_seqs: int, cached_len: int, new_len: int
) -> list[tuple[list[int], BlockTable]]:
    """``build_batch`` inputs for ``num_seqs`` sequences that each have ``cached_len``
    tokens in the cache and prefill ``new_len`` more: a prefix-cache hit (M5)."""
    items = []
    for _ in range(num_seqs):
        t = BlockTable(allocator, block_size)
        t.append(cached_len + new_len)
        items.append(([0] * new_len, t))
    return items


def fill_to_decode(engine: Engine, prompts: list[list[int]], max_new_tokens: int) -> list[int]:
    """Add every prompt, then step until each has its first token. From then on, until
    the first one finishes, every step decodes one token for all of them."""
    ids = [engine.add_request(p, max_new_tokens) for p in prompts]
    while any(not engine.seqs[i].output for i in ids):
        engine.step()
    return ids


_PHASES = ("schedule_ms", "build_ms", "forward_ms", "sample_ms", "update_ms", "total_ms")


def summarize_steps(records: list[StepRecord]) -> dict[str, float]:
    """Means over ``records``: each phase's ms, the batch size and the context read.
    Empty input gives ``{"steps": 0}``."""
    n = len(records)
    if n == 0:
        return {"steps": 0}
    out: dict[str, float] = {"steps": n}
    for phase in _PHASES:
        out[phase] = sum(getattr(r, phase) for r in records) / n
    out["num_seqs"] = sum(r.num_seqs for r in records) / n
    out["num_prefill_seqs"] = sum(r.num_prefill_seqs for r in records) / n
    out["num_tokens"] = sum(r.num_tokens for r in records) / n
    out["context_tokens"] = sum(r.context_tokens for r in records) / n
    out["graph_steps"] = sum(r.graph for r in records) / n  # share replayed as a CUDA graph
    return out


# Where a request's time to first token went, in order; they add up to the whole TTFT.
TTFT_SEGMENTS = ("to_handler", "parse", "to_engine", "engine", "to_loop", "to_client")


def ttft_segments(sent: float, first_chunk: float, trace: dict[str, Any]) -> dict[str, float]:
    """Split one request's TTFT (client clock: ``sent`` -> ``first_chunk``) at the times
    the server recorded (a ``RequestTrace`` as a dict), in ms:

        to_handler  client sends -> HTTP handler starts (socket, uvicorn, a busy loop)
        parse       handler -> "add" queued (body parsing, validation, tokenising)
        to_engine   queued -> the engine thread adds it (it only looks between steps)
        engine      added -> first token out: waiting for admission, then the prefill
        to_loop     first token -> the event loop picks it up
        to_client   -> the first SSE chunk reaches the client (detokenise, send, parse)

    Both processes read ``time.perf_counter``, one system-wide clock on Linux and Windows.
    """
    points = [sent, trace["received"], trace["submitted"], trace["added"],
              trace["first_token"], trace["first_seen"], first_chunk]  # fmt: skip
    if any(p is None for p in points):
        raise ValueError(f"trace is missing a time: {trace}")
    return {
        name: (b - a) * 1e3 for name, a, b in zip(TTFT_SEGMENTS, points, points[1:], strict=False)
    }
