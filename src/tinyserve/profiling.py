"""Measuring where a decode step's time goes, before writing the M4 kernel.

The M3 T4 numbers suggest that attention dominates a large decode step: TPOT grows by
about 11 ms per extra sequence, and ``forward_batch`` runs attention once per sequence per
layer. These helpers test that guess directly instead of inferring it:

    decode_tables       B sequences that each have ``context_len`` tokens cached and one
                        new token, i.e. the shape of every step once all requests decode.
    time_ms             median wall time of a call, waiting for the GPU before and after.
    attention_replaced  swaps the model's attention for a stand-in that returns q, so
                        time(real) - time(stand-in) is the cost of attention as written.
    profile_summary     op calls, kernel launches and GPU kernel time from torch.profiler.
"""

import statistics
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch
from torch.profiler import DeviceType, profile

import tinyserve.model
from tinyserve.block_allocator import BlockAllocator
from tinyserve.block_table import BlockTable


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
    kernel_launches: CUDA kernels launched (0 on CPU).
    gpu_kernel_ms: time the GPU spent running kernels; gpu_busy is that over wall time.
    A low gpu_busy means the GPU mostly waits for Python to launch the next kernel.
    """
    rows = prof.key_averages()
    aten_calls = sum(r.count for r in rows if r.key.startswith("aten::"))
    launches = sum(r.count for r in rows if r.key in ("cudaLaunchKernel", "cuLaunchKernel"))
    # Kernels and copies appear as rows of their own, on the GPU; CPU-side rows also carry
    # their kernels' time, so only GPU rows are summed (as the profiler's own total does).
    gpu_us = sum(r.self_device_time_total for r in rows if r.device_type != DeviceType.CPU)
    return {
        "aten_calls": aten_calls,
        "kernel_launches": launches,
        "gpu_kernel_ms": gpu_us / 1e3,
        "wall_ms": wall_ms,
        "gpu_busy": gpu_us / 1e3 / wall_ms if wall_ms > 0 else 0.0,
    }
