"""Measuring where a step's time goes: one forward pass (M4) and whole engine steps (M7).

The M3 T4 numbers suggest that attention dominates a large decode step: TPOT grows by
about 11 ms per extra sequence, and ``forward_batch`` runs attention once per sequence per
layer. These helpers test that guess directly instead of inferring it:

    decode_tables       B sequences that each have ``context_len`` tokens cached and one
                        new token, i.e. the shape of every step once all requests decode.
    time_ms             median wall time of a call, waiting for the GPU before and after.
    loop_ms             mean time of a call run back to back, for short kernels (M10).
    attention_replaced  swaps the model's attention for a stand-in that returns q, so
                        time(real) - time(stand-in) is the cost of attention as written.
    profile_summary     op calls, kernel launches and GPU kernel time from torch.profiler.
    gpu_time_ms         GPU time of the kernels whose name matches, e.g. only the GEMMs.
    is_gemm             whether a kernel is a matrix multiply (cuBLAS GEMM or GEMV).
    prefill_items       batch inputs of sequences that prefill after a cached prefix (M5).
    fill_to_decode      admit requests and run their prefills, so every later step decodes.
    summarize_steps     mean phase times of an engine's ``StepRecord`` list (M7).
    ttft_segments       one HTTP request's time to first token, split where it crossed
                        from the client to the event loop to the engine thread and back.
    GpuSampler          the GPU's clocks, power, temperature and the reasons the driver
                        held the clock down, while a block runs (M10).
"""

import statistics
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import TracebackType
from typing import Any, ClassVar

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


def loop_ms(fn: Callable[[], object], device: torch.device, iters: int = 100,
            warmup: int = 3) -> float:  # fmt: skip
    """Mean time of ``fn()`` over ``iters`` calls launched back to back, in milliseconds,
    after ``warmup`` untimed calls.

    For a kernel that runs for tens of microseconds, ``time_ms``'s synchronisation around
    each call costs about as much as the kernel; here the GPU runs the calls one after
    another and only the whole loop waits, so launches overlap the kernels in flight. A
    GPU that was idle runs at a low clock for a while: give a long ``warmup`` then.
    """
    for _ in range(warmup):
        fn()
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - start) * 1e3 / iters


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


def is_gemm(name: str) -> bool:
    """A matrix-multiply kernel. cuBLAS runs a 1-row product as a GEMV (``gemv2T_kernel``,
    ``gemvx``), not a GEMM, so matching "gemm" alone misses every GEMM at batch 1 (M10)."""
    lower = name.lower()
    return "gemm" in lower or "gemv" in lower


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


class GpuSampler:
    """Samples the GPU with ``nvidia-smi`` every ``interval_ms`` while the ``with`` block
    runs (M10): SM and memory clocks (MHz), power draw (W), temperature (C) and the
    reasons the driver gave for holding the clock below its maximum.

    A T4 is capped at 70 W and lowers its clock under load or when hot, so two runs of the
    same kernel can differ by the clock alone; this records the clock next to a timing.
    Without ``nvidia-smi`` (or on a CPU) ``samples`` stays empty and ``summary()`` is {}.
    """

    QUERY = "clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active"
    # The bits of clocks_event_reasons.active (nvidia-smi --help-query-gpu).
    REASONS: ClassVar[dict[int, str]] = {
        0x1: "idle", 0x2: "app_clocks", 0x4: "sw_power_cap", 0x8: "hw_slowdown",
        0x10: "sync_boost", 0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake",
        0x100: "display_clocks"}  # fmt: skip

    def __init__(self, device_index: int = 0, interval_ms: int = 50,
                 command: list[str] | None = None) -> None:  # fmt: skip
        self.command = command or [
            "nvidia-smi", f"--id={device_index}", f"--query-gpu={self.QUERY}",
            "--format=csv,noheader,nounits", f"--loop-ms={interval_ms}",
        ]  # fmt: skip
        self.samples: list[tuple[float, float, float, float, int]] = []
        self._proc: subprocess.Popen[str] | None = None

    def __enter__(self) -> "GpuSampler":
        try:
            self._proc = subprocess.Popen(self.command, stdout=subprocess.PIPE, text=True)
        except OSError:  # no nvidia-smi
            self._proc = None
        return self

    def __exit__(self, kind: type[BaseException] | None, err: BaseException | None,
                 tb: TracebackType | None) -> None:  # fmt: skip
        if self._proc is None:
            return
        self._proc.terminate()
        out, _ = self._proc.communicate()
        for line in out.splitlines():
            try:
                *fields, reasons = (x.strip() for x in line.split(","))
                sm, mem, power, temp = (float(x) for x in fields)
                self.samples.append((sm, mem, power, temp, int(reasons, 16)))
            except ValueError:  # "[N/A]", a header, a line cut off by terminate()
                continue

    def summary(self) -> dict[str, Any]:
        """Means and minimums, and for each reason the share of samples that gave it."""
        if not self.samples:
            return {}
        sm, mem, power, temp, reasons = zip(*self.samples, strict=True)
        held = {name: sum(1 for r in reasons if r & bit) / len(reasons)
                for bit, name in self.REASONS.items() if any(r & bit for r in reasons)}  # fmt: skip
        return {"samples": len(sm), "sm_mhz_mean": statistics.mean(sm),
                "sm_mhz_min": min(sm), "mem_mhz_mean": statistics.mean(mem),
                "mem_mhz_min": min(mem), "power_w_mean": statistics.mean(power),
                "temp_c_max": max(temp), "clock_held_by": held}  # fmt: skip

    @staticmethod
    def describe(g: dict[str, Any]) -> str:
        """One line for ``summary()``'s result."""
        if not g:
            return "no nvidia-smi"
        held = ", ".join(f"{k} {v:.0%}" for k, v in g["clock_held_by"].items()) or "nothing"
        return (f"SM {g['sm_mhz_mean']:.0f} MHz (min {g['sm_mhz_min']:.0f}), memory "
                f"{g['mem_mhz_mean']:.0f} MHz (min {g['mem_mhz_min']:.0f}), "
                f"{g['power_w_mean']:.1f} W, {g['temp_c_max']:.0f} C, "
                f"clock held by: {held}")  # fmt: skip
