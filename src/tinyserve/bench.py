"""Benchmark harness: drive the engine at a fixed concurrency and summarise the run.

Closed loop: exactly ``concurrency`` requests are in the engine at any time (fewer only
at the end). When one finishes, the next one is added before the following step, like
``concurrency`` users who each send a new request as soon as the last one is answered.

Reported per run:
    output tok/s   output tokens / wall time, over the whole run
    TTFT p50/p99   time to first token: arrival -> first output token (queueing + prefill)
    TPOT p50/p99   time per output token after the first, averaged per request
    KV utilization fraction of KV-cache blocks in use, sampled every step (mean, max)
"""

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch

from tinyserve.engine import Engine

# Qwen3's special tokens (<|im_start|>, <|endoftext|>, ...) start at id 151,643.
MAX_PROMPT_TOKEN_ID = 150_000


def synthetic_prompts(num: int, length: int, vocab_size: int, seed: int = 0) -> list[list[int]]:
    """``num`` prompts of ``length`` random token ids, the same for a given seed.

    Shared by our benchmark and the HF baseline, so both run exactly the same inputs.
    """
    high = min(vocab_size, MAX_PROMPT_TOKEN_ID)
    g = torch.Generator().manual_seed(seed)
    ids: list[list[int]] = torch.randint(0, high, (num, length), generator=g).tolist()
    return ids


def num_requests_for(concurrency: int, requested: int | None = None) -> int:
    """Requests per run: ``requested`` if given, else 4 per slot (at least 16), so every
    level runs several "waves" of requests rather than one."""
    return requested if requested is not None else max(16, 4 * concurrency)


def percentile(values: Sequence[float], p: float) -> float:
    """``p``-th percentile (0..100) with linear interpolation, like ``numpy.percentile``."""
    if not values:
        raise ValueError("percentile of no values")
    xs = sorted(values)
    rank = (len(xs) - 1) * p / 100
    lo, hi = math.floor(rank), math.ceil(rank)
    return xs[lo] + (xs[hi] - xs[lo]) * (rank - lo)


@dataclass(frozen=True)
class BenchResult:
    concurrency: int
    num_requests: int
    output_tokens: int
    seconds: float
    steps: int
    preemptions: int
    output_tok_per_s: float
    ttft_p50_ms: float
    ttft_p99_ms: float
    tpot_p50_ms: float | None  # None when every request produced a single token
    tpot_p99_ms: float | None
    kv_util_mean: float
    kv_util_max: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def run_closed_loop(
    engine: Engine, prompts: list[list[int]], max_new_tokens: int, concurrency: int
) -> BenchResult:
    """Run every prompt through ``engine`` with ``concurrency`` in flight. No EOS: every
    request produces exactly ``max_new_tokens``, so runs are comparable."""
    if concurrency <= 0:
        raise ValueError(f"concurrency must be positive, got {concurrency}")
    todo = list(reversed(prompts))  # pop() from the end = arrival order
    ids: list[int] = []
    in_flight = 0
    start = engine.clock()
    while todo or in_flight:
        while todo and in_flight < concurrency:
            ids.append(engine.add_request(todo.pop(), max_new_tokens))
            in_flight += 1
        in_flight -= len(engine.step())
    seconds = engine.clock() - start

    times = [engine.times[i] for i in ids]
    out_tokens = sum(len(engine.seqs[i].output) for i in ids)
    ttft = [t.ttft * 1e3 for t in times]
    tpot = [x * 1e3 for t in times if (x := t.tpot) is not None]
    util = engine.kv_utilization
    return BenchResult(
        concurrency=concurrency,
        num_requests=len(ids),
        output_tokens=out_tokens,
        seconds=seconds,
        steps=engine.num_steps,
        preemptions=engine.scheduler.num_preemptions,
        output_tok_per_s=out_tokens / seconds,
        ttft_p50_ms=percentile(ttft, 50),
        ttft_p99_ms=percentile(ttft, 99),
        tpot_p50_ms=percentile(tpot, 50) if tpot else None,
        tpot_p99_ms=percentile(tpot, 99) if tpot else None,
        kv_util_mean=sum(util) / len(util),
        kv_util_max=max(util),
    )
