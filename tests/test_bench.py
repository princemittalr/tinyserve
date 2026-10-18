"""The benchmark harness: percentiles, the closed loop, and the summary numbers."""

import itertools

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from tests.test_model import CFG, tiny_model
from tinyserve.bench import percentile, run_closed_loop
from tinyserve.engine import Engine
from tinyserve.kv_cache import KVCache
from tinyserve.sequence import Sequence

PROMPTS = [[1, 2, 3], [5, 9], [3, 8, 11, 2], [4], [9, 30, 12], [7, 7], [2, 6, 1]]


def make_engine(num_blocks: int = 64, max_num_seqs: int = 64) -> Engine:
    ticks = itertools.count()  # fake clock: +1 per call, so timings are deterministic
    cache = KVCache(CFG, num_blocks, 4, dtype=torch.float32)
    return Engine(tiny_model(), cache, max_num_seqs, clock=lambda: float(next(ticks)))


@given(st.lists(st.floats(-1e6, 1e6), min_size=1, max_size=50), st.sampled_from([0, 50, 99]))
def test_percentile_matches_numpy(xs: list[float], p: int) -> None:
    assert percentile(xs, p) == pytest.approx(float(np.percentile(xs, p)), abs=1e-6)


def test_percentile_of_nothing_is_an_error() -> None:
    with pytest.raises(ValueError):
        percentile([], 50)


@pytest.mark.parametrize("concurrency", [1, 3, 64])
def test_closed_loop_keeps_at_most_concurrency_in_flight(concurrency: int) -> None:
    e = make_engine()
    peak = 0
    real_step = e.step

    def counting_step() -> list[Sequence]:
        nonlocal peak
        s = e.scheduler
        peak = max(peak, len(s.running) + len(s.waiting))
        return real_step()

    e.step = counting_step  # type: ignore[method-assign]
    r = run_closed_loop(e, PROMPTS, max_new_tokens=5, concurrency=concurrency)
    assert peak == min(concurrency, len(PROMPTS))
    assert r.num_requests == len(PROMPTS)
    assert r.output_tokens == 5 * len(PROMPTS)
    assert not e.has_work()


def test_concurrency_one_runs_requests_back_to_back() -> None:
    e = make_engine()
    r = run_closed_loop(e, PROMPTS, max_new_tokens=4, concurrency=1)
    assert r.steps == 4 * len(PROMPTS)  # no overlap: one prefill + 3 decodes each


def test_summary_numbers_are_consistent() -> None:
    e = make_engine()
    r = run_closed_loop(e, PROMPTS, max_new_tokens=6, concurrency=4)
    assert r.output_tok_per_s == pytest.approx(r.output_tokens / r.seconds)
    assert 0 < r.ttft_p50_ms <= r.ttft_p99_ms
    assert r.tpot_p50_ms is not None and r.tpot_p99_ms is not None
    assert 0 < r.tpot_p50_ms <= r.tpot_p99_ms
    assert 0 < r.kv_util_mean <= r.kv_util_max <= 1
    assert r.preemptions == 0
    assert set(r.as_dict()) >= {"output_tok_per_s", "ttft_p99_ms", "kv_util_max"}


def test_tight_cache_shows_up_as_preemptions_and_full_utilization() -> None:
    e = make_engine(num_blocks=6)
    r = run_closed_loop(e, PROMPTS, max_new_tokens=12, concurrency=7)
    assert r.preemptions > 0
    assert r.kv_util_max == 1.0
    assert r.output_tokens == 12 * len(PROMPTS)


def test_single_token_requests_have_no_tpot() -> None:
    r = run_closed_loop(make_engine(), PROMPTS, max_new_tokens=1, concurrency=2)
    assert r.tpot_p50_ms is None and r.tpot_p99_ms is None


def test_concurrency_must_be_positive() -> None:
    with pytest.raises(ValueError):
        run_closed_loop(make_engine(), PROMPTS, 3, concurrency=0)
