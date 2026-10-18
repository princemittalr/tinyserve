"""The engine (scheduler + batched model) must give every request M1's greedy tokens,
however requests arrive and however small the cache is. Tiny random Qwen3."""

import itertools

import pytest
import torch

from tests.test_model import CFG, tiny_model
from tinyserve.engine import Engine, RequestTimes
from tinyserve.kv_cache import KVCache
from tinyserve.model import greedy_generate

PROMPTS = [[1], [5, 9, 2], [3, 8, 11, 2, 7, 7, 1], [4, 4, 6, 1, 2], [9, 30, 12, 5, 6, 2, 8]]


def engine(num_blocks: int = 64, block_size: int = 4, max_num_seqs: int = 8) -> Engine:
    cache = KVCache(CFG, num_blocks, block_size, dtype=torch.float32)
    return Engine(tiny_model(), cache, max_num_seqs=max_num_seqs)


def expected(prompt: list[int], n: int) -> list[int]:
    return greedy_generate(tiny_model(), prompt, n)


@pytest.mark.parametrize(
    ("num_blocks", "max_num_seqs"),
    [(64, 8), (64, 1), (7, 8)],  # roomy; one at a time; tight cache (preemptions)
)
def test_generate_matches_m1_greedy(num_blocks: int, max_num_seqs: int) -> None:
    e = engine(num_blocks, max_num_seqs=max_num_seqs)
    assert e.generate(PROMPTS, 10) == [expected(p, 10) for p in PROMPTS]
    assert e.scheduler.allocator.num_free == num_blocks
    if num_blocks == 7:
        assert e.scheduler.num_preemptions > 0


def test_requests_can_join_while_others_are_running() -> None:
    e = engine()
    first = [e.add_request(p, 10) for p in PROMPTS[:2]]
    for _ in range(3):
        e.step()
    later = [e.add_request(p, 10) for p in PROMPTS[2:]]
    while e.has_work():
        e.step()
    for i, p in zip(first + later, PROMPTS, strict=True):
        assert e.seqs[i].output == expected(p, 10)


def test_continuous_batching_beats_static_batching_in_steps() -> None:
    # One long request and four short ones. Static batching would hold the short ones'
    # slots until the long one is done; here a finished request leaves at once, so the
    # whole workload takes as many steps as the longest request.
    e = engine(max_num_seqs=8)
    e.add_request(PROMPTS[2], 12)
    for p in PROMPTS[:2]:
        e.add_request(p, 2)
    e.step()
    e.step()
    assert len(e.scheduler.running) == 1  # the short ones are already gone
    while e.has_work():
        e.step()
    assert e.num_steps == 12


def test_stops_at_eos() -> None:
    e = engine()
    eos = frozenset({expected(PROMPTS[1], 3)[2]})
    out = e.generate([PROMPTS[1]], 10, eos)[0]
    assert out == greedy_generate(tiny_model(), PROMPTS[1], 10, eos)
    assert out[-1] in eos and len(out) <= 3


def test_records_one_time_per_output_token() -> None:
    ticks = itertools.count()
    cache = KVCache(CFG, 64, 4, dtype=torch.float32)
    e = Engine(tiny_model(), cache, clock=lambda: float(next(ticks)))
    e.generate(PROMPTS[:3], 5)
    for i, q in e.seqs.items():
        t = e.times[i]
        assert len(t.tokens) == len(q.output)
        assert t.tokens == sorted(t.tokens)
        assert t.ttft > 0
    assert len(e.kv_utilization) == e.num_steps
    assert all(0 < u <= 1 for u in e.kv_utilization)


def test_ttft_and_tpot() -> None:
    t = RequestTimes(arrival=1.0, tokens=[3.0, 3.5, 4.0, 4.5])
    assert t.ttft == 2.0
    assert t.tpot == 0.5
    assert RequestTimes(arrival=0.0, tokens=[1.0]).tpot is None
