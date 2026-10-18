"""Spec for the scheduler and engine with the prefix cache on (M5).

The cache must change only how much prefill work is done, never what comes out: every
test checks outputs against the same fake model or M1 greedy as without the cache, and
that no block leaks (at the end, every block in use is held by the cache alone, and
evicting everything gives back the whole pool).
"""

import contextlib

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.test_model import CFG, tiny_model
from tests.test_scheduler import fake_greedy, fake_next, sched, step
from tinyserve.batch import Batch
from tinyserve.block_table import BlockTable
from tinyserve.engine import Engine
from tinyserve.kv_cache import KVCache
from tinyserve.model import greedy_generate
from tinyserve.scheduler import Scheduler
from tinyserve.sequence import SeqStatus, Sequence


def run_all(s: Scheduler) -> None:
    for _ in range(2000):
        if not s.has_work():
            return
        assert step(s), "a step with work must schedule at least one sequence"
    raise AssertionError("did not finish")


def check_no_leak(s: Scheduler) -> None:
    assert s.cache is not None
    held = {b for q in s.running for b in q.table.blocks} | set(s.cache.cached_blocks())
    assert s.allocator.num_used == len(held)
    for b in held:
        owners = sum(b in q.table.blocks for q in s.running) + (b in s.cache.cached_blocks())
        assert s.allocator.ref_count(b) == owners


# --- BlockTable.adopt -------------------------------------------------------------------


def test_adopt_shares_blocks_and_counts_their_tokens() -> None:
    s = sched(num_blocks=8, block_size=4)
    owner = BlockTable(s.allocator, 4)
    owner.append(8)
    t = BlockTable(s.allocator, 4)
    t.adopt(owner.blocks)
    assert (t.blocks, t.num_tokens) == (owner.blocks, 8)
    assert [s.allocator.ref_count(b) for b in t.blocks] == [2, 2]
    assert t.append(1) == [t.blocks[2] * 4]  # the next token goes into a new own block
    with pytest.raises(ValueError):
        t.adopt(owner.blocks)  # only an empty table can adopt
    t.release()
    assert [s.allocator.ref_count(b) for b in owner.blocks] == [1, 1]


# --- scheduler with the fake model ------------------------------------------------------


def test_off_by_default() -> None:
    assert sched().cache is None


def test_second_request_reuses_the_first_ones_prompt_blocks() -> None:
    s = sched(block_size=4, prefix_cache=True)
    a = s.add(0, [1, 2, 3, 4, 5, 6, 7, 8, 9], 3)
    run_all(s)
    b = s.add(1, [1, 2, 3, 4, 5, 6, 7, 8, 50, 51], 3)
    assert s.schedule() == [b]
    assert b.num_scheduled == 2  # 8 of its 10 prompt tokens came from the cache
    assert s.num_cached_tokens == 8 and s.num_admitted_tokens == 9 + 10
    s.update([b], [fake_next(b.tokens)])
    run_all(s)
    assert a.output == fake_greedy(a.prompt, 3)
    assert b.output == fake_greedy(b.prompt, 3)
    check_no_leak(s)


def test_output_tokens_are_cached_too() -> None:
    # A follow-up turn of a chat repeats the whole earlier exchange: prompt + answer.
    s = sched(block_size=2, prefix_cache=True)
    a = s.add(0, [1, 2, 3], 4)
    run_all(s)  # a's cached tokens: prompt + 3 of its 4 outputs = 6 tokens, 3 blocks
    b = s.add(1, [*a.tokens, 9], 2)
    s.schedule()
    assert b.num_scheduled == len(b.tokens) - 6


def test_a_fully_cached_prompt_still_computes_its_last_token() -> None:
    s = sched(block_size=4, prefix_cache=True)
    s.add(0, [1, 2, 3, 4, 5, 6, 7, 8], 2)
    run_all(s)
    b = s.add(1, [1, 2, 3, 4, 5, 6, 7, 8], 2)  # every block of b's prompt is cached
    s.schedule()
    # The logits of the last prompt token pick the first output token, so it must run.
    # Whole blocks only: reusing 7 of 8 tokens is not possible, so 4 are reused.
    assert b.num_scheduled == 4
    run_all(s)
    assert b.output == fake_greedy(b.prompt, 2)


def test_short_prompts_reuse_nothing() -> None:
    s = sched(block_size=4, prefix_cache=True)
    s.add(0, [1, 2, 3, 4], 2)
    run_all(s)
    b = s.add(1, [1, 2, 3], 2)  # less than one whole block before its last token
    s.schedule()
    assert b.num_scheduled == 3 and s.num_cached_tokens == 0


def test_cached_blocks_are_evicted_before_anyone_is_preempted() -> None:
    # 4 blocks of 2. a finishes and leaves 2 blocks cached; b and c then need all 4.
    s = sched(num_blocks=4, block_size=2, prefix_cache=True)
    s.add(0, [1, 2, 3, 4, 5], 1)
    run_all(s)
    assert s.cache is not None and s.cache.num_cached_blocks == 2
    b, c = s.add(1, [7, 7, 7], 2), s.add(2, [8, 8, 8], 2)
    run_all(s)
    assert s.num_preemptions == 0
    assert b.output == fake_greedy(b.prompt, 2) and c.output == fake_greedy(c.prompt, 2)


def test_a_request_that_does_not_fit_gives_back_its_adopted_blocks() -> None:
    s = sched(num_blocks=4, block_size=2, prefix_cache=True)
    s.add(0, [1, 2, 3, 4, 5], 1)
    run_all(s)
    hog = s.add(1, [9, 9, 9, 9, 9], 3)
    s.schedule()  # hog evicts the cached blocks and takes 3 of the 4
    late = s.add(2, [9, 9, 9, 9, 6, 6], 2)  # its prefix [9 9 9 9] is not cached yet
    assert s.schedule() == [hog]
    assert late.status is SeqStatus.WAITING and late.table.blocks == []
    check_no_leak(s)


def test_preemption_with_the_cache_on() -> None:
    # 6 blocks of 2. a and b prefill 6 tokens each (3 blocks each, cache full). a needs a
    # 4th block, nothing is cached yet, so b is preempted. Its 3 full blocks go into the
    # cache and are then the only evictable ones, so they are evicted for a.
    s = sched(num_blocks=6, block_size=2, prefix_cache=True)
    a = s.add(0, [1, 1, 1, 1, 1, 1], 3)
    b = s.add(1, [2, 2, 2, 2, 3, 3], 3)
    step(s)
    assert s.schedule() == [a]
    assert b.status is SeqStatus.WAITING and s.num_preemptions == 1
    check_no_leak(s)
    run_all(s)
    assert a.output == fake_greedy(a.prompt, 3)
    assert b.output == fake_greedy(b.prompt, 3)


def test_a_preempted_sequence_reuses_what_is_left_of_its_cached_blocks() -> None:
    # b's first 4 tokens were cached by an earlier request, so b's table shares them. When
    # b is preempted, only its own last block is evictable; the shared prefix survives in
    # the cache and b's recompute starts after it.
    s = sched(num_blocks=6, block_size=2, prefix_cache=True)
    s.add(0, [2, 2, 2, 2, 9], 1)
    run_all(s)  # caches [2 2 | 2 2]: 2 blocks
    a = s.add(1, [1, 1, 1], 4)
    b = s.add(2, [2, 2, 2, 2, 3], 4)
    step(s)  # a: 2 blocks; b: 2 shared + 1 own; cache full (6 used)
    assert s.num_cached_tokens == 4
    step(s)  # a 4 tokens (2 blocks), b 6 tokens (3 blocks)
    s.schedule()  # a needs a 3rd block: b is preempted, its own last block evicted
    assert b.status is SeqStatus.WAITING and s.num_preemptions == 1
    run_all(s)
    assert s.num_cached_tokens >= 8  # b's re-admission reused its cached prefix again
    assert a.output == fake_greedy(a.prompt, 4)
    assert b.output == fake_greedy(b.prompt, 4)
    check_no_leak(s)


def test_a_self_preempted_sequence_gets_its_own_blocks_back() -> None:
    # 3 blocks of 2. a (oldest) keeps going; b needs a new block, nothing newer can be
    # evicted, so b preempts itself. Its full block is cached, and when it is re-admitted
    # after a finishes, it reuses that block instead of recomputing it.
    s = sched(num_blocks=3, block_size=2, prefix_cache=True)
    a, b = s.add(0, [1, 2, 3], 2), s.add(1, [4, 5], 3)
    step(s)  # a: 2 blocks (3 tokens), b: 1 block (2 tokens), cache full
    assert s.schedule() == [a]
    assert b.status is SeqStatus.WAITING and b.table.blocks == []
    check_no_leak(s)
    s.update([a], [fake_next(a.tokens)])  # a finishes
    s.schedule()
    assert b in s.running and s.num_cached_tokens == 2
    run_all(s)
    assert b.output == fake_greedy(b.prompt, 3)


def test_a_request_over_the_token_budget_gives_back_its_adopted_blocks() -> None:
    s = sched(block_size=2, prefix_cache=True, max_batch_tokens=8)
    s.add(0, [1, 2, 3, 4, 5], 1)
    run_all(s)  # caches [1 2 | 3 4]
    s.add(1, [7] * 7, 1)  # 7 tokens: leaves a budget of 1 this step
    late = s.add(2, [1, 2, 3, 4, 9, 9, 9], 1)  # 3 tokens after its cached prefix
    assert len(s.schedule()) == 1
    assert late.status is SeqStatus.WAITING and late.table.blocks == []
    check_no_leak(s)


def test_the_engine_only_runs_the_tokens_after_the_cached_prefix() -> None:
    def tokens_run(prefix_cache: bool) -> int:
        cache = KVCache(CFG, 64, 4, dtype=torch.float32)
        e = Engine(tiny_model(), cache, max_num_seqs=1, prefix_cache=prefix_cache)
        seen: list[int] = []
        forward = e.model.forward_batch

        def counting_forward(batch: Batch, cache: KVCache) -> torch.Tensor:
            seen.append(len(batch.token_ids))
            return forward(batch, cache)

        e.model.forward_batch = counting_forward  # type: ignore[method-assign]
        e.generate([SYSTEM + q for q in QUESTIONS], 3)
        return sum(seen)

    # Without the cache: 5 prompts of 10-13 tokens, plus 2 decode tokens each.
    # With it, 4 of them skip the 8 cached tokens (2 blocks) of SYSTEM, and the last
    # one, a repeat of the first, skips more.
    assert tokens_run(False) - tokens_run(True) >= 4 * 8


@settings(max_examples=200, deadline=None)
@given(
    prefixes=st.lists(st.lists(st.integers(1, 3), max_size=8), min_size=1, max_size=3),
    reqs=st.lists(
        st.tuples(st.integers(0, 2), st.lists(st.integers(1, 3), max_size=4), st.integers(1, 6)),
        min_size=1,
        max_size=10,
    ),
    num_blocks=st.integers(6, 14),
    block_size=st.sampled_from([1, 2, 4]),
    max_num_seqs=st.integers(1, 5),
    arrive_every=st.integers(0, 3),
)
def test_random_shared_prefix_workloads_match_and_do_not_leak(
    prefixes: list[list[int]],
    reqs: list[tuple[int, list[int], int]],
    num_blocks: int,
    block_size: int,
    max_num_seqs: int,
    arrive_every: int,
) -> None:
    s = sched(
        num_blocks, block_size, prefix_cache=True, max_num_seqs=max_num_seqs, max_batch_tokens=20
    )
    pending = [(prefixes[p % len(prefixes)] + tail, n) for p, tail, n in reqs]
    pending = [(prompt, n) for prompt, n in pending if prompt]
    seqs: list[Sequence] = []
    for t in range(3000):
        # Requests arrive over time, so later ones can hit what earlier ones cached.
        if pending and (arrive_every == 0 or t % arrive_every == 0):
            prompt, n = pending.pop(0)
            with contextlib.suppress(ValueError):  # too long for this cache
                seqs.append(s.add(len(seqs), prompt, n))
        if not s.has_work():
            if not pending:
                break
            continue
        assert step(s)
        check_no_leak(s)
    assert not s.has_work() and not pending
    for q in seqs:
        assert q.output == fake_greedy(q.prompt, q.max_new_tokens)
    assert s.cache is not None
    s.cache.evict(num_blocks)
    assert s.allocator.num_free == num_blocks


# --- engine with the tiny model ---------------------------------------------------------

SYSTEM = [7, 3, 9, 1, 4, 4, 2, 8, 5]  # a shared "system prompt" of 9 tokens
QUESTIONS = [[11, 12], [13], [11, 14, 15], [16, 17, 18, 19], [11, 12]]


@pytest.mark.parametrize("num_blocks", [64, 9])  # roomy; tight (eviction + preemption)
def test_engine_with_prefix_cache_matches_m1_greedy(num_blocks: int) -> None:
    cache = KVCache(CFG, num_blocks, 4, dtype=torch.float32)
    e = Engine(tiny_model(), cache, max_num_seqs=2, prefix_cache=True)
    prompts = [SYSTEM + q for q in QUESTIONS]
    m = tiny_model()
    assert e.generate(prompts, 6) == [greedy_generate(m, p, 6) for p in prompts]
    s = e.scheduler
    assert s.num_cached_tokens > 0
    assert s.cache is not None
    s.cache.evict(num_blocks)
    assert s.allocator.num_free == num_blocks


def test_engine_reuses_the_system_prompt_after_the_first_request() -> None:
    cache = KVCache(CFG, 64, 4, dtype=torch.float32)
    e = Engine(tiny_model(), cache, max_num_seqs=1, prefix_cache=True)
    e.generate([SYSTEM + QUESTIONS[0]], 3)
    e.generate([SYSTEM + q for q in QUESTIONS[1:]], 3)
    # The 9-token prefix covers 2 whole blocks of 4: 8 tokens reused by each of the 4
    # later requests, plus more for the last one, which repeats the first exactly.
    assert e.scheduler.num_cached_tokens >= 4 * 8
