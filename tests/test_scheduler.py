"""Spec for the continuous-batching scheduler (M3).

Most tests drive the scheduler with a fake model: the "next token" is a fixed function
of the sequence's tokens so far, so the right output of every request is known without
running a network. The last tests run the tiny random Qwen3 through ``forward_batch``.
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.test_model import CFG, tiny_model
from tinyserve.batch import build_batch
from tinyserve.block_allocator import BlockAllocator
from tinyserve.kv_cache import KVCache
from tinyserve.model import greedy_generate
from tinyserve.scheduler import Scheduler
from tinyserve.sequence import SeqStatus, Sequence


def fake_next(tokens: list[int]) -> int:
    return (sum(tokens) * 7 + len(tokens)) % 97 + 1


def fake_greedy(prompt: list[int], max_new: int, eos: frozenset[int] = frozenset()) -> list[int]:
    ids, out = list(prompt), []
    for _ in range(max_new):
        out.append(fake_next(ids))
        ids.append(out[-1])
        if out[-1] in eos:
            break
    return out


def step(s: Scheduler) -> list[Sequence]:
    seqs = s.schedule()
    s.update(seqs, [fake_next(q.tokens) for q in seqs])
    return seqs


def check_invariants(s: Scheduler) -> None:
    assert len(s.running) <= s.max_num_seqs
    assert sum(len(q.table.blocks) for q in s.running) == s.allocator.num_used
    for q in s.waiting:
        assert q.status is SeqStatus.WAITING
        assert q.table.blocks == []  # waiting sequences hold no cache memory


def sched(num_blocks: int = 64, block_size: int = 4, **kw: int) -> Scheduler:
    return Scheduler(BlockAllocator(num_blocks), block_size, **kw)


# --- add ----------------------------------------------------------------------------------


def test_add_rejects_requests_that_could_never_finish() -> None:
    s = sched(num_blocks=4, block_size=4, max_batch_tokens=100)  # cache holds 16 tokens
    s.add(0, [1] * 10, 7)  # up to 16 tokens cached: fits exactly
    with pytest.raises(ValueError):
        s.add(1, [1] * 10, 8)  # 17 tokens: more than the whole cache
    with pytest.raises(ValueError):
        sched(max_batch_tokens=8).add(2, [1] * 5, 5)  # recompute would be 9 tokens
    with pytest.raises(ValueError):
        s.add(3, [], 5)
    with pytest.raises(ValueError):
        s.add(4, [1], 0)


# --- admission ----------------------------------------------------------------------------


def test_first_step_prefills_every_prompt_that_fits() -> None:
    s = sched()
    a, b = s.add(0, [5, 9, 2], 4), s.add(1, [3, 8], 4)
    assert s.schedule() == [a, b]
    assert a.status is b.status is SeqStatus.RUNNING
    assert a.table.num_tokens == 3 and b.table.num_tokens == 2
    assert a.pending() == [] and b.pending() == []


def test_running_sequences_then_get_one_token_each() -> None:
    s = sched()
    a, b = s.add(0, [5, 9, 2], 4), s.add(1, [3, 8], 4)
    step(s)
    assert a.pending() == a.output[-1:] and b.pending() == b.output[-1:]
    s.schedule()
    assert a.table.num_tokens == 4 and b.table.num_tokens == 3


def test_admission_respects_max_num_seqs() -> None:
    s = sched(max_num_seqs=2)
    seqs = [s.add(i, [i + 1], 3) for i in range(3)]
    assert s.schedule() == seqs[:2]
    assert list(s.waiting) == [seqs[2]]


def test_admission_respects_the_token_budget() -> None:
    s = sched(max_batch_tokens=10)
    a, b = s.add(0, [1] * 6, 3), s.add(1, [2] * 6, 3)
    assert s.schedule() == [a]  # 6 + 6 > 10
    s.update([a], [7])
    assert s.schedule() == [a, b]  # 1 decode token + 6 <= 10


def test_admission_is_fcfs_and_does_not_skip_ahead() -> None:
    # 3 blocks of 4. a takes 2 blocks; b needs 2 more but only 1 is free. c needs 1 and
    # would fit, but must not jump the queue ahead of b.
    s = sched(num_blocks=3, block_size=4)
    a, b, c = s.add(0, [1] * 8, 3), s.add(1, [2] * 5, 3), s.add(2, [3], 3)
    assert s.schedule() == [a]
    assert list(s.waiting) == [b, c]


def test_finished_sequences_free_their_blocks_and_leave() -> None:
    s = sched()
    a = s.add(0, [5, 9, 2], 2)
    b = s.add(1, [3, 8], 10, eos_ids=frozenset({fake_next([3, 8])}))
    seqs = s.schedule()
    finished = s.update(seqs, [fake_next(q.tokens) for q in seqs])
    assert finished == [b]  # b's first token was EOS
    assert b.status is SeqStatus.FINISHED and b.table.blocks == []
    assert s.running == [a]
    step(s)
    assert a.status is SeqStatus.FINISHED and len(a.output) == 2
    assert s.allocator.num_used == 0 and not s.has_work()


# --- preemption ---------------------------------------------------------------------------


def preemption_setup() -> tuple[Scheduler, Sequence, Sequence]:
    # 4 blocks of 2 = 8 slots. a and b prefill 4 tokens each (2 blocks each, cache full).
    s = sched(num_blocks=4, block_size=2)
    a, b = s.add(0, [1, 2, 3, 4], 4), s.add(1, [5, 6, 7, 8], 4)
    step(s)
    return s, a, b


def test_newest_sequence_is_preempted_when_the_cache_is_full() -> None:
    s, a, b = preemption_setup()
    # Both need a 3rd block for their 5th token; only by evicting b can a get one.
    assert s.schedule() == [a]
    assert a.table.num_tokens == 5
    assert b.status is SeqStatus.WAITING and b.table.blocks == []
    assert b.num_preemptions == 1 and s.num_preemptions == 1
    assert list(s.waiting) == [b]
    check_invariants(s)


def test_the_victim_is_the_newest_not_the_next_oldest() -> None:
    # 6 blocks of 2. a, b, c prefill 4 tokens each: 2 blocks each, cache full.
    s = sched(num_blocks=6, block_size=2)
    a, b, c = (s.add(i, [i + 1] * 4, 4) for i in range(3))
    step(s)
    # a needs a 3rd block: evicting c frees 2, a takes 1, then b takes the other.
    assert s.schedule() == [a, b]
    assert list(s.waiting) == [c]


def test_preempted_sequence_is_recomputed_from_prompt_and_output() -> None:
    s, _, b = preemption_setup()
    s.schedule()
    assert b.pending() == b.prompt + b.output  # nothing of b is cached any more


def test_preempted_sequence_goes_ahead_of_later_arrivals() -> None:
    s, _, b = preemption_setup()
    c = s.add(2, [9], 2)
    s.schedule()
    assert list(s.waiting) == [b, c]


def test_a_sequence_with_nothing_newer_to_evict_preempts_itself() -> None:
    # a keeps its blocks; b is the newest, so b waits rather than evicting a.
    s = sched(num_blocks=3, block_size=2)
    a, b = s.add(0, [1, 2, 3], 3), s.add(1, [4, 5], 3)
    step(s)  # a: 2 blocks (3 tokens), b: 1 block (2 tokens), cache full
    assert s.schedule() == [a]  # a's 4th token fits in its 2nd block; b needs a new one
    assert b.status is SeqStatus.WAITING and a.status is SeqStatus.RUNNING


@settings(max_examples=60, deadline=None)
@given(
    reqs=st.lists(
        st.tuples(st.lists(st.integers(1, 99), min_size=1, max_size=10), st.integers(1, 8)),
        min_size=1,
        max_size=8,
    ),
    num_blocks=st.integers(5, 12),
    block_size=st.sampled_from([1, 2, 4]),
    max_num_seqs=st.integers(1, 5),
)
def test_random_workloads_finish_with_correct_outputs_and_no_leaks(
    reqs: list[tuple[list[int], int]], num_blocks: int, block_size: int, max_num_seqs: int
) -> None:
    s = Scheduler(BlockAllocator(num_blocks), block_size, max_num_seqs, max_batch_tokens=20)
    seqs = []
    for i, (prompt, max_new) in enumerate(reqs):
        try:
            seqs.append(s.add(i, prompt, max_new))
        except ValueError:
            continue  # too long for this cache
    for _ in range(2000):  # bounded: a livelock fails the test instead of hanging
        if not s.has_work():
            break
        before = {q.seq_id: len(q.pending()) for q in [*s.running, *s.waiting]}
        seqs_run = step(s)
        assert seqs_run, "a step with work must schedule at least one sequence"
        assert sum(before[q.seq_id] for q in seqs_run) <= s.max_batch_tokens
        check_invariants(s)
    assert not s.has_work()
    assert s.allocator.num_free == num_blocks
    for q in seqs:
        assert q.status is SeqStatus.FINISHED
        assert q.output == fake_greedy(q.prompt, q.max_new_tokens)


# --- with the real model ----------------------------------------------------------------


def run_engine(s: Scheduler, cache: KVCache) -> None:
    m = tiny_model()
    while s.has_work():
        pending = {q.seq_id: q.pending() for q in [*s.running, *s.waiting]}
        seqs = s.schedule()
        batch = build_batch([(pending[q.seq_id], q.table) for q in seqs])
        s.update(seqs, m.forward_batch(batch, cache).argmax(-1).tolist())


@pytest.mark.parametrize("num_blocks", [64, 6])  # 6 blocks of 4 forces preemptions
def test_engine_outputs_match_m1_greedy_with_and_without_preemption(num_blocks: int) -> None:
    cache = KVCache(CFG, num_blocks, 4, dtype=torch.float32)
    s = Scheduler(BlockAllocator(num_blocks), 4, max_num_seqs=4, max_batch_tokens=64)
    prompts = [[1], [5, 9, 2], [3, 8, 11, 2, 7, 7, 1], [4, 4, 6, 1, 2]]
    seqs = [s.add(i, p, 10) for i, p in enumerate(prompts)]
    run_engine(s, cache)
    if num_blocks == 6:
        assert s.num_preemptions > 0
    m = tiny_model()
    for q in seqs:
        assert q.output == greedy_generate(m, q.prompt, 10)
    assert s.allocator.num_free == num_blocks
