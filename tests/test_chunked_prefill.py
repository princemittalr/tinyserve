"""Chunked prefill (M9): a long prompt is spread over several steps of at most
``max_batch_tokens`` tokens, and only the step that runs its last chunk samples.

The scheduler tests use the fake model of test_scheduler.py; the engine tests check the
tiny Qwen3 still gives M1's greedy tokens however the prompts are cut.
"""

import importlib.util

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.test_model import CFG, tiny_model
from tests.test_scheduler import check_invariants, fake_greedy, fake_next
from tinyserve.block_allocator import BlockAllocator
from tinyserve.engine import Engine
from tinyserve.kv_cache import KVCache
from tinyserve.model import Qwen3, greedy_generate
from tinyserve.sampling import SamplingParams
from tinyserve.scheduler import Scheduler
from tinyserve.sequence import SeqStatus, Sequence

HAS_TRITON = importlib.util.find_spec("triton") is not None


def chunked(max_batch_tokens: int, max_num_seqs: int) -> Scheduler:
    return Scheduler(BlockAllocator(64), 4, max_num_seqs, max_batch_tokens, chunked_prefill=True)


def cstep(s: Scheduler) -> list[Sequence]:
    """One step with the fake model: only sequences that finished their prompt sample."""
    seqs = s.schedule()
    ready = [q for q in seqs if not q.in_prefill()]
    s.update(ready, [fake_next(q.tokens) for q in ready])
    return seqs


# --- the scheduler ------------------------------------------------------------------------


def test_off_by_default_a_prompt_over_the_budget_is_rejected() -> None:
    with pytest.raises(ValueError):
        Scheduler(BlockAllocator(64), 4, max_batch_tokens=8).add(0, [1] * 9, 2)
    chunked(max_batch_tokens=8, max_num_seqs=4).add(0, [1] * 9, 2)  # fine when chunked


def test_the_budget_must_hold_one_decode_token_per_sequence() -> None:
    with pytest.raises(ValueError):
        chunked(max_batch_tokens=3, max_num_seqs=4)


def test_a_long_prompt_is_split_into_chunks_and_samples_once() -> None:
    s = chunked(max_batch_tokens=4, max_num_seqs=2)
    a = s.add(0, list(range(1, 11)), 3)  # 10 prompt tokens: chunks of 4, 4, 2
    sizes = []
    while not a.output:
        assert cstep(s) == [a]
        sizes.append(a.num_scheduled)
    assert sizes == [4, 4, 2]
    assert a.status is SeqStatus.RUNNING and a.table.num_tokens == 10
    while s.has_work():
        cstep(s)
    assert a.output == fake_greedy(a.prompt, 3)


def test_a_sequence_mid_prompt_samples_nothing() -> None:
    s = chunked(max_batch_tokens=4, max_num_seqs=2)
    a = s.add(0, list(range(1, 11)), 3)
    s.schedule()
    assert a.in_prefill()
    with pytest.raises(ValueError):
        s.update([a], [5])


def test_decodes_get_their_token_before_a_prompt_gets_a_chunk() -> None:
    s = chunked(max_batch_tokens=4, max_num_seqs=4)
    a, b = s.add(0, [1, 2], 5), s.add(1, [3, 4], 5)
    cstep(s)  # both prefill whole (2 + 2 = 4)
    c = s.add(2, list(range(1, 21)), 2)  # 20 tokens
    assert cstep(s) == [a, b, c]
    assert (a.num_scheduled, b.num_scheduled, c.num_scheduled) == (1, 1, 2)
    # c is now running mid-prompt; next step the decodes still come first.
    assert cstep(s) == [a, b, c]
    assert c.num_scheduled == 2 and c.in_prefill()


def test_a_new_prompt_is_admitted_with_what_is_left_of_the_budget() -> None:
    s = chunked(max_batch_tokens=3, max_num_seqs=3)
    a = s.add(0, [1, 2], 5)
    b = s.add(1, list(range(1, 11)), 2)
    c = s.add(2, [7], 5)
    cstep(s)  # a 2 tokens, b the 1 left over; the budget is gone, so c waits
    assert (a.num_scheduled, b.num_scheduled) == (2, 1)
    assert list(s.waiting) == [c]
    # a decodes (1), b gets the other 2; c still does not fit.
    assert cstep(s) == [a, b]
    assert b.num_scheduled == 2


@settings(max_examples=80, deadline=None)
@given(
    reqs=st.lists(
        st.tuples(st.lists(st.integers(1, 99), min_size=1, max_size=14), st.integers(1, 8)),
        min_size=1,
        max_size=8,
    ),
    num_blocks=st.integers(5, 12),
    block_size=st.sampled_from([1, 2, 4]),
    max_num_seqs=st.integers(1, 4),
    extra_budget=st.integers(0, 6),
    prefix_cache=st.booleans(),
)
def test_random_workloads_respect_the_budget_and_finish_correctly(
    reqs: list[tuple[list[int], int]],
    num_blocks: int,
    block_size: int,
    max_num_seqs: int,
    extra_budget: int,
    prefix_cache: bool,
) -> None:
    budget = max_num_seqs + extra_budget
    s = Scheduler(
        BlockAllocator(num_blocks),
        block_size,
        max_num_seqs,
        max_batch_tokens=budget,
        prefix_cache=prefix_cache,
        chunked_prefill=True,
    )
    seqs = []
    for i, (prompt, max_new) in enumerate(reqs):
        try:
            seqs.append(s.add(i, prompt, max_new))
        except ValueError:
            continue  # longer than the whole cache
    for _ in range(3000):  # bounded: a livelock fails the test instead of hanging
        if not s.has_work():
            break
        seqs_run = s.schedule()
        assert seqs_run, "a step with work must schedule at least one sequence"
        assert sum(q.num_scheduled for q in seqs_run) <= budget
        assert all(q.num_scheduled >= 1 for q in seqs_run)
        ready = [q for q in seqs_run if not q.in_prefill()]
        assert len(seqs_run) - len(ready) <= 1  # only the last one admitted is mid-prompt
        s.update(ready, [fake_next(q.tokens) for q in ready])
        if not prefix_cache:
            check_invariants(s)
    assert not s.has_work()
    if s.cache is not None:
        s.cache.evict(num_blocks)
    assert s.allocator.num_free == num_blocks
    for q in seqs:
        assert q.status is SeqStatus.FINISHED
        assert q.output == fake_greedy(q.prompt, q.max_new_tokens)


# --- the engine on the tiny Qwen3 ----------------------------------------------------------

PROMPTS = [
    [1],
    [5, 9, 2],
    [3, 8, 11, 2, 7, 7, 1, 4, 9, 9, 2, 6, 3],
    [4, 4, 6, 1, 2],
    [9, 30, 12, 5, 6, 2, 8, 17, 40, 3, 3, 1],
]


def expected(prompt: list[int], n: int) -> list[int]:
    return greedy_generate(tiny_model(), prompt, n)


def chunked_engine(
    budget: int, num_blocks: int = 64, prefix_cache: bool = False, max_num_seqs: int = 3
) -> Engine:
    cache = KVCache(CFG, num_blocks, 4, dtype=torch.float32)
    return Engine(
        tiny_model(),
        cache,
        max_num_seqs=max_num_seqs,
        max_batch_tokens=budget,
        prefix_cache=prefix_cache,
        chunked_prefill=True,
    )


@pytest.mark.parametrize(
    ("budget", "num_blocks"),
    [(3, 64), (5, 64), (8, 64), (5, 9)],  # 9 blocks of 4 forces preemptions
)
def test_engine_matches_m1_greedy_with_chunked_prompts(budget: int, num_blocks: int) -> None:
    e = chunked_engine(budget, num_blocks)
    assert e.generate(PROMPTS, 10) == [expected(p, 10) for p in PROMPTS]
    assert e.scheduler.allocator.num_free == num_blocks
    if num_blocks == 9:
        assert e.scheduler.num_preemptions > 0


def test_engine_with_chunks_and_the_prefix_cache_matches_m1_greedy() -> None:
    system = [7, 3, 9, 1, 4, 4, 8, 2, 6, 5, 1, 3]  # 3 blocks of 4, shared
    prompts = [system + p for p in PROMPTS]
    e = chunked_engine(5, prefix_cache=True, max_num_seqs=2)
    assert e.generate(prompts, 6) == [expected(p, 6) for p in prompts]
    assert e.scheduler.num_cached_tokens > 0


def test_ttft_is_recorded_once_when_the_last_chunk_ran() -> None:
    e = chunked_engine(3)
    i = e.add_request(PROMPTS[2], 4)  # 13 tokens: 5 steps of 3, 3, 3, 3, 1
    steps = 0
    while not e.seqs[i].output:
        e.step()
        steps += 1
    assert steps == 5
    assert len(e.times[i].tokens) == 1
    while e.has_work():
        e.step()
    assert len(e.times[i].tokens) == len(e.seqs[i].output) == 4


def test_a_seeded_request_samples_the_same_tokens_chunked_or_not() -> None:
    params = SamplingParams(temperature=0.8, top_p=0.9, seed=11)
    whole = Engine(tiny_model(), KVCache(CFG, 64, 4, dtype=torch.float32), max_num_seqs=3)
    want = whole.generate([PROMPTS[2]], 8, sampling=params)
    e = chunked_engine(3)
    assert e.generate([PROMPTS[2]], 8, sampling=params) == want


@pytest.mark.skipif(not HAS_TRITON, reason="Triton is not installed (no Windows wheels)")
@pytest.mark.parametrize("attention", ["triton", "triton_grouped"])
def test_engine_with_chunks_kernel_and_batched_prefill_matches_m1_greedy(
    attention: str,
) -> None:
    from tinyserve.cuda_graphs import DecodeGraphs

    dev = "cuda" if torch.cuda.is_available() else "cpu"  # the kernels need the GPU there
    w = {n: t.to(dev) for n, t in tiny_model().w.items()}
    model = Qwen3(CFG, w, decode_attention=attention, prefill_attention="batched")  # type: ignore[arg-type]
    cache = KVCache(CFG, 64, 4, dtype=torch.float32, device=dev)
    graphs = DecodeGraphs(model, cache, max_batch=3)  # replays eagerly on a CPU
    e = Engine(
        model, cache, max_num_seqs=3, max_batch_tokens=4, decode_graphs=graphs, chunked_prefill=True
    )
    assert e.generate(PROMPTS, 8) == [expected(p, 8) for p in PROMPTS]
    assert e.num_steps > 0
