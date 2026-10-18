"""Spec for tinyserve.radix_cache (M5). Do not edit these tests to make them pass.

Block size 2 in most tests, so token lists are short: [1, 2, 3, 4] is two blocks.

The last test is a Hypothesis state machine. It plays the scheduler: requests with
random prompts from a tiny vocabulary (so prefixes collide often) reuse what the cache
matches, allocate the rest, insert themselves, and finish at random times; eviction runs
at random. Every block remembers which token prefix it was computed for, and every match
must return blocks computed for exactly the query's prefix. That is the safety property:
a sequence never reads another prompt's keys and values.
"""

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from tinyserve.block_allocator import BlockAllocator
from tinyserve.radix_cache import RadixCache

BS = 2


def setup(num_blocks: int = 32) -> tuple[BlockAllocator, RadixCache]:
    a = BlockAllocator(num_blocks)
    return a, RadixCache(a, BS)


def run(a: BlockAllocator, c: RadixCache, tokens: list[int]) -> list[int]:
    """What a finished request does: blocks for all its tokens, insert, free its own refs."""
    blocks = a.allocate(len(tokens) // BS)
    c.insert(tokens, blocks)
    a.free(blocks)
    return blocks


# --- match and insert -------------------------------------------------------------------


def test_empty_cache_matches_nothing() -> None:
    _, c = setup()
    assert c.match_prefix([1, 2, 3, 4]) == []
    assert c.match_prefix([]) == []


def test_rejects_bad_block_size() -> None:
    with pytest.raises(ValueError):
        RadixCache(BlockAllocator(4), 0)


def test_insert_then_match_the_same_tokens() -> None:
    a, c = setup()
    blocks = run(a, c, [1, 2, 3, 4])
    assert c.match_prefix([1, 2, 3, 4]) == blocks
    assert c.num_cached_blocks == 2


def test_match_returns_the_longest_cached_prefix() -> None:
    a, c = setup()
    blocks = run(a, c, [1, 2, 3, 4])
    assert c.match_prefix([1, 2, 3, 4, 5, 6]) == blocks  # longer query
    assert c.match_prefix([1, 2, 9, 9]) == blocks[:1]  # diverges in block 2
    assert c.match_prefix([1, 9, 3, 4]) == []  # diverges inside block 1


def test_only_whole_blocks_match() -> None:
    a, c = setup()
    blocks = run(a, c, [1, 2, 3, 4])
    assert c.match_prefix([1, 2, 3]) == blocks[:1]  # [3] is half a block
    assert c.match_prefix([1]) == []


def test_insert_needs_whole_blocks() -> None:
    a, c = setup()
    blocks = a.allocate(2)
    with pytest.raises(ValueError):
        c.insert([1, 2, 3], blocks)
    with pytest.raises(ValueError):
        c.insert([1, 2, 3, 4, 5, 6], blocks)


def test_cache_holds_a_reference_on_what_it_caches() -> None:
    a, c = setup(8)
    blocks = a.allocate(2)
    assert c.insert([1, 2, 3, 4], blocks) == 2
    assert [a.ref_count(b) for b in blocks] == [2, 2]
    a.free(blocks)  # the sequence finishes: the blocks stay cached, out of the pool
    assert [a.ref_count(b) for b in blocks] == [1, 1]
    assert a.num_free == 6


def test_inserting_a_cached_prefix_again_keeps_the_trees_blocks() -> None:
    a, c = setup()
    first = run(a, c, [1, 2, 3, 4])
    # A second request with the same prompt, computed before the first was cached.
    dup = a.allocate(3)
    assert c.insert([1, 2, 3, 4, 5, 6], dup) == 1  # only the new last block is cached
    assert c.match_prefix([1, 2, 3, 4, 5, 6]) == first + dup[2:]
    assert [a.ref_count(b) for b in dup] == [1, 1, 2]  # duplicates stay the caller's alone
    a.free(dup)
    assert a.num_used == 3
    assert sorted(c.cached_blocks()) == sorted(first + dup[2:])


def test_diverging_prompts_share_their_common_blocks() -> None:
    a, c = setup()
    x = run(a, c, [1, 2, 3, 4, 5, 6])
    y = a.allocate(3)
    assert c.insert([1, 2, 3, 4, 7, 8], x[:2] + y[2:]) == 1  # reuses x's first 2 blocks
    a.free(y)
    assert c.match_prefix([1, 2, 3, 4, 5, 6]) == x
    assert c.match_prefix([1, 2, 3, 4, 7, 8]) == x[:2] + y[2:]
    assert c.match_prefix([1, 2, 3, 4, 9, 9]) == x[:2]
    assert c.num_cached_blocks == 4


def test_split_inside_a_node_keeps_every_path() -> None:
    a, c = setup()
    x = run(a, c, [1, 2, 3, 4, 5, 6, 7, 8])  # one node of 4 blocks
    assert c.match_prefix([1, 2, 3, 4, 0, 0]) == x[:2]  # splits it after block 2
    y = run(a, c, [1, 2, 9, 9])  # splits again after block 1
    assert c.match_prefix([1, 2, 3, 4, 5, 6, 7, 8]) == x
    assert c.match_prefix([1, 2, 9, 9]) == x[:1] + y[1:]
    assert c.num_cached_blocks == 5


# --- eviction ---------------------------------------------------------------------------


def test_evict_frees_least_recently_used_first() -> None:
    a, c = setup(8)
    old = run(a, c, [1, 2])
    new = run(a, c, [3, 4])
    c.match_prefix([1, 2])  # touch `old`: now `new` is the least recently used
    assert c.evict(1) == 1
    assert c.match_prefix([3, 4]) == []
    assert c.match_prefix([1, 2]) == old
    assert a.ref_count(new[0]) == 0  # back in the pool


def test_evict_never_frees_blocks_a_sequence_uses() -> None:
    a, c = setup(8)
    blocks = a.allocate(2)
    c.insert([1, 2, 3, 4], blocks)  # the sequence still holds its blocks
    assert c.evict(10) == 0
    assert c.match_prefix([1, 2, 3, 4]) == blocks
    a.free(blocks)
    assert c.evict(10) == 2
    assert (a.num_free, c.num_cached_blocks) == (8, 0)


def test_evict_keeps_a_shared_prefix_while_a_sequence_reads_it() -> None:
    a, c = setup()
    x = run(a, c, [1, 2, 3, 4, 5, 6])
    hit = c.match_prefix([1, 2, 3, 4, 7, 7])  # a new request reuses the first 2 blocks
    a.share(hit)
    assert c.evict(10) == 1  # only [5 6]: the shared part is in use
    assert c.match_prefix([1, 2, 3, 4]) == x[:2]
    a.free(hit)
    assert c.evict(10) == 2


def test_evicting_a_leaf_makes_its_parent_evictable() -> None:
    a, c = setup()
    run(a, c, [1, 2, 3, 4])
    run(a, c, [1, 2, 3, 4, 5, 6])  # child under the first
    assert c.evict(3) == 3  # leaf [5 6] first, then its parent [1 2 3 4]
    assert c.num_cached_blocks == 0
    assert a.num_free == 32


def test_evict_whole_leaves_may_free_more_than_asked() -> None:
    a, c = setup(8)
    run(a, c, [1, 2, 3, 4, 5, 6])
    assert c.evict(1) == 3  # one leaf of 3 blocks
    assert c.evict(1) == 0  # nothing left


def test_evict_zero_does_nothing() -> None:
    a, c = setup()
    run(a, c, [1, 2])
    assert c.evict(0) == 0
    assert c.num_cached_blocks == 1


def test_insert_counts_as_a_use_for_lru() -> None:
    a, c = setup(8)
    run(a, c, [1, 2])
    run(a, c, [3, 4])
    run(a, c, [1, 2])  # inserting [1 2] again touches it
    c.evict(1)
    assert c.match_prefix([1, 2]) != []
    assert c.match_prefix([3, 4]) == []


# --- property test ----------------------------------------------------------------------

prompts = st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=9)


class PrefixCacheMachine(RuleBasedStateMachine):
    """The scheduler's use of the cache, against a model of what each block holds.

    ``content[b]`` is the token prefix block b's k/v were computed for, set when the
    block is allocated. ``running`` holds the blocks of requests that have not finished.
    """

    NUM_BLOCKS = 24

    def __init__(self) -> None:
        super().__init__()
        self.alloc = BlockAllocator(self.NUM_BLOCKS)
        self.cache = RadixCache(self.alloc, BS)
        self.content: dict[int, tuple[int, ...]] = {}
        self.running: list[list[int]] = []

    def _check_match(self, tokens: list[int]) -> list[int]:
        hit = self.cache.match_prefix(tokens)
        assert len(hit) <= len(tokens) // BS
        for j, b in enumerate(hit):
            assert self.content[b] == tuple(tokens[: (j + 1) * BS]), "wrong prefix's k/v"
        return hit

    @rule(tokens=prompts, finish=st.booleans())
    def request(self, tokens: list[int], finish: bool) -> None:
        hit = self._check_match(tokens)
        self.alloc.share(hit)
        need = -(-len(tokens) // BS) - len(hit)  # ceil: the last block may be partial
        if need > self.alloc.num_free:
            self.cache.evict(need - self.alloc.num_free)
        if need > self.alloc.num_free:
            self.alloc.free(hit)  # does not fit: give up, like a request left waiting
            return
        new = self.alloc.allocate(need)
        for j, b in enumerate(new, start=len(hit)):
            self.content[b] = tuple(tokens[: (j + 1) * BS])
        blocks = hit + new
        full = len(tokens) // BS
        self.cache.insert(tokens[: full * BS], blocks[:full])
        if finish:
            self.alloc.free(blocks)
        else:
            self.running.append(blocks)

    @precondition(lambda self: bool(self.running))
    @rule(data=st.data())
    def finish_one(self, data: st.DataObject) -> None:
        i = data.draw(st.integers(min_value=0, max_value=len(self.running) - 1))
        self.alloc.free(self.running.pop(i))

    @rule(n=st.integers(min_value=0, max_value=NUM_BLOCKS))
    def evict(self, n: int) -> None:
        cached_before = self.cache.num_cached_blocks
        freed = self.cache.evict(n)
        assert freed <= cached_before
        assert self.cache.num_cached_blocks == cached_before - freed
        if freed < n:
            # Asked for more than could be freed: everything left must be in use.
            held = {b for blocks in self.running for b in blocks}
            assert set(self.cache.cached_blocks()) <= held or not self.cache.cached_blocks()

    @rule(tokens=prompts)
    def query(self, tokens: list[int]) -> None:
        self._check_match(tokens)

    @invariant()
    def ref_counts_match_owners(self) -> None:
        cached = self.cache.cached_blocks()
        assert len(cached) == len(set(cached)) == self.cache.num_cached_blocks
        for b in range(self.NUM_BLOCKS):
            owners = sum(b in blocks for blocks in self.running) + (b in cached)
            assert self.alloc.ref_count(b) == owners

    @invariant()
    def no_leak(self) -> None:
        held = {b for blocks in self.running for b in blocks} | set(self.cache.cached_blocks())
        assert self.alloc.num_used == len(held)

    def teardown(self) -> None:
        # Once every request has finished, eviction can give back the whole pool.
        for blocks in self.running:
            self.alloc.free(blocks)
        self.cache.evict(self.NUM_BLOCKS)
        assert self.cache.num_cached_blocks == 0
        assert self.alloc.num_free == self.NUM_BLOCKS


PrefixCacheMachine.TestCase.settings = settings(max_examples=300, stateful_step_count=40)
TestPrefixCacheProperties = PrefixCacheMachine.TestCase


def test_full_prompt_matches_right_after_insert() -> None:
    a, c = setup()
    for tokens in ([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 7], [1, 2, 3, 8], [1, 2, 3, 4, 5, 6, 9]):
        hit = c.match_prefix(tokens)
        a.share(hit)
        own = a.allocate(-(-len(tokens) // BS) - len(hit))
        full = len(tokens) // BS
        c.insert(tokens[: full * BS], (hit + own)[:full])
        a.free(hit + own)
        assert len(c.match_prefix(tokens)) == full
