"""Spec for tinyserve.block_table. Do not edit these tests to make them pass."""

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from tinyserve.block_allocator import BlockAllocator, OutOfBlocksError
from tinyserve.block_table import BlockTable

# ---------------------------------------------------------------- unit tests


def test_starts_empty() -> None:
    t = BlockTable(BlockAllocator(4), block_size=4)
    assert (t.blocks, t.num_tokens) == ([], 0)


def test_rejects_non_positive_block_size() -> None:
    with pytest.raises(ValueError):
        BlockTable(BlockAllocator(4), block_size=0)


@pytest.mark.parametrize(
    ("n", "expected"), [(0, 0), (1, 1), (3, 1), (4, 1), (5, 2), (8, 2), (9, 3)]
)
def test_blocks_needed_rounds_up(n: int, expected: int) -> None:
    t = BlockTable(BlockAllocator(8), block_size=4)
    assert t.blocks_needed(n) == expected


def test_blocks_needed_counts_room_in_last_block() -> None:
    t = BlockTable(BlockAllocator(8), block_size=4)
    t.append(3)  # one block, one slot left in it
    assert t.blocks_needed(1) == 0
    assert t.blocks_needed(2) == 1
    assert t.blocks_needed(6) == 2


def test_blocks_needed_changes_nothing() -> None:
    a = BlockAllocator(8)
    t = BlockTable(a, block_size=4)
    t.blocks_needed(10)
    assert (t.blocks, t.num_tokens, a.num_free) == ([], 0, 8)


def test_append_returns_slots_in_order() -> None:
    t = BlockTable(BlockAllocator(8), block_size=4)
    assert t.append(6) == [0, 1, 2, 3, 4, 5]  # blocks [0, 1] from a fresh allocator
    assert t.blocks == [0, 1]
    assert t.num_tokens == 6


def test_decode_appends_one_token_at_a_time() -> None:
    a = BlockAllocator(8)
    t = BlockTable(a, block_size=4)
    t.append(3)
    assert t.append(1) == [3]  # fills the last slot of block 0, no new block
    assert a.num_used == 1
    assert t.append(1) == [4]  # first slot of a new block
    assert a.num_used == 2


def test_slot_follows_the_block_table_not_the_position() -> None:
    # The docstring example: block_size 4, blocks [7, 2].
    a = BlockAllocator(8)
    a.allocate(7)  # takes 0..6, so the next allocation is block 7
    t = BlockTable(a, block_size=4)
    t.append(4)
    a.free([2])  # block 2 is now on top of the free stack
    t.append(4)
    assert t.blocks == [7, 2]
    assert [t.slot(p) for p in range(8)] == [28, 29, 30, 31, 8, 9, 10, 11]


def test_slot_out_of_range() -> None:
    t = BlockTable(BlockAllocator(4), block_size=4)
    t.append(3)
    with pytest.raises(IndexError):
        t.slot(3)  # room is reserved in the block, but position 3 is not filled yet
    with pytest.raises(IndexError):
        t.slot(-1)


def test_out_of_blocks_leaves_table_unchanged() -> None:
    a = BlockAllocator(2)
    t = BlockTable(a, block_size=4)
    t.append(5)
    with pytest.raises(OutOfBlocksError):
        t.append(4)  # needs 1 more block, none free
    assert (t.blocks, t.num_tokens) == ([0, 1], 5)
    assert t.append(3) == [5, 6, 7]  # still fits in the last block


def test_release_returns_every_block() -> None:
    a = BlockAllocator(4)
    t = BlockTable(a, block_size=4)
    t.append(10)
    t.release()
    assert (t.blocks, t.num_tokens, a.num_free) == ([], 0, 4)
    t.release()  # releasing an empty table is a no-op, not a double free
    assert a.num_free == 4


def test_negative_append_is_an_error() -> None:
    with pytest.raises(ValueError):
        BlockTable(BlockAllocator(4), block_size=4).append(-1)


# ---------------------------------------------------------------- property test


class TablesMachine(RuleBasedStateMachine):
    """Several sequences growing and finishing on one shared allocator."""

    NUM_BLOCKS = 12
    BLOCK_SIZE = 4

    def __init__(self) -> None:
        super().__init__()
        self.alloc = BlockAllocator(self.NUM_BLOCKS)
        self.tables: list[BlockTable] = []

    @rule()
    def new_sequence(self) -> None:
        self.tables.append(BlockTable(self.alloc, self.BLOCK_SIZE))

    @precondition(lambda self: bool(self.tables))
    @rule(data=st.data(), n=st.integers(min_value=0, max_value=20))
    def append(self, data: st.DataObject, n: int) -> None:
        t = data.draw(st.sampled_from(self.tables))
        before = (list(t.blocks), t.num_tokens)
        if t.blocks_needed(n) > self.alloc.num_free:
            with pytest.raises(OutOfBlocksError):
                t.append(n)
            assert (t.blocks, t.num_tokens) == before
            return
        slots = t.append(n)
        assert slots == [t.slot(p) for p in range(before[1], before[1] + n)]

    @precondition(lambda self: bool(self.tables))
    @rule(data=st.data())
    def finish_sequence(self, data: st.DataObject) -> None:
        t = data.draw(st.sampled_from(self.tables))
        t.release()
        self.tables.remove(t)

    @invariant()
    def block_count_matches_tokens(self) -> None:
        for t in self.tables:
            assert len(t.blocks) == -(-t.num_tokens // self.BLOCK_SIZE)

    @invariant()
    def no_two_tokens_share_a_slot(self) -> None:
        slots = [t.slot(p) for t in self.tables for p in range(t.num_tokens)]
        assert len(slots) == len(set(slots))
        assert all(0 <= s < self.NUM_BLOCKS * self.BLOCK_SIZE for s in slots)

    @invariant()
    def no_leak(self) -> None:
        assert self.alloc.num_used == sum(len(t.blocks) for t in self.tables)


TablesMachine.TestCase.settings = settings(max_examples=200, stateful_step_count=50)
TestBlockTableProperties = TablesMachine.TestCase
