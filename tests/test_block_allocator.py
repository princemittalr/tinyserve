"""Spec for tinyserve.block_allocator. Do not edit these tests to make them pass.

The last test is a Hypothesis state machine: it runs thousands of random sequences of
allocate/free calls, including bad ones, and after every call checks that no block is
owned twice, no block leaked and the counts add up. That is M2's acceptance criterion.
"""

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from tinyserve.block_allocator import BlockAllocator, OutOfBlocksError

# ---------------------------------------------------------------- unit tests


def test_fresh_allocator_is_all_free() -> None:
    a = BlockAllocator(8)
    assert (a.num_free, a.num_used) == (8, 0)


def test_rejects_non_positive_size() -> None:
    with pytest.raises(ValueError):
        BlockAllocator(0)


def test_hands_out_ids_in_order() -> None:
    a = BlockAllocator(8)
    assert a.allocate(3) == [0, 1, 2]
    assert a.allocate(2) == [3, 4]
    assert (a.num_free, a.num_used) == (3, 5)


def test_allocate_zero_returns_empty() -> None:
    a = BlockAllocator(4)
    assert a.allocate(0) == []
    assert a.num_free == 4


def test_allocate_negative_is_an_error() -> None:
    with pytest.raises(ValueError):
        BlockAllocator(4).allocate(-1)


def test_can_allocate_every_block() -> None:
    a = BlockAllocator(4)
    assert sorted(a.allocate(4)) == [0, 1, 2, 3]
    assert a.num_free == 0


def test_out_of_blocks_changes_nothing() -> None:
    a = BlockAllocator(4)
    a.allocate(3)
    with pytest.raises(OutOfBlocksError):
        a.allocate(2)  # only 1 free: must not hand out that 1 and then fail
    assert (a.num_free, a.num_used) == (1, 3)
    assert a.allocate(1) == [3]


def test_freed_blocks_are_reused() -> None:
    a = BlockAllocator(4)
    a.allocate(4)
    a.free([1, 2])
    assert (a.num_free, a.num_used) == (2, 2)
    assert sorted(a.allocate(2)) == [1, 2]


def test_double_free_is_caught() -> None:
    a = BlockAllocator(4)
    blocks = a.allocate(2)
    a.free(blocks)
    with pytest.raises(ValueError):
        a.free(blocks)
    assert a.num_free == 4  # the pool did not grow past its size


def test_free_never_allocated_is_caught() -> None:
    a = BlockAllocator(4)
    with pytest.raises(ValueError):
        a.free([0])


def test_free_out_of_range_is_caught() -> None:
    a = BlockAllocator(4)
    a.allocate(4)
    with pytest.raises(ValueError):
        a.free([4])
    with pytest.raises(ValueError):
        a.free([-1])


def test_duplicate_ids_in_one_free_is_caught() -> None:
    a = BlockAllocator(4)
    a.allocate(2)
    with pytest.raises(ValueError):
        a.free([0, 0])
    assert a.num_used == 2


def test_bad_free_changes_nothing() -> None:
    # Block 0 is valid, block 3 was never allocated. Nothing may be freed.
    a = BlockAllocator(4)
    a.allocate(2)
    with pytest.raises(ValueError):
        a.free([0, 3])
    assert (a.num_free, a.num_used) == (2, 2)
    a.free([0, 1])  # both still in use, so this must succeed
    assert a.num_free == 4


# ---------------------------------------------------------------- property test


class AllocatorMachine(RuleBasedStateMachine):
    """Random allocate/free traffic against a simple model of who owns which block.

    ``owners`` maps a fake sequence id to the blocks it holds, like the scheduler will.
    """

    NUM_BLOCKS = 16

    def __init__(self) -> None:
        super().__init__()
        self.alloc = BlockAllocator(self.NUM_BLOCKS)
        self.owners: dict[int, list[int]] = {}
        self.next_seq = 0
        self.ever_freed: list[int] = []

    def _held(self) -> list[int]:
        return [b for blocks in self.owners.values() for b in blocks]

    @rule(n=st.integers(min_value=0, max_value=NUM_BLOCKS + 2))
    def allocate(self, n: int) -> None:
        free_before = self.alloc.num_free
        if n > free_before:
            with pytest.raises(OutOfBlocksError):
                self.alloc.allocate(n)
            return
        blocks = self.alloc.allocate(n)
        assert len(blocks) == n
        assert not set(blocks) & set(self._held()), "handed out a block someone owns"
        self.owners[self.next_seq] = blocks
        self.next_seq += 1

    @precondition(lambda self: bool(self.owners))
    @rule(data=st.data())
    def free_a_sequence(self, data: st.DataObject) -> None:
        seq = data.draw(st.sampled_from(sorted(self.owners)))
        blocks = self.owners.pop(seq)
        self.alloc.free(blocks)
        self.ever_freed.extend(blocks)

    @precondition(lambda self: bool(self.owners))
    @rule(data=st.data())
    def free_part_of_a_sequence(self, data: st.DataObject) -> None:
        # A preempted or shortened sequence can give back only some of its blocks.
        seq = data.draw(st.sampled_from(sorted(self.owners)))
        blocks = self.owners[seq]
        k = data.draw(st.integers(min_value=0, max_value=len(blocks)))
        self.alloc.free(blocks[k:])
        self.ever_freed.extend(blocks[k:])
        self.owners[seq] = blocks[:k]

    @precondition(lambda self: bool(self.ever_freed))
    @rule(data=st.data())
    def double_free_is_rejected(self, data: st.DataObject) -> None:
        b = data.draw(st.sampled_from(self.ever_freed))
        if b in self._held():
            return  # it was freed and then handed out again, so freeing it is legal
        used_before = self.alloc.num_used
        with pytest.raises(ValueError):
            self.alloc.free([b])
        assert self.alloc.num_used == used_before

    @rule(b=st.integers(min_value=-2, max_value=NUM_BLOCKS + 2))
    def freeing_an_unowned_id_is_rejected(self, b: int) -> None:
        if b in self._held():
            return
        with pytest.raises(ValueError):
            self.alloc.free([b])

    @invariant()
    def no_block_owned_twice(self) -> None:
        held = self._held()
        assert len(held) == len(set(held))

    @invariant()
    def ids_in_range(self) -> None:
        assert all(0 <= b < self.NUM_BLOCKS for b in self._held())

    @invariant()
    def no_leak(self) -> None:
        # Every block is either free or held by exactly one sequence.
        assert self.alloc.num_used == len(self._held())
        assert self.alloc.num_free + self.alloc.num_used == self.NUM_BLOCKS


AllocatorMachine.TestCase.settings = settings(max_examples=200, stateful_step_count=50)
TestAllocatorProperties = AllocatorMachine.TestCase


def test_freeing_everything_restores_the_full_pool() -> None:
    # After all traffic ends, the pool must be whole again: allocating every block works.
    a = BlockAllocator(6)
    x, y, z = a.allocate(2), a.allocate(3), a.allocate(1)
    a.free(y)
    w = a.allocate(2)
    a.free(x + z + w)
    assert sorted(a.allocate(6)) == list(range(6))
