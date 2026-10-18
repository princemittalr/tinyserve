"""Per-sequence block table: maps a sequence's token positions to slots in the KV cache.

The KV cache (step 3) is one tensor of ``num_blocks * block_size`` token slots. Block b
owns slots ``b * block_size`` to ``b * block_size + block_size - 1``. A sequence's
blocks can be anywhere in the cache, in any order; its block table lists them in the
order the sequence uses them. Example with block_size 4 and blocks [7, 2]:

    position:  0  1  2  3 | 4  5  6  7
    block:     7  7  7  7 | 2  2  2  2
    slot:     28 29 30 31 | 8  9 10 11

Like the allocator, this is plain Python bookkeeping. ``tests/test_block_table.py`` is
the spec.
"""

from tinyserve.block_allocator import BlockAllocator


class BlockTable:
    """The blocks one sequence owns, and how many token positions it has filled.

    Invariant: ``len(blocks) == ceil(num_tokens / block_size)``. A sequence never holds an
    empty block, and its last block may be partly filled.
    """

    def __init__(self, allocator: BlockAllocator, block_size: int) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.allocator = allocator
        self.block_size = block_size
        self.blocks: list[int] = []
        self.num_tokens = 0

    def blocks_needed(self, n: int) -> int:
        """How many NEW blocks appending ``n`` tokens would take. Changes nothing.

        The scheduler (M3) asks this before admitting work, to check it fits.
        """
        total = self.num_tokens + n
        # TODO(you) 1: blocks needed to hold `total` tokens is total / block_size rounded
        # UP. With integers: (total + block_size - 1) // block_size.
        blocks_for_total = (total + self.block_size - 1) // self.block_size
        return blocks_for_total - len(self.blocks)

    def append(self, n: int) -> list[int]:
        """Reserve room for ``n`` more tokens and return their slots, in position order.

        All or nothing: if the allocator is out of blocks, ``OutOfBlocksError`` propagates
        and this table is unchanged.
        """
        if n < 0:
            raise ValueError(f"cannot append {n} tokens")
        new_blocks = self.allocator.allocate(self.blocks_needed(n))
        # TODO(you) 2: add the new blocks to the end of this sequence's block list.
        self.blocks.extend(new_blocks)
        start = self.num_tokens
        self.num_tokens += n
        return [self.slot(pos) for pos in range(start, start + n)]

    def slot(self, pos: int) -> int:
        """KV-cache slot of token position ``pos`` (see the table in the module docstring)."""
        if not 0 <= pos < self.num_tokens:
            raise IndexError(f"position {pos} not in 0..{self.num_tokens - 1}")
        # TODO(you) 3: which entry of self.blocks holds position pos? (integer division)
        block = self.blocks[pos // self.block_size]
        # TODO(you) 4: how far into that block is pos? (remainder)
        offset = pos % self.block_size
        return block * self.block_size + offset

    def release(self) -> None:
        """Give every block back to the allocator, e.g. when the sequence finishes."""
        # TODO(you) 5: free this sequence's blocks.
        self.allocator.free(self.blocks)
        self.blocks = []
        self.num_tokens = 0
