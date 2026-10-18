"""Block allocator for the paged KV cache (the idea behind vLLM's PagedAttention).

The KV cache is one big tensor cut into ``num_blocks`` fixed-size blocks; each block
holds the keys and values of ``block_size`` tokens. A sequence does not get one long
contiguous slab of memory. It gets blocks one at a time as it grows, and they can sit
anywhere in the big tensor. So no memory is reserved for tokens that were never
generated, and a finished sequence's blocks go straight back to the pool.

This file only does the bookkeeping: which block ids are free and which are in use.
It never touches a tensor. Block ids are plain ints ``0 .. num_blocks - 1``.

``tests/test_block_allocator.py`` is the spec. Its property tests are M2's acceptance
criteria: no block is ever handed out twice, a double free is caught, and no block leaks.
"""


class OutOfBlocksError(RuntimeError):
    """Raised when an allocation asks for more blocks than are free.

    The scheduler (M3) will catch this and preempt a sequence, so the allocator must
    leave its state unchanged when it raises.
    """


class BlockAllocator:
    """Hands out and takes back KV-cache block ids.

    Data structures:
        ``_free``: a list used as a stack of free block ids. ``pop()`` and ``append()`` at
            the end are O(1). A freshly built allocator hands out 0, 1, 2, ... in order,
            so the stack starts as ``[num_blocks - 1, ..., 1, 0]`` (0 on top).
        ``_used``: a set of the block ids currently handed out. ``in`` is O(1), which is
            what makes the double-free check cheap.

    Invariant, true after every public call: every id ``0 .. num_blocks - 1`` is in exactly
    one of ``_free`` and ``_used``.
    """

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        self.num_blocks = num_blocks
        self._free: list[int] = list(range(num_blocks - 1, -1, -1))
        self._used: set[int] = set()

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return len(self._used)

    def allocate(self, n: int) -> list[int]:
        """Take ``n`` free blocks and return their ids.

        All or nothing: if fewer than ``n`` blocks are free, raise ``OutOfBlocksError``
        and change nothing. ``n == 0`` returns ``[]``. Negative ``n`` is a ``ValueError``.
        Cost: O(n).
        """
        if n < 0:
            raise ValueError(f"cannot allocate {n} blocks")
        # TODO(you) 1: when must we refuse? Compare n with the number of free blocks.
        if n > self.num_free:
            raise OutOfBlocksError(f"asked for {n} blocks, only {self.num_free} free")
        # TODO(you) 2: take n ids off the top of the free stack (list.pop() takes the last one).
        blocks = [self._free.pop() for _ in range(n)]
        # TODO(you) 3: remember that these ids are now in use (set.update adds many at once).
        self._used.update(blocks)
        return blocks

    def free(self, blocks: list[int]) -> None:
        """Give blocks back to the pool.

        All or nothing: if any id is not currently in use (never allocated, already freed,
        or out of range), or the same id appears twice in ``blocks``, raise ``ValueError``
        and change nothing. Cost: O(len(blocks)).
        """
        if len(set(blocks)) != len(blocks):
            raise ValueError(f"duplicate block ids in free(): {blocks}")
        # Check every id BEFORE changing anything, so a bad call cannot half-free a list.
        for b in blocks:
            # TODO(you) 4: a block may only be freed if it is currently in use.
            if b not in self._used:
                raise ValueError(f"double free or unknown block: {b}")
        for b in blocks:
            # TODO(you) 5: move b from the used set back onto the free stack (two lines).
            self._used.remove(b)
            self._free.append(b)
