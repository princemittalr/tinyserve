"""Block allocator for the paged KV cache (the idea behind vLLM's PagedAttention).

The KV cache is one big tensor cut into ``num_blocks`` fixed-size blocks; each block
holds the keys and values of ``block_size`` tokens. A sequence does not get one long
contiguous slab of memory. It gets blocks one at a time as it grows, and they can sit
anywhere in the big tensor. So no memory is reserved for tokens that were never
generated, and a finished sequence's blocks go straight back to the pool.

M5 adds sharing. Two requests with the same prompt prefix compute the same keys and
values for it, so the prefix cache lets the second one point at the first one's blocks
instead of recomputing them. A block can then have several owners (sequences, and the
prefix cache itself), so the allocator counts them:

    allocate(n)   each new block starts with 1 reference
    share(blocks) one more owner for each block (it must already be in use)
    free(blocks)  one owner fewer; a block goes back to the pool when its count hits 0

Shared blocks are never written to: only full blocks are shared, and a sequence only
ever writes past its last token, so it writes into blocks it alone owns.

This file only does the bookkeeping: which block ids are free and who holds the rest.
It never touches a tensor. Block ids are plain ints ``0 .. num_blocks - 1``.

``tests/test_block_allocator.py`` is the spec. Its property tests are M2's acceptance
criteria (no block handed out twice, double free caught, no leak), plus M5's: a shared
block stays in use until its last owner frees it.
"""


class OutOfBlocksError(RuntimeError):
    """Raised when an allocation asks for more blocks than are free.

    The scheduler (M3) will catch this and preempt a sequence, so the allocator must
    leave its state unchanged when it raises.
    """


class BlockAllocator:
    """Hands out and takes back KV-cache block ids, counting each block's owners.

    Data structures:
        ``_free``: a list used as a stack of free block ids. ``pop()`` and ``append()`` at
            the end are O(1). A freshly built allocator hands out 0, 1, 2, ... in order,
            so the stack starts as ``[num_blocks - 1, ..., 1, 0]`` (0 on top).
        ``_refs``: block id -> number of owners, for every block currently handed out.
            ``in`` and updates are O(1), which keeps the double-free check cheap.

    Invariant, true after every public call: every id ``0 .. num_blocks - 1`` is in exactly
    one of ``_free`` and ``_refs``, and every count in ``_refs`` is at least 1.
    """

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        self.num_blocks = num_blocks
        self._free: list[int] = list(range(num_blocks - 1, -1, -1))
        self._refs: dict[int, int] = {}

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        """Blocks with at least one owner (a shared block counts once)."""
        return len(self._refs)

    def ref_count(self, block: int) -> int:
        """Owners of ``block`` right now; 0 if it is free."""
        return self._refs.get(block, 0)

    def allocate(self, n: int) -> list[int]:
        """Take ``n`` free blocks and return their ids, each with 1 reference.

        All or nothing: if fewer than ``n`` blocks are free, raise ``OutOfBlocksError``
        and change nothing. ``n == 0`` returns ``[]``. Negative ``n`` is a ``ValueError``.
        Cost: O(n).
        """
        if n < 0:
            raise ValueError(f"cannot allocate {n} blocks")
        if n > self.num_free:
            raise OutOfBlocksError(f"asked for {n} blocks, only {self.num_free} free")
        blocks = [self._free.pop() for _ in range(n)]
        for b in blocks:
            self._refs[b] = 1
        return blocks

    def share(self, blocks: list[int]) -> None:
        """Add one owner to each block, e.g. a new sequence reusing a cached prefix.

        All or nothing: every id must be in use (you cannot share a free block: its
        contents are garbage and it could be handed out at any time), and no id may
        appear twice; otherwise raise ``ValueError`` and change nothing.
        Cost: O(len(blocks)).
        """
        if len(set(blocks)) != len(blocks):
            raise ValueError(f"duplicate block ids in share(): {blocks}")
        for b in blocks:
            if b not in self._refs:
                raise ValueError(f"cannot share a block that is not in use: {b}")
        for b in blocks:
            self._refs[b] += 1

    def free(self, blocks: list[int]) -> None:
        """Drop one owner from each block; blocks nobody owns go back to the pool.

        All or nothing: if any id is not currently in use (never allocated, already freed
        by its last owner, or out of range), or the same id appears twice in ``blocks``,
        raise ``ValueError`` and change nothing. Cost: O(len(blocks)).
        """
        if len(set(blocks)) != len(blocks):
            raise ValueError(f"duplicate block ids in free(): {blocks}")
        # Check every id BEFORE changing anything, so a bad call cannot half-free a list.
        for b in blocks:
            if b not in self._refs:
                raise ValueError(f"double free or unknown block: {b}")
        for b in blocks:
            self._refs[b] -= 1
            if self._refs[b] == 0:
                del self._refs[b]
                self._free.append(b)
