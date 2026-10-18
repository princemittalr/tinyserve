"""Radix-tree prefix cache: reuse the KV blocks of prompts that start the same way (M5).

Many requests share a prefix: the same system prompt, the same few-shot examples, the
same document with different questions. Their keys and values for that prefix are
identical, because a token's k/v depend only on the tokens up to it. So once one request
has computed them, the next can point its block table at the same blocks and prefill
only the part after the shared prefix (SGLang's RadixAttention, 2024).

The tree is keyed by token ids, one KV block (``block_size`` tokens) at a time. Each node
holds a run of whole blocks; the path from the root to a node spells out a token prefix,
and the node's blocks hold that prefix's k/v. Example, block size 2, after caching
[1 2 3 4 5 6] and [1 2 3 4 7 8]:

    root ── [1 2 | 3 4] ─┬─ [5 6]      blocks of [1 2 3 4] are shared by both prompts
                          └─ [7 8]

Only whole blocks are cached. A partly filled last block is still being written by its
sequence, so it is never shared (see block_allocator.py).

The cache holds one reference (``BlockAllocator.share``) on every block in the tree, so
a cached block stays out of the free pool after its sequence finishes. Memory is
reclaimed by ``evict``: least recently used leaves first, and only blocks no sequence is
using (reference count 1, the cache's own). Evicting a leaf can turn its parent into a
leaf, which then becomes a candidate too.

Like the allocator, this is plain Python bookkeeping with no tensors.
``tests/test_radix_cache.py`` is the spec.
"""

import heapq
from itertools import count

from tinyserve.block_allocator import BlockAllocator


class _Node:
    """A run of whole cached blocks: ``len(tokens) == len(blocks) * block_size``.

    ``children`` is keyed by the child's FIRST block of tokens (a tuple of block_size
    ids). Two children can never start with the same block: they would share it, and
    the shared part would be split into its own node.
    """

    __slots__ = ("blocks", "children", "last_access", "parent", "tokens")

    def __init__(self, tokens: tuple[int, ...], blocks: list[int], parent: "_Node | None") -> None:
        self.tokens = tokens
        self.blocks = blocks
        self.parent = parent
        self.children: dict[tuple[int, ...], _Node] = {}
        self.last_access = 0


class RadixCache:
    def __init__(self, allocator: BlockAllocator, block_size: int) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.allocator = allocator
        self.block_size = block_size
        self.root = _Node((), [], None)
        self.num_cached_blocks = 0
        self._clock = count(1)  # LRU time: a counter, so tests are deterministic

    def _first_block(self, tokens: tuple[int, ...] | list[int], i: int) -> tuple[int, ...]:
        """The block of tokens starting at ``i`` (shorter than a block at the end)."""
        return tuple(tokens[i : i + self.block_size])

    def _common_blocks(self, node: _Node, tokens: list[int], i: int) -> int:
        """How many of ``node``'s blocks equal the blocks of ``tokens`` starting at ``i``."""
        bs = self.block_size
        k = 0
        # Compare one whole block at a time; a partial block of `tokens` never matches.
        while k < len(node.blocks) and i + (k + 1) * bs <= len(tokens):
            if node.tokens[k * bs : (k + 1) * bs] != tuple(tokens[i + k * bs : i + (k + 1) * bs]):
                break
            k += 1
        return k

    def _split(self, node: _Node, k: int) -> _Node:
        """Cut ``node`` after its first ``k`` blocks; return the new upper part.

        parent ── node[0:n]   becomes   parent ── upper[0:k] ── node[k:n]
        """
        assert 0 < k < len(node.blocks) and node.parent is not None
        bs = self.block_size
        upper = _Node(node.tokens[: k * bs], node.blocks[:k], node.parent)
        upper.last_access = node.last_access
        node.parent.children[self._first_block(node.tokens, 0)] = upper
        node.tokens, node.blocks, node.parent = node.tokens[k * bs :], node.blocks[k:], upper
        upper.children[self._first_block(node.tokens, 0)] = node
        return upper

    def match_prefix(self, tokens: list[int]) -> list[int]:
        """Blocks holding the k/v of the longest cached prefix of ``tokens``, in order.

        Whole blocks only: the match covers ``len(result) * block_size`` tokens. Marks
        the matched nodes as just used (LRU). The caller does NOT get a reference: call
        ``allocator.share`` on the result before anything can evict it.
        A match that ends inside a node splits it, so every node is either fully
        matched or not matched at all; ``evict`` relies on that.
        """
        now = next(self._clock)
        node, i, out = self.root, 0, []
        while True:
            # TODO(you) 1: the child whose first block equals the block of tokens at i
            # (None if there is none). Use self._first_block and node.children.get.
            child = node.children.get(self._first_block(tokens, i))
            if child is None:
                break
            k = self._common_blocks(child, tokens, i)
            # TODO(you) 2: when does the match stop inside `child`? Split it there.
            if k < len(child.blocks):
                child = self._split(child, k)
            child.last_access = now
            out.extend(child.blocks)
            # TODO(you) 3: move past the tokens `child` covers.
            i += len(child.tokens)
            node = child
        return out

    def insert(self, tokens: list[int], blocks: list[int]) -> int:
        """Cache ``blocks``, which hold the k/v of ``tokens`` (whole blocks only).

        Parts already in the tree keep the tree's blocks; the caller's duplicate blocks
        for them are not cached (the caller still owns them and frees them as usual).
        The rest becomes a new leaf, and the cache takes a reference on its blocks.
        Returns how many blocks were newly cached.
        """
        if len(tokens) != len(blocks) * self.block_size:
            raise ValueError(f"{len(tokens)} tokens do not fill {len(blocks)} blocks exactly")
        now = next(self._clock)
        node, i = self.root, 0
        while i < len(tokens):
            child = node.children.get(self._first_block(tokens, i))
            if child is None:
                j = i // self.block_size  # blocks[j:] are not in the tree yet
                leaf = _Node(tuple(tokens[i:]), blocks[j:], node)
                # TODO(you) 4: the cache becomes one more owner of the leaf's blocks.
                self.allocator.share(leaf.blocks)
                leaf.last_access = now
                node.children[self._first_block(tokens, i)] = leaf
                self.num_cached_blocks += len(leaf.blocks)
                return len(leaf.blocks)
            k = self._common_blocks(child, tokens, i)
            if k < len(child.blocks):
                child = self._split(child, k)
            child.last_access = now
            i += len(child.tokens)
            node = child
        return 0

    def _evictable(self, node: _Node) -> bool:
        """A leaf (not the root) whose blocks only the cache holds."""
        if node is self.root or node.children:
            return False
        # TODO(you) 5: True if only the cache holds each of its blocks (ref_count 1).
        return all(self.allocator.ref_count(b) == 1 for b in node.blocks)

    def evict(self, num_blocks: int) -> int:
        """Free at least ``num_blocks`` cached blocks if possible, oldest leaves first.

        Whole leaves are evicted, so this may free a few more than asked. Blocks a
        sequence still uses are never touched. Returns how many blocks were freed
        (fewer than asked if nothing else is evictable).
        Cost: O(N log N) for N nodes (one heap over the leaves).
        """
        heap = [(n.last_access, id(n), n) for n in self._nodes() if self._evictable(n)]
        heapq.heapify(heap)
        freed = 0
        while freed < num_blocks and heap:
            # TODO(you) 6: take the least recently used evictable leaf off the heap.
            _, _, node = heapq.heappop(heap)
            # TODO(you) 7: drop the cache's reference on its blocks.
            self.allocator.free(node.blocks)
            freed += len(node.blocks)
            self.num_cached_blocks -= len(node.blocks)
            parent = node.parent
            assert parent is not None
            del parent.children[self._first_block(node.tokens, 0)]
            # TODO(you) 8: the parent may have just become an evictable leaf.
            if self._evictable(parent):
                heapq.heappush(heap, (parent.last_access, id(parent), parent))
        return freed

    def _nodes(self) -> list[_Node]:
        """Every node except the root."""
        out, stack = [], list(self.root.children.values())
        while stack:
            node = stack.pop()
            out.append(node)
            stack.extend(node.children.values())
        return out

    def cached_blocks(self) -> list[int]:
        """Every block the cache holds a reference on (for tests and debugging)."""
        return [b for n in self._nodes() for b in n.blocks]
