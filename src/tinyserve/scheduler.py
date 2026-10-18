"""Continuous-batching scheduler: decides which sequences run in each model step.

Static batching (HF ``generate()`` with a batch) starts B requests together and waits
for the slowest one. Continuous batching instead re-plans before EVERY step (Orca,
OSDI 2022): finished sequences leave right away, and waiting ones join as soon as there
are KV-cache blocks for them. One step of the engine is:

    seqs = scheduler.schedule()      # pick sequences, reserve cache slots for them
    logits = model.forward_batch(...) # one forward pass over all their pending tokens
    scheduler.update(seqs, next_tokens)  # record tokens, retire finished sequences

``schedule()`` works in two phases:
  1. Running sequences first, oldest first. Each needs room for one more token. If the
     cache is full, PREEMPT the newest running sequence: free its blocks and put it back
     at the front of the waiting queue. Its k/v are not saved anywhere; when it is
     admitted again, its prompt and output are re-run as one prefill (recompute; vLLM
     does the same by default). The oldest sequence is never evicted to make room for a
     newer one, so it always makes progress.
  2. Admit waiting sequences in arrival order (FCFS) while they fit: under max_num_seqs,
     under the token budget, and enough free blocks. Stop at the first one that doesn't
     fit, instead of skipping ahead to smaller ones, so a long request can't be starved
     by a stream of short ones.

With ``prefix_cache=True`` (M5) the scheduler also owns a ``RadixCache``:
  - admission first looks up the sequence's tokens in the cache and adopts the matched
    blocks, so only the tokens after the cached prefix are prefilled;
  - a sequence puts its full blocks into the cache right after its prefill, and again
    when it finishes or is preempted, so a later request with the same prefix (or the
    preempted sequence itself) can reuse them, even while this one is still decoding;
  - when blocks run short, cached blocks nobody is using are evicted (LRU) before any
    running sequence is preempted.

Like the allocator, this is plain Python with no tensors. ``tests/test_scheduler.py``
and ``tests/test_prefix_scheduler.py`` are the spec.
"""

from collections import deque

from tinyserve.block_allocator import BlockAllocator
from tinyserve.block_table import BlockTable
from tinyserve.radix_cache import RadixCache
from tinyserve.sequence import SeqStatus, Sequence


class Scheduler:
    def __init__(
        self,
        allocator: BlockAllocator,
        block_size: int,
        max_num_seqs: int = 64,
        max_batch_tokens: int = 2048,
        *,
        prefix_cache: bool = False,
    ) -> None:
        self.allocator = allocator
        self.block_size = block_size
        self.max_num_seqs = max_num_seqs
        self.max_batch_tokens = max_batch_tokens
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []  # in admission order: oldest first
        self.num_preemptions = 0
        self.cache = RadixCache(allocator, block_size) if prefix_cache else None
        # Prompt tokens at admission (preempted sequences count again when re-admitted),
        # and how many of them came from the prefix cache instead of being prefilled.
        self.num_admitted_tokens = 0
        self.num_cached_tokens = 0

    def add(
        self,
        seq_id: int,
        prompt: list[int],
        max_new_tokens: int,
        eos_ids: frozenset[int] = frozenset(),
    ) -> Sequence:
        """Queue a new request. Rejects requests that could never finish."""
        seq = Sequence(
            seq_id, prompt, max_new_tokens, BlockTable(self.allocator, self.block_size), eos_ids
        )
        # After a preemption the whole sequence is recomputed in one step, so its longest
        # possible length must fit in one batch and in the whole cache, or it would wait
        # (or preempt everyone else) forever.
        if seq.max_cached_tokens > self.max_batch_tokens:
            raise ValueError(
                f"request needs up to {seq.max_cached_tokens} tokens in one step, "
                f"max_batch_tokens is {self.max_batch_tokens}"
            )
        if seq.table.blocks_needed(seq.max_cached_tokens) > self.allocator.num_blocks:
            raise ValueError(
                f"request needs up to {seq.max_cached_tokens} tokens of KV cache, "
                f"the cache holds {self.allocator.num_blocks * self.block_size}"
            )
        self.waiting.append(seq)
        return seq

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    @property
    def kv_utilization(self) -> float:
        """Fraction of KV-cache blocks in use right now (cached-only blocks included)."""
        return self.allocator.num_used / self.allocator.num_blocks

    def _evict(self, n: int) -> int:
        """Ask the prefix cache to free ``n`` blocks; returns how many it freed."""
        return self.cache.evict(n) if self.cache is not None else 0

    def _cache_blocks(self, seq: Sequence) -> None:
        """Put the sequence's full blocks into the prefix cache (before releasing them).

        Every token in the table has its k/v written: the table only grows in
        ``schedule()``, and the forward pass that fills those slots runs before the next
        ``update()`` or ``schedule()`` can get here.
        """
        if self.cache is None:
            return
        full = seq.table.num_tokens // self.block_size
        self.cache.insert(seq.tokens[: full * self.block_size], seq.table.blocks[:full])

    def schedule(self) -> list[Sequence]:
        """Pick this step's sequences and append their pending tokens to their tables.

        Returns the running sequences, oldest first. Each one's input for the forward pass
        is ``seq.tokens[-n:]`` with n the tokens just appended, i.e. what ``pending()``
        returned before this call.
        """
        self._reserve_running()
        self._admit_waiting()
        return list(self.running)

    def _reserve_running(self) -> None:
        """Phase 1: one more token for each running sequence, preempting if needed."""
        queue = deque(self.running)  # oldest on the left, newest on the right
        self.running = []
        while queue:
            seq = queue.popleft()
            fits = True
            # TODO(you) 1: loop while this sequence's pending tokens need more NEW blocks
            # than are free. (BlockTable.blocks_needed, allocator.num_free)
            while seq.table.blocks_needed(len(seq.pending())) > self.allocator.num_free:
                short = seq.table.blocks_needed(len(seq.pending())) - self.allocator.num_free
                if self._evict(short):
                    continue  # unused cached blocks go before any running sequence
                if queue:
                    # TODO(you) 2: take the NEWEST remaining running sequence off the queue.
                    self._preempt(queue.pop())
                else:
                    self._preempt(seq)  # nothing newer left to evict: wait our turn
                    fits = False
                    break
            if fits:
                seq.num_scheduled = len(seq.pending())
                seq.table.append(seq.num_scheduled)
                self.running.append(seq)

    def _admit_waiting(self) -> None:
        """Phase 2: FCFS admission while the next waiting sequence fits."""
        budget = self.max_batch_tokens - len(self.running)  # each running seq brings 1 token
        while self.waiting:
            seq = self.waiting[0]
            # TODO(you) 3: stop if the batch already has the maximum number of sequences.
            if len(self.running) >= self.max_num_seqs:
                break
            hit = self._adopt_cached_prefix(seq)
            n = len(seq.pending())
            # TODO(you) 4: stop if this sequence's n tokens don't fit in the token budget.
            if n > budget:
                seq.table.release()  # give back the adopted blocks; they stay cached
                break
            # TODO(you) 5: stop if the cache doesn't have enough free blocks for n tokens.
            short = seq.table.blocks_needed(n) - self.allocator.num_free
            if short > 0 and self._evict(short) < short:
                seq.table.release()
                break
            self.waiting.popleft()
            seq.num_scheduled = n
            seq.table.append(n)
            seq.status = SeqStatus.RUNNING
            self.running.append(seq)
            budget -= n
            self.num_admitted_tokens += len(seq.tokens)
            self.num_cached_tokens += hit

    def _adopt_cached_prefix(self, seq: Sequence) -> int:
        """Start ``seq``'s empty table with its longest cached prefix; return its tokens.

        At least one token is always left to compute: its logits are what samples the
        next token. Whole blocks only, so a 7-token prompt with block size 4 can reuse at
        most 4 tokens.
        """
        if self.cache is None:
            return 0
        limit = (len(seq.tokens) - 1) // self.block_size * self.block_size
        hit = self.cache.match_prefix(seq.tokens[:limit])
        seq.table.adopt(hit)
        return seq.table.num_tokens

    def _preempt(self, seq: Sequence) -> None:
        """Evict a running sequence; it will be recomputed when admitted again.

        With the prefix cache its full blocks are cached first, so the recompute can
        reuse whatever of them has not been evicted by then.
        """
        self._cache_blocks(seq)
        # TODO(you) 6: give all of its blocks back to the allocator.
        seq.table.release()
        seq.status = SeqStatus.WAITING
        seq.num_preemptions += 1
        self.num_preemptions += 1
        # TODO(you) 7: put it at the FRONT of the waiting queue (it arrived before them).
        self.waiting.appendleft(seq)

    def abort(self, seq: Sequence) -> None:
        """Drop a request between steps (its client went away, M6); free its blocks now.

        A running sequence's full blocks go into the prefix cache first, like at finish:
        their k/v are complete, and a retry of the same request can reuse them. Aborting a
        finished sequence does nothing.
        """
        if seq.status is SeqStatus.FINISHED:
            return
        if seq.status is SeqStatus.RUNNING:
            self._cache_blocks(seq)
            self.running.remove(seq)
        else:
            self.waiting.remove(seq)  # a waiting sequence holds no blocks
        seq.table.release()
        seq.status = SeqStatus.FINISHED

    def update(self, seqs: list[Sequence], next_tokens: list[int]) -> list[Sequence]:
        """Record each sequence's new token; retire and return the ones that finished."""
        finished = []
        for seq, tok in zip(seqs, next_tokens, strict=True):
            seq.output.append(tok)
            if seq.num_scheduled > 1 or seq.is_done():
                # Cache right after a prefill, not only at the end: requests that arrive
                # while this one is still decoding can then reuse its prompt (M5 step 4).
                self._cache_blocks(seq)
            # TODO(you) 8: has this sequence hit EOS or max_new_tokens? (a Sequence method)
            if seq.is_done():
                # TODO(you) 9: free its blocks.
                seq.table.release()
                seq.status = SeqStatus.FINISHED
                finished.append(seq)
        self.running = [s for s in self.running if s.status is SeqStatus.RUNNING]
        return finished
