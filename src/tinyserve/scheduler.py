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

Like the allocator, this is plain Python with no tensors. ``tests/test_scheduler.py``
is the spec.
"""

from collections import deque

from tinyserve.block_allocator import BlockAllocator
from tinyserve.block_table import BlockTable
from tinyserve.sequence import SeqStatus, Sequence


class Scheduler:
    def __init__(
        self,
        allocator: BlockAllocator,
        block_size: int,
        max_num_seqs: int = 64,
        max_batch_tokens: int = 2048,
    ) -> None:
        self.allocator = allocator
        self.block_size = block_size
        self.max_num_seqs = max_num_seqs
        self.max_batch_tokens = max_batch_tokens
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []  # in admission order: oldest first
        self.num_preemptions = 0

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
        """Fraction of KV-cache blocks in use right now."""
        return self.allocator.num_used / self.allocator.num_blocks

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
                if queue:
                    # TODO(you) 2: take the NEWEST remaining running sequence off the queue.
                    self._preempt(queue.pop())
                else:
                    self._preempt(seq)  # nothing newer left to evict: wait our turn
                    fits = False
                    break
            if fits:
                seq.table.append(len(seq.pending()))
                self.running.append(seq)

    def _admit_waiting(self) -> None:
        """Phase 2: FCFS admission while the next waiting sequence fits."""
        budget = self.max_batch_tokens - len(self.running)  # each running seq brings 1 token
        while self.waiting:
            seq = self.waiting[0]
            n = len(seq.pending())
            # TODO(you) 3: stop if the batch already has the maximum number of sequences.
            if len(self.running) >= self.max_num_seqs:
                break
            # TODO(you) 4: stop if this sequence's n tokens don't fit in the token budget.
            if n > budget:
                break
            # TODO(you) 5: stop if the cache doesn't have enough free blocks for n tokens.
            if seq.table.blocks_needed(n) > self.allocator.num_free:
                break
            self.waiting.popleft()
            seq.table.append(n)
            seq.status = SeqStatus.RUNNING
            self.running.append(seq)
            budget -= n

    def _preempt(self, seq: Sequence) -> None:
        """Evict a running sequence; it will be recomputed when admitted again."""
        # TODO(you) 6: give all of its blocks back to the allocator.
        seq.table.release()
        seq.status = SeqStatus.WAITING
        seq.num_preemptions += 1
        self.num_preemptions += 1
        # TODO(you) 7: put it at the FRONT of the waiting queue (it arrived before them).
        self.waiting.appendleft(seq)

    def update(self, seqs: list[Sequence], next_tokens: list[int]) -> list[Sequence]:
        """Record each sequence's new token; retire and return the ones that finished."""
        finished = []
        for seq, tok in zip(seqs, next_tokens, strict=True):
            seq.output.append(tok)
            # TODO(you) 8: has this sequence hit EOS or max_new_tokens? (a Sequence method)
            if seq.is_done():
                # TODO(you) 9: free its blocks.
                seq.table.release()
                seq.status = SeqStatus.FINISHED
                finished.append(seq)
        self.running = [s for s in self.running if s.status is SeqStatus.RUNNING]
        return finished
