"""One request's state as the scheduler moves it between queues.

A sequence's tokens are its prompt followed by what it has generated so far. The KV
cache holds the k/v of the first ``table.num_tokens`` of them; the rest are "pending" and
go into the next forward pass:

    fresh request:          pending = the whole prompt              (prefill)
    running, after a step:  pending = the one token just generated  (decode)
    after preemption:       the table was released, so pending = prompt + output
                            (recompute: the k/v are rebuilt by one prefill)
    prefix cache hit (M5):  the table starts with cached blocks, so pending = only
                            the tokens after the cached prefix
    chunked prefill (M9):   a long prompt is appended a chunk per step, so pending =
                            the rest of the prompt until its last chunk has run
"""

from dataclasses import dataclass, field
from enum import Enum

from tinyserve.block_table import BlockTable


class SeqStatus(Enum):
    WAITING = "waiting"  # queued: new, or preempted and waiting to be recomputed
    RUNNING = "running"  # holds KV-cache blocks and is in every step's batch
    FINISHED = "finished"  # hit EOS or max_new_tokens; its blocks are freed


@dataclass
class Sequence:
    seq_id: int
    prompt: list[int]
    max_new_tokens: int
    table: BlockTable
    eos_ids: frozenset[int] = frozenset()
    output: list[int] = field(default_factory=list)
    status: SeqStatus = SeqStatus.WAITING
    num_preemptions: int = 0
    # Tokens appended to the table by the last schedule(): this step's input is the last
    # num_scheduled tokens. Less than the prompt when a cached prefix was reused (M5).
    num_scheduled: int = 0

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("prompt must not be empty")
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {self.max_new_tokens}")

    @property
    def tokens(self) -> list[int]:
        return self.prompt + self.output

    @property
    def max_cached_tokens(self) -> int:
        """The most tokens this sequence can ever hold in the cache.

        The last generated token is never fed back (the sequence ends on it), hence -1.
        """
        return len(self.prompt) + self.max_new_tokens - 1

    def pending(self) -> list[int]:
        """Tokens whose k/v are not in the cache yet: the next step's input."""
        return self.tokens[self.table.num_tokens :]

    def in_prefill(self) -> bool:
        """After ``schedule()``: True if prompt tokens are left for later steps (chunked
        prefill, M9). Such a sequence's logits this step predict nothing, so it does not
        sample; every other scheduled sequence ran its last pending token and does."""
        return bool(self.pending())

    def is_done(self) -> bool:
        if len(self.output) >= self.max_new_tokens:
            return True
        return bool(self.output) and self.output[-1] in self.eos_ids
