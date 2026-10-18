"""One model step for many sequences at once: the input of ``Qwen3.forward_batch``.

Continuous batching (M3) runs every active sequence's new tokens in a single forward
pass. A sequence in prefill brings its whole prompt (or a chunk of it), a sequence in
decode brings one token. Their tokens are flattened into one ``[T]`` batch with no
padding. Example: sequence A prefills 3 tokens, B decodes 1, C decodes 1:

    flat index:  0  1  2 | 3 | 4
    sequence:    A  A  A | B | C
    position:    0  1  2 | 9 | 4     (each sequence counts its own positions)

The linear layers and the MLP don't care which sequence a row came from, so they run on
all T rows together. Only attention is per sequence: each one reads its own history from
the KV cache through its own block table.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tinyserve.block_table import BlockTable


@dataclass(frozen=True)
class Batch:
    """Flattened new tokens of B sequences, plus where each one's k/v live.

    Attributes:
        token_ids: ``[T]`` new tokens of every sequence, sequence after sequence.
        positions: ``[T]`` each token's position inside its own sequence (for RoPE).
        new_slots: ``[T]`` KV-cache slot each new token's k/v is written to.
        seq_slots: B tensors; ``seq_slots[b]`` is ``[total_b]``, the slots of every token
            of sequence b so far (new ones included), in position order.
        query_lens: B ints; sequence b owns flat rows ``sum(query_lens[:b]) ..`` onward.
        block_tables: ``[B, max_blocks]`` int32, each sequence's block ids in order, padded
            with 0 (never read: the Triton kernel stops at ``context_lens[b]``).
        context_lens: ``[B]`` int32, tokens of each sequence so far, new ones included.
    """

    token_ids: torch.Tensor
    positions: torch.Tensor
    new_slots: torch.Tensor
    seq_slots: list[torch.Tensor]
    query_lens: list[int]
    block_tables: torch.Tensor
    context_lens: torch.Tensor

    @property
    def num_seqs(self) -> int:
        return len(self.query_lens)

    @property
    def last_rows(self) -> torch.Tensor:
        """``[B]`` flat row of each sequence's last new token: the one that predicts next."""
        ends = torch.tensor(self.query_lens, device=self.token_ids.device).cumsum(0)
        return ends - 1


def build_batch(
    items: Sequence[tuple[Sequence[int], BlockTable]], device: torch.device | str = "cpu"
) -> Batch:
    """Build a ``Batch`` from ``(new_tokens, table)`` pairs, one pair per sequence.

    The tokens must already be appended to the table (``table.append(len(new_tokens))``),
    so they are the table's LAST ``len(new_tokens)`` positions. Building a batch only
    reads the tables; it never allocates.
    """
    if not items:
        raise ValueError("a batch needs at least one sequence")
    tokens: list[int] = []
    positions: list[int] = []
    new_slots: list[int] = []
    seq_slots: list[torch.Tensor] = []
    query_lens: list[int] = []
    blocks: list[list[int]] = []
    for new_tokens, table in items:
        n, total = len(new_tokens), table.num_tokens
        if not 0 < n <= total:
            raise ValueError(f"{n} new tokens but the table holds {total} positions")
        every = [table.slot(p) for p in range(total)]
        tokens.extend(new_tokens)
        positions.extend(range(total - n, total))
        new_slots.extend(every[total - n :])
        seq_slots.append(torch.tensor(every, device=device))
        query_lens.append(n)
        blocks.append(table.blocks)
    width = max(len(b) for b in blocks)
    return Batch(
        token_ids=torch.tensor(tokens, device=device),
        positions=torch.tensor(positions, device=device),
        new_slots=torch.tensor(new_slots, device=device),
        seq_slots=seq_slots,
        query_lens=query_lens,
        block_tables=torch.tensor(
            [b + [0] * (width - len(b)) for b in blocks], dtype=torch.int32, device=device
        ),
        context_lens=torch.tensor(
            [t.num_tokens for _, t in items], dtype=torch.int32, device=device
        ),
    )
