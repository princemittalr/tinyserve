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

    Slots come from block lists (M9): a sequence's slots are its blocks' slot ranges laid
    end to end, ``block * block_size + [0, 1, ..., block_size - 1]`` per block, cut at
    its length. That is one small tensor op per sequence instead of one ``table.slot(p)``
    call per token (M7 measured 13-15 ms per step for ~1,100-token contexts), and all
    slots reach the device in one copy.
    """
    if not items:
        raise ValueError("a batch needs at least one sequence")
    tokens: list[int] = []
    positions: list[int] = []
    every: list[torch.Tensor] = []  # per sequence, on the host: the slots of all its tokens
    new_rows: list[torch.Tensor] = []  # per sequence: which of those are the new tokens
    query_lens: list[int] = []
    totals: list[int] = []
    blocks: list[list[int]] = []
    offsets = torch.arange(items[0][1].block_size)
    start = 0  # where this sequence's slots begin in the concatenation of all of them
    for new_tokens, table in items:
        n, total = len(new_tokens), table.num_tokens
        if not 0 < n <= total:
            raise ValueError(f"{n} new tokens but the table holds {total} positions")
        ids = torch.tensor(table.blocks, dtype=torch.int64)
        every.append((ids[:, None] * table.block_size + offsets).flatten()[:total])
        new_rows.append(torch.arange(start + total - n, start + total))
        tokens.extend(new_tokens)
        positions.extend(range(total - n, total))
        query_lens.append(n)
        totals.append(total)
        blocks.append(table.blocks)
        start += total
    all_slots = torch.cat(every)
    new_slots = all_slots[torch.cat(new_rows)]
    on_device = all_slots.to(device)
    width = max(len(b) for b in blocks)
    return Batch(
        token_ids=torch.tensor(tokens, device=device),
        positions=torch.tensor(positions, device=device),
        new_slots=new_slots.to(device),
        seq_slots=list(torch.split(on_device, totals)),
        query_lens=query_lens,
        block_tables=torch.tensor(
            [b + [0] * (width - len(b)) for b in blocks], dtype=torch.int32, device=device
        ),
        context_lens=torch.tensor(
            [t.num_tokens for _, t in items], dtype=torch.int32, device=device
        ),
    )


@dataclass(frozen=True)
class DecodeInputs:
    """Fixed-size inputs of a decode step: the input of ``Qwen3.forward_decode`` (M8).

    A CUDA graph records kernels that read fixed addresses with fixed shapes, so its inputs
    live in buffers allocated once and overwritten every step. Row r is one sequence's one
    new token; rows past the real sequences are padding (``fill_decode_inputs``).

    Attributes:
        token_ids: ``[N]`` int64, each sequence's new token.
        positions: ``[N]`` int64, its position (= tokens before it).
        slots: ``[N]`` int64, the cache slot its k/v are written to.
        block_tables: ``[N, W]`` int32, the sequence's blocks in order, then padding.
        context_lens: ``[N]`` int32, tokens of the sequence so far, the new one included.
    """

    token_ids: torch.Tensor
    positions: torch.Tensor
    slots: torch.Tensor
    block_tables: torch.Tensor
    context_lens: torch.Tensor

    @classmethod
    def empty(cls, num_rows: int, width: int, device: torch.device | str) -> "DecodeInputs":
        def zeros(*shape: int, dtype: torch.dtype = torch.int64) -> torch.Tensor:
            return torch.zeros(shape, dtype=dtype, device=device)

        return cls(
            zeros(num_rows),
            zeros(num_rows),
            zeros(num_rows),
            zeros(num_rows, width, dtype=torch.int32),
            zeros(num_rows, dtype=torch.int32),
        )

    @property
    def num_rows(self) -> int:
        return self.token_ids.shape[0]

    def head(self, n: int) -> "DecodeInputs":
        """The first ``n`` rows, as views: same memory, so a fill shows through."""
        return DecodeInputs(
            self.token_ids[:n],
            self.positions[:n],
            self.slots[:n],
            self.block_tables[:n],
            self.context_lens[:n],
        )


def fill_decode_inputs(
    dst: DecodeInputs, items: Sequence[tuple[int, BlockTable]], pad_block: int
) -> None:
    """Write B decode sequences into the first B rows of ``dst``; the rest become padding.

    ``items`` are ``(new_token, table)`` pairs, the token already appended to the table.
    Reads only each table's block list and length: no per-token slot list, unlike
    ``build_batch``. A padding row is a one-token sequence on ``pad_block`` (the KV cache's
    padding block): token 0 at position 0, written to and read from that block's first slot.
    The values are built on the host and copied into ``dst`` in place.

    Block tables are written only up to this step's longest table (padded with
    ``pad_block``); later columns keep stale values from earlier steps. The kernel reads
    only the first ``ceil(context_len / block_size)`` entries of a row, so it never sees
    them, and the copy stays as small as the step instead of the buffer's full width.
    """
    n, width = dst.block_tables.shape
    if not 0 < len(items) <= n:
        raise ValueError(f"{len(items)} sequences do not fit {n} rows")
    ints: list[list[int]] = []  # per row: token, position, slot, context length
    for token, table in items:
        total = table.num_tokens
        if total < 1:
            raise ValueError("the new token must already be appended to the table")
        if len(table.blocks) > width:
            raise ValueError(f"a sequence holds {len(table.blocks)} blocks, rows hold {width}")
        ints.append([token, total - 1, table.slot(total - 1), total])
    used = max(len(table.blocks) for _, table in items)
    blocks = [table.blocks + [pad_block] * (used - len(table.blocks)) for _, table in items]
    block_size = items[0][1].block_size
    for _ in range(n - len(items)):
        ints.append([0, 0, pad_block * block_size, 1])
        blocks.append([pad_block] * used)
    host = torch.tensor(ints, dtype=torch.int64)
    dst.token_ids.copy_(host[:, 0])
    dst.positions.copy_(host[:, 1])
    dst.slots.copy_(host[:, 2])
    dst.context_lens.copy_(host[:, 3])
    dst.block_tables[:, :used].copy_(torch.tensor(blocks, dtype=torch.int32))
