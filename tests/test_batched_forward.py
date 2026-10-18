"""Running many sequences in one batch must give each one the same logits it gets alone.

The reference is M1's no-cache ``forward`` on each sequence's full token list, on a tiny
random Qwen3.
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.test_model import CFG, tiny_model
from tinyserve.batch import build_batch
from tinyserve.block_allocator import BlockAllocator
from tinyserve.block_table import BlockTable
from tinyserve.kv_cache import KVCache
from tinyserve.model import Qwen3, greedy_generate

ATOL = RTOL = 1e-5


def make_cache(num_blocks: int = 64, block_size: int = 4) -> tuple[KVCache, BlockAllocator]:
    return KVCache(CFG, num_blocks, block_size, dtype=torch.float32), BlockAllocator(num_blocks)


def reference_last_logits(m: Qwen3, ids: list[int]) -> torch.Tensor:
    return m.forward(torch.tensor(ids))[-1]


# --- build_batch ------------------------------------------------------------------------


def test_build_batch_flattens_positions_and_slots() -> None:
    alloc = BlockAllocator(16)
    a, b = BlockTable(alloc, 4), BlockTable(alloc, 4)
    a.append(3)  # A prefills 3 tokens: positions 0..2
    b.append(5)  # B already has 5 tokens cached...
    b.append(1)  # ...and decodes its 6th, position 5
    batch = build_batch([([7, 8, 9], a), ([4], b)])
    assert batch.token_ids.tolist() == [7, 8, 9, 4]
    assert batch.positions.tolist() == [0, 1, 2, 5]
    assert batch.new_slots.tolist() == [a.slot(0), a.slot(1), a.slot(2), b.slot(5)]
    assert batch.seq_slots[1].tolist() == [b.slot(p) for p in range(6)]
    assert batch.query_lens == [3, 1]
    assert batch.num_seqs == 2
    assert batch.last_rows.tolist() == [2, 3]


def test_build_batch_does_not_allocate() -> None:
    alloc = BlockAllocator(8)
    t = BlockTable(alloc, 4)
    t.append(5)
    free = alloc.num_free
    build_batch([([1, 2, 3, 4, 5], t)])
    assert alloc.num_free == free
    assert t.num_tokens == 5


def test_build_batch_rejects_bad_input() -> None:
    alloc = BlockAllocator(8)
    t = BlockTable(alloc, 4)
    t.append(2)
    with pytest.raises(ValueError):
        build_batch([])
    with pytest.raises(ValueError):
        build_batch([([], t)])  # no new tokens
    with pytest.raises(ValueError):
        build_batch([([1, 2, 3], t)])  # more new tokens than the table holds


# --- forward_batch ----------------------------------------------------------------------


def test_one_sequence_matches_forward_paged() -> None:
    m = tiny_model()
    cache, alloc = make_cache()
    ids = [3, 8, 11, 2, 7, 7, 1]
    t = BlockTable(alloc, cache.block_size)
    slots = torch.tensor(t.append(len(ids)))
    got = m.forward_batch(build_batch([(ids, t)]), cache)
    other_cache, _ = make_cache()
    expected = m.forward_paged(torch.tensor(ids), other_cache, slots, slots)[-1:]
    assert got.shape == (1, CFG.vocab_size)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, expected, atol=ATOL, rtol=RTOL)


def test_mixed_prefill_and_decode_batch_matches_each_sequence_alone() -> None:
    m = tiny_model()
    cache, alloc = make_cache(block_size=2)
    hist = {"a": [5, 9, 2, 7, 1], "b": [3, 8], "c": [1, 4, 13, 6, 2, 2, 9]}
    tables = {name: BlockTable(alloc, cache.block_size) for name in hist}

    # Step 1: prefill a and b together.
    for name in "ab":
        tables[name].append(len(hist[name]))
    got = m.forward_batch(build_batch([(hist[n], tables[n]) for n in "ab"]), cache)
    for row, name in enumerate("ab"):
        torch.testing.assert_close(got[row], reference_last_logits(m, hist[name]))

    # Step 2: a and b decode one token each while c prefills, all in one batch.
    step = {"a": [11], "b": [6], "c": hist["c"]}
    for name in "abc":
        tables[name].append(len(step[name]))
    got = m.forward_batch(build_batch([(step[n], tables[n]) for n in "abc"]), cache)
    for row, name in enumerate("abc"):
        full = hist[name] + step[name] if name != "c" else hist[name]
        torch.testing.assert_close(got[row], reference_last_logits(m, full))


def test_batch_order_does_not_matter() -> None:
    m = tiny_model()
    prompts = [[5, 9, 2], [3, 8, 11, 2, 7], [1]]
    rows = []
    for order in ([0, 1, 2], [2, 0, 1]):
        cache, alloc = make_cache()
        tables = [BlockTable(alloc, cache.block_size) for _ in prompts]
        for i in order:
            tables[i].append(len(prompts[i]))
        logits = m.forward_batch(build_batch([(prompts[i], tables[i]) for i in order]), cache)
        rows.append({i: logits[r] for r, i in enumerate(order)})
    for i in range(len(prompts)):
        torch.testing.assert_close(rows[0][i], rows[1][i], atol=ATOL, rtol=RTOL)


def test_lockstep_batched_greedy_matches_m1_greedy() -> None:
    # Three sequences decode together for 8 steps; each must produce M1's tokens.
    m = tiny_model()
    cache, alloc = make_cache(block_size=4)
    prompts = [[1], [5, 9, 2], [3, 8, 11, 2, 7, 7, 1, 4, 9, 30, 12, 5]]
    tables = [BlockTable(alloc, cache.block_size) for _ in prompts]
    pending = [list(p) for p in prompts]
    outs: list[list[int]] = [[] for _ in prompts]
    for _ in range(8):
        for t, new in zip(tables, pending, strict=True):
            t.append(len(new))
        logits = m.forward_batch(build_batch(list(zip(pending, tables, strict=True))), cache)
        nxt = logits.argmax(-1).tolist()
        for out, tok in zip(outs, nxt, strict=True):
            out.append(tok)
        pending = [[tok] for tok in nxt]
    for prompt, out in zip(prompts, outs, strict=True):
        assert out == greedy_generate(m, prompt, max_new_tokens=8)


@settings(max_examples=25, deadline=None)
@given(
    lens=st.lists(st.integers(1, 9), min_size=1, max_size=4),
    block_size=st.sampled_from([1, 3, 4, 16]),
    split=st.integers(1, 9),
)
def test_chunked_prefill_in_a_batch_matches_reference(
    lens: list[int], block_size: int, split: int
) -> None:
    # Each prompt is fed in two chunks: the first `split` tokens, then the rest. The
    # second step's logits must equal the no-cache forward on the whole prompt.
    m = tiny_model()
    cache, alloc = make_cache(num_blocks=64, block_size=block_size)
    g = torch.Generator().manual_seed(sum(lens) * 31 + split)
    prompts = [torch.randint(0, CFG.vocab_size, (n + 1,), generator=g).tolist() for n in lens]
    tables = [BlockTable(alloc, block_size) for _ in prompts]
    chunks = [(p[: min(split, len(p) - 1)], p[min(split, len(p) - 1) :]) for p in prompts]
    for step in range(2):
        for t, c in zip(tables, chunks, strict=True):
            t.append(len(c[step]))
        batch = build_batch([(c[step], t) for c, t in zip(chunks, tables, strict=True)])
        logits = m.forward_batch(batch, cache)
    for row, p in enumerate(prompts):
        torch.testing.assert_close(logits[row], reference_last_logits(m, p))


def test_build_batch_block_tables_and_lengths_for_the_kernel() -> None:
    alloc = BlockAllocator(16)
    a, b = BlockTable(alloc, 4), BlockTable(alloc, 4)
    a.append(9)  # 3 blocks
    b.append(2)  # 1 block
    b.append(1)
    batch = build_batch([([7] * 9, a), ([4], b)])
    assert batch.block_tables.dtype == batch.context_lens.dtype == torch.int32
    assert batch.block_tables.tolist() == [a.blocks, [*b.blocks, 0, 0]]  # padded with 0
    assert batch.context_lens.tolist() == [9, 3]
