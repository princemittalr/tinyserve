"""The fixed-shape decode step (M8): ``fill_decode_inputs`` and ``Qwen3.forward_decode``.

A CUDA graph will replay ``forward_decode`` on buffers padded to a fixed number of rows.
These tests pin down what must hold before any graph exists: the buffers say what the
block tables say, padding rows change nothing a real sequence can see, and the step gives
the same logits and the same greedy tokens as the M4 path. Tiny random Qwen3; on CPU the
kernel runs in Triton's interpreter.
"""

import pytest
import torch

pytest.importorskip("triton", reason="Triton is not installed (no Windows wheels)")

from tests.test_engine import PROMPTS, expected
from tests.test_kernel_forward import models, tables
from tests.test_model import CFG
from tinyserve.batch import DecodeInputs, build_batch, fill_decode_inputs
from tinyserve.block_allocator import BlockAllocator
from tinyserve.block_table import BlockTable
from tinyserve.kv_cache import KVCache

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def random_cache(num_blocks: int = 64, block_size: int = 4) -> KVCache:
    cache = KVCache(CFG, num_blocks, block_size, dtype=torch.float32, device=DEVICE)
    g = torch.Generator().manual_seed(0)
    cache.k.copy_(torch.randn(cache.k.shape, generator=g))  # stands in for earlier steps
    cache.v.copy_(torch.randn(cache.v.shape, generator=g))
    return cache


# ---------------------------------------------------------------- fill_decode_inputs


def test_fill_writes_each_sequence_then_padding() -> None:
    alloc = BlockAllocator(16)
    a, b = tables(alloc, [5, 0], [1, 1])  # a: 6 tokens over 2 blocks; b: 1 token
    inp = DecodeInputs.empty(4, 3, "cpu")
    fill_decode_inputs(inp, [(7, a), (9, b)], pad_block=16)
    assert inp.token_ids.tolist() == [7, 9, 0, 0]
    assert inp.positions.tolist() == [5, 0, 0, 0]
    assert inp.slots.tolist() == [a.slot(5), b.slot(0), 64, 64]  # padding: block 16, slot 0
    assert inp.context_lens.tolist() == [6, 1, 1, 1]
    # Columns up to the longest table (2 blocks) are written; column 3 is never read.
    assert inp.block_tables[:, :2].tolist() == [a.blocks, [*b.blocks, 16], [16] * 2, [16] * 2]
    assert inp.block_tables[:, 2].tolist() == [0] * 4
    assert inp.block_tables.dtype == inp.context_lens.dtype == torch.int32


def test_fill_overwrites_the_previous_step() -> None:
    alloc = BlockAllocator(16)
    a, b, c = tables(alloc, [2, 3, 4], [1, 1, 1])
    inp = DecodeInputs.empty(4, 3, "cpu")
    fill_decode_inputs(inp, [(1, a), (2, b), (3, c)], pad_block=16)
    fill_decode_inputs(inp, [(4, b)], pad_block=16)
    assert inp.token_ids.tolist() == [4, 0, 0, 0]
    assert inp.context_lens.tolist() == [4, 1, 1, 1]  # b: 3 cached + 1 new


def test_head_is_a_view() -> None:
    inp = DecodeInputs.empty(8, 2, "cpu")
    alloc = BlockAllocator(4)
    (a,) = tables(alloc, [0], [1])
    fill_decode_inputs(inp.head(2), [(5, a)], pad_block=4)
    assert inp.head(2).num_rows == 2
    assert inp.token_ids.tolist()[:2] == [5, 0]
    assert inp.slots.tolist()[1] == 16  # the padding row written through the view


def test_fill_rejects_what_does_not_fit() -> None:
    alloc = BlockAllocator(16)
    a, b = tables(alloc, [9, 0], [1, 1])  # a holds 3 blocks of 4
    with pytest.raises(ValueError, match="do not fit"):
        fill_decode_inputs(DecodeInputs.empty(1, 4, "cpu"), [(1, a), (2, b)], pad_block=16)
    with pytest.raises(ValueError, match="do not fit"):
        fill_decode_inputs(DecodeInputs.empty(1, 4, "cpu"), [], pad_block=16)
    with pytest.raises(ValueError, match="3 blocks"):
        fill_decode_inputs(DecodeInputs.empty(2, 2, "cpu"), [(1, a)], pad_block=16)
    with pytest.raises(ValueError, match="already be appended"):
        fill_decode_inputs(DecodeInputs.empty(2, 2, "cpu"), [(1, BlockTable(alloc, 4))], 16)


# ---------------------------------------------------------------- forward_decode


@pytest.mark.parametrize("num_rows", [4, 8])  # no padding; 4 padding rows
def test_forward_decode_matches_forward_batch(num_rows: int) -> None:
    _, model = models()
    cache = random_cache()
    ts = tables(BlockAllocator(cache.num_blocks), [5, 0, 12, 3], [1, 1, 1, 1])
    tokens = [3, 8, 1, 2]
    items = [([t], s) for t, s in zip(tokens, ts, strict=True)]
    want = model.forward_batch(build_batch(items, DEVICE), cache)

    inp = DecodeInputs.empty(num_rows, 8, DEVICE)
    fill_decode_inputs(inp, list(zip(tokens, ts, strict=True)), cache.pad_block)
    before_k = cache.k.clone()
    got = model.forward_decode(inp, cache)
    assert got.shape == (num_rows, CFG.vocab_size)
    torch.testing.assert_close(got[:4], want, atol=1e-5, rtol=1e-5)
    # Only the 4 new tokens' slots and the padding block changed.
    changed = (cache.k != before_k).flatten(2).any(-1).any(0).nonzero().flatten().tolist()
    pad_slots = range(cache.num_slots, cache.num_slots + cache.block_size)
    assert set(changed) - set(pad_slots) <= {s.slot(s.num_tokens - 1) for s in ts}


def test_padding_rows_do_not_change_real_rows() -> None:
    _, model = models()
    ts = tables(BlockAllocator(64), [5, 2], [1, 1])
    out = []
    for n in (2, 8):
        inp = DecodeInputs.empty(n, 8, DEVICE)
        fill_decode_inputs(inp, [(3, ts[0]), (4, ts[1])], 64)
        out.append(model.forward_decode(inp, random_cache())[:2])
    torch.testing.assert_close(out[1], out[0], atol=1e-5, rtol=1e-5)


def test_greedy_through_forward_decode_matches_m1() -> None:
    # Prefill with forward_batch, then every decode step through padded fixed buffers.
    _, model = models()
    cache = KVCache(CFG, 64, 4, dtype=torch.float32, device=DEVICE)
    alloc = BlockAllocator(cache.num_blocks)
    ts = [BlockTable(alloc, 4) for _ in PROMPTS]
    for p, t in zip(PROMPTS, ts, strict=True):
        t.append(len(p))
    logits = model.forward_batch(build_batch(list(zip(PROMPTS, ts, strict=True)), DEVICE), cache)
    out = [[t] for t in logits.argmax(-1).tolist()]
    inp = DecodeInputs.empty(8, 8, DEVICE)
    for _ in range(9):
        for t in ts:
            t.append(1)
        fill_decode_inputs(inp, [(o[-1], t) for o, t in zip(out, ts, strict=True)], 64)
        nxt = model.forward_decode(inp, cache)[: len(ts)].argmax(-1).tolist()
        for o, n in zip(out, nxt, strict=True):
            o.append(n)
    assert out == [expected(p, 10) for p in PROMPTS]


def test_forward_decode_needs_the_kernel() -> None:
    torch_model, _ = models()
    with pytest.raises(ValueError, match="triton"):
        torch_model.forward_decode(DecodeInputs.empty(1, 1, DEVICE), random_cache())


@pytest.mark.gpu
def test_forward_decode_never_waits_for_the_gpu() -> None:
    # A host sync inside the step would make CUDA graph capture fail; catch it here first.
    _, model = models()
    cache = random_cache()
    ts = tables(BlockAllocator(cache.num_blocks), [5, 0, 12], [1, 1, 1])
    inp = DecodeInputs.empty(8, 8, DEVICE)
    fill_decode_inputs(inp, [(1, t) for t in ts], cache.pad_block)
    model.forward_decode(inp, cache)  # compile the kernel first: compiling may sync
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        model.forward_decode(inp, cache)
    finally:
        torch.cuda.set_sync_debug_mode("default")
