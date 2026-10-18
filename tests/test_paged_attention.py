"""Spec for tinyserve.kv_cache and paged_gqa_attention. Do not edit these tests to make them pass.

Every attention test compares against causal_gqa_attention from M1: reading k/v through
a block table must give the same answer as having them in one contiguous tensor.
"""

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from tinyserve.attention import causal_gqa_attention, paged_gqa_attention
from tinyserve.block_allocator import BlockAllocator
from tinyserve.block_table import BlockTable
from tinyserve.config import ModelConfig
from tinyserve.kv_cache import KVCache

HQ, HKV, D = 4, 2, 8
SCALE = 1 / math.sqrt(D)


def tiny_cfg(num_layers: int = 2) -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_layers=num_layers,
        num_heads=HQ,
        num_kv_heads=HKV,
        head_dim=D,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )


def rand(*shape: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def qkv(t: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return rand(t, HQ, D, seed=seed), rand(t, HKV, D, seed=seed + 1), rand(t, HKV, D, seed=seed + 2)


def scattered_cache(
    k: torch.Tensor, v: torch.Tensor, block_size: int, blocks: list[int], num_blocks: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Write k/v into a cache through the given block order. Returns (k_cache, v_cache, slots)."""
    t = k.shape[0]
    slots = torch.tensor([blocks[p // block_size] * block_size + p % block_size for p in range(t)])
    k_cache = rand(num_blocks * block_size, HKV, D, seed=99)  # garbage everywhere else
    v_cache = rand(num_blocks * block_size, HKV, D, seed=98)
    k_cache[slots], v_cache[slots] = k, v
    return k_cache, v_cache, slots


# ---------------------------------------------------------------- KVCache


def test_cache_shape_and_memory() -> None:
    # 5 blocks for the allocator plus the padding block (M8): 6 blocks of 4 slots.
    c = KVCache(tiny_cfg(num_layers=3), num_blocks=5, block_size=4)
    assert c.k.shape == c.v.shape == (3, 24, HKV, D)
    assert (c.num_blocks, c.num_slots, c.pad_block) == (5, 20, 5)
    assert c.k.dtype == torch.float16
    assert c.memory_bytes == 2 * 3 * 24 * HKV * D * 2


def test_write_lands_in_the_right_slots_only() -> None:
    c = KVCache(tiny_cfg(), num_blocks=4, block_size=4, dtype=torch.float32)
    k, v = rand(3, HKV, D, seed=1), rand(3, HKV, D, seed=2)
    slots = torch.tensor([9, 2, 14])
    c.write(1, slots, k, v)
    torch.testing.assert_close(c.k[1, slots], k)
    torch.testing.assert_close(c.v[1, slots], v)
    untouched = [s for s in range(16) if s not in (9, 2, 14)]
    assert c.k[1, untouched].eq(0).all() and c.v[1, untouched].eq(0).all()
    assert c.k[0].eq(0).all(), "wrote to the wrong layer"


def test_write_casts_to_cache_dtype() -> None:
    c = KVCache(tiny_cfg(), num_blocks=2, block_size=4)
    c.write(0, torch.tensor([0]), rand(1, HKV, D), rand(1, HKV, D))
    assert c.k.dtype == torch.float16


# ---------------------------------------------------------------- paged attention


def test_prefill_with_contiguous_slots_matches_reference() -> None:
    q, k, v = qkv(7)
    out = paged_gqa_attention(q, k, v, torch.arange(7), SCALE)
    torch.testing.assert_close(out, causal_gqa_attention(q, k, v, SCALE))


def test_prefill_with_scattered_blocks_matches_reference() -> None:
    q, k, v = qkv(10, seed=3)
    k_cache, v_cache, slots = scattered_cache(k, v, block_size=4, blocks=[5, 2, 7], num_blocks=8)
    out = paged_gqa_attention(q, k_cache, v_cache, slots, SCALE)
    torch.testing.assert_close(out, causal_gqa_attention(q, k, v, SCALE))


def test_decode_one_token_matches_last_row() -> None:
    q, k, v = qkv(9, seed=5)
    k_cache, v_cache, slots = scattered_cache(k, v, block_size=4, blocks=[3, 0, 6], num_blocks=8)
    out = paged_gqa_attention(q[-1:], k_cache, v_cache, slots, SCALE)
    assert out.shape == (1, HQ, D)
    torch.testing.assert_close(out, causal_gqa_attention(q, k, v, SCALE)[-1:])


def test_decode_sees_every_earlier_token() -> None:
    # q = 0 makes all scores equal, so the newest token averages v over ALL positions.
    t = 6
    v = rand(t, HKV, D, seed=7)
    out = paged_gqa_attention(torch.zeros(1, HQ, D), rand(t, HKV, D), v, torch.arange(t), 1.0)
    expected = v.mean(dim=0).repeat_interleave(HQ // HKV, dim=0)
    torch.testing.assert_close(out[0], expected)


def test_chunked_prefill_matches_reference() -> None:
    # Feed the prompt 3 tokens at a time; chunk c may see every position up to its own end.
    q, k, v = qkv(8, seed=11)
    ref = causal_gqa_attention(q, k, v, SCALE)
    k_cache, v_cache, slots = scattered_cache(k, v, block_size=2, blocks=[4, 1, 6, 0], num_blocks=8)
    for lo in range(0, 8, 3):
        hi = min(lo + 3, 8)
        out = paged_gqa_attention(q[lo:hi], k_cache, v_cache, slots[:hi], SCALE)
        torch.testing.assert_close(out, ref[lo:hi])


def test_keeps_dtype() -> None:
    q, k, v = (x.half() for x in qkv(5))
    out = paged_gqa_attention(q, k, v, torch.arange(5), SCALE)
    assert out.dtype == torch.float16


@settings(max_examples=50, deadline=None)
@given(
    t=st.integers(min_value=1, max_value=24),
    n=st.integers(min_value=1, max_value=24),
    block_size=st.sampled_from([1, 2, 4, 16]),
    seed=st.integers(min_value=0, max_value=2**16),
    data=st.data(),
)
def test_matches_reference_for_any_block_order(
    t: int, n: int, block_size: int, seed: int, data: st.DataObject
) -> None:
    n = min(n, t)
    num_blocks = -(-t // block_size) + 3
    blocks = data.draw(st.permutations(list(range(num_blocks))))
    q, k, v = qkv(t, seed=seed)
    k_cache, v_cache, slots = scattered_cache(k, v, block_size, blocks, num_blocks)
    out = paged_gqa_attention(q[t - n :], k_cache, v_cache, slots, SCALE)
    torch.testing.assert_close(out, causal_gqa_attention(q, k, v, SCALE)[t - n :])


# ---------------------------------------------------------------- all pieces together


@pytest.mark.parametrize("block_size", [1, 4, 16])
def test_prefill_then_decode_through_block_tables(block_size: int) -> None:
    """Two sequences share one cache: prefill each, then decode them interleaved."""
    cache = KVCache(
        tiny_cfg(num_layers=1), num_blocks=32, block_size=block_size, dtype=torch.float32
    )
    alloc = BlockAllocator(32)
    lengths = {"a": 6, "b": 5}
    full = {name: qkv(t + 4, seed=i * 10) for i, (name, t) in enumerate(lengths.items())}
    tables = {name: BlockTable(alloc, block_size) for name in lengths}

    def step(name: str, lo: int, hi: int) -> None:
        q, k, v = full[name]
        table = tables[name]
        new_slots = torch.tensor(table.append(hi - lo))
        cache.write(0, new_slots, k[lo:hi], v[lo:hi])
        all_slots = torch.tensor([table.slot(p) for p in range(hi)])
        out = paged_gqa_attention(q[lo:hi], cache.k[0], cache.v[0], all_slots, SCALE)
        ref = causal_gqa_attention(q[:hi], k[:hi], v[:hi], SCALE)[lo:hi]
        torch.testing.assert_close(out, ref)

    for name, t in lengths.items():
        step(name, 0, t)  # prefill
    for i in range(4):  # 4 decode steps, alternating sequences
        for name, t in lengths.items():
            step(name, t + i, t + i + 1)
    assert alloc.num_used == sum(len(tb.blocks) for tb in tables.values())
