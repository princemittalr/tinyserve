"""The profiling helpers behind scripts/profile_step.py (M4 step 1), on a tiny random Qwen3."""

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

import tinyserve.model
from tests.test_model import CFG, tiny_model
from tinyserve.attention import paged_gqa_attention
from tinyserve.batch import build_batch
from tinyserve.block_allocator import BlockAllocator
from tinyserve.kv_cache import KVCache
from tinyserve.profiling import attention_replaced, decode_tables, profile_summary, time_ms

CPU = torch.device("cpu")


def test_decode_tables_hold_context_plus_one_new_token() -> None:
    alloc = BlockAllocator(32)
    tables = decode_tables(alloc, 4, num_seqs=3, context_len=6)
    assert [t.num_tokens for t in tables] == [7, 7, 7]
    assert alloc.num_used == 3 * 2  # ceil(7 / 4) blocks each
    batch = build_batch([([0], t) for t in tables])
    assert batch.query_lens == [1, 1, 1]
    assert batch.positions.tolist() == [6, 6, 6]  # every sequence decodes its 7th token


def test_decode_tables_never_share_a_slot() -> None:
    tables = decode_tables(BlockAllocator(32), 4, num_seqs=4, context_len=6)
    slots = [t.slot(p) for t in tables for p in range(t.num_tokens)]
    assert len(slots) == len(set(slots))


def test_attention_replaced_changes_the_output_and_restores_the_function() -> None:
    m = tiny_model()
    cache = KVCache(CFG, 32, 4, dtype=torch.float32)
    tables = decode_tables(BlockAllocator(cache.num_blocks), 4, 2, 6)
    batch = build_batch([([0], t) for t in tables])
    real = paged_gqa_attention
    before = m.forward_batch(batch, cache)
    with attention_replaced():
        assert vars(tinyserve.model)["paged_gqa_attention"] is not real
        replaced = m.forward_batch(batch, cache)
    assert vars(tinyserve.model)["paged_gqa_attention"] is real
    assert replaced.shape == before.shape
    assert not torch.allclose(replaced, before)
    torch.testing.assert_close(m.forward_batch(batch, cache), before)


def test_attention_replaced_restores_after_an_error() -> None:
    real = paged_gqa_attention
    with pytest.raises(ZeroDivisionError), attention_replaced():
        _ = 1 / 0
    assert vars(tinyserve.model)["paged_gqa_attention"] is real


def test_time_ms_runs_warmup_then_iters_and_returns_the_median() -> None:
    calls = []
    t = time_ms(lambda: calls.append(1), CPU, warmup=2, iters=3)
    assert len(calls) == 5
    assert t >= 0


def test_profile_summary_on_cpu_counts_ops_and_no_kernels() -> None:
    m = tiny_model()
    cache = KVCache(CFG, 32, 4, dtype=torch.float32)

    def aten_calls(num_seqs: int) -> float:
        tables = decode_tables(BlockAllocator(cache.num_blocks), 4, num_seqs, 6)
        batch = build_batch([([0], t) for t in tables])
        with profile(activities=[ProfilerActivity.CPU], acc_events=True) as prof:
            m.forward_batch(batch, cache)
        s = profile_summary(prof, wall_ms=1.0)
        assert s["kernel_launches"] == 0
        assert s["gpu_kernel_ms"] == 0
        return s["aten_calls"]

    one, four = aten_calls(1), aten_calls(4)
    assert one > 0
    # Attention runs once per sequence per layer, so op calls grow with the batch.
    assert four > one
