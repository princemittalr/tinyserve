"""The profiling helpers behind scripts/profile_step.py (M4) and scripts/profile_engine.py
(M7), on a tiny random Qwen3."""

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

import tinyserve.model
from tests.test_model import CFG, tiny_model
from tinyserve.attention import paged_gqa_attention
from tinyserve.batch import build_batch
from tinyserve.block_allocator import BlockAllocator
from tinyserve.engine import Engine, StepRecord
from tinyserve.kv_cache import KVCache
from tinyserve.profiling import (
    TTFT_SEGMENTS,
    attention_replaced,
    decode_tables,
    fill_to_decode,
    gpu_time_ms,
    prefill_items,
    profile_summary,
    summarize_steps,
    time_ms,
    ttft_segments,
)

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


def test_prefill_items_put_new_tokens_after_the_cached_prefix() -> None:
    alloc = BlockAllocator(32)
    items = prefill_items(alloc, 4, num_seqs=2, cached_len=6, new_len=3)
    assert [t.num_tokens for _, t in items] == [9, 9]
    assert alloc.num_used == 2 * 3  # ceil(9 / 4) blocks each
    batch = build_batch(items)
    assert batch.query_lens == [3, 3]
    assert batch.positions.tolist() == [6, 7, 8, 6, 7, 8]


def test_fill_to_decode_leaves_only_decode_steps() -> None:
    cache = KVCache(CFG, 64, 4, dtype=torch.float32)
    # A 10-token step budget fits two 5-token prompts, so the third prefills a step later.
    e = Engine(tiny_model(), cache, max_num_seqs=4, max_batch_tokens=10, trace_steps=True)
    ids = fill_to_decode(e, [[1, 2, 3, 4, 5]] * 3, max_new_tokens=6)
    assert all(e.seqs[i].output for i in ids)
    assert e.step_records is not None and len(e.step_records) >= 2
    e.step()
    assert e.step_records[-1].num_prefill_seqs == 0
    assert e.step_records[-1].num_seqs == 3


def record(forward_ms: float, num_seqs: int) -> StepRecord:
    return StepRecord(num_seqs=num_seqs, num_prefill_seqs=0, num_tokens=num_seqs,
                      context_tokens=10 * num_seqs, schedule_ms=1.0, build_ms=2.0,
                      forward_ms=forward_ms, sample_ms=0.5, update_ms=0.5)  # fmt: skip


def test_summarize_steps_means() -> None:
    s = summarize_steps([record(10.0, 2), record(20.0, 4)])
    assert s["steps"] == 2
    assert s["forward_ms"] == 15.0
    assert s["total_ms"] == 19.0
    assert s["num_seqs"] == 3.0
    assert s["context_tokens"] == 30.0
    assert summarize_steps([]) == {"steps": 0}


def test_gpu_time_ms_is_zero_without_gpu_rows() -> None:
    m = tiny_model()
    cache = KVCache(CFG, 32, 4, dtype=torch.float32)
    batch = build_batch([([0], t) for t in decode_tables(BlockAllocator(32), 4, 2, 6)])
    with profile(activities=[ProfilerActivity.CPU], acc_events=True) as prof:
        m.forward_batch(batch, cache)
    assert gpu_time_ms(prof, lambda name: True) == 0.0


def test_ttft_segments_split_the_whole_ttft_in_order() -> None:
    trace = {"received": 1.0, "submitted": 1.5, "added": 3.0, "first_token": 6.0,
             "first_seen": 6.25, "tag": "cmpl-x"}  # fmt: skip
    segs = ttft_segments(sent=0.5, first_chunk=7.0, trace=trace)
    assert list(segs) == list(TTFT_SEGMENTS)
    assert segs == {"to_handler": 500.0, "parse": 500.0, "to_engine": 1500.0,
                    "engine": 3000.0, "to_loop": 250.0, "to_client": 750.0}  # fmt: skip
    assert sum(segs.values()) == 6500.0
    with pytest.raises(ValueError, match="missing"):
        ttft_segments(0.0, 1.0, {**trace, "added": None})
