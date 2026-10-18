"""Batched prefill attention (M8 step 4) must equal the per-sequence path it replaces.

The reference is ``paged_gqa_attention`` run one sequence at a time, the M2-M7 prefill
path. Then the model and the engine with ``prefill_attention="batched"`` must give the
same logits and the same greedy tokens as before. Tiny random Qwen3.
"""

import pytest
import torch

from tests.test_engine import PROMPTS, expected
from tests.test_model import CFG, tiny_model
from tinyserve.attention import batched_prefill_attention, paged_gqa_attention, prefill_plan
from tinyserve.batch import build_batch
from tinyserve.block_allocator import BlockAllocator
from tinyserve.block_table import BlockTable
from tinyserve.engine import Engine
from tinyserve.kv_cache import KVCache
from tinyserve.model import Qwen3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HQ, HKV, D = 4, 2, 8


def scattered_slots(lens: list[int], num_slots: int, seed: int = 0) -> list[torch.Tensor]:
    """Distinct random cache slots for each sequence (blocks don't matter here)."""
    perm = torch.randperm(num_slots, generator=torch.Generator().manual_seed(seed))
    out, at = [], 0
    for n in lens:
        out.append(perm[at : at + n].to(DEVICE))
        at += n
    return out


def test_plan_pads_queries_and_keys() -> None:
    # Sequence 0: 3 new of 5 tokens, rows 0..2. Sequence 1: 1 new of 2 tokens, row 3.
    slots = [torch.tensor([10, 11, 12, 13, 14]), torch.tensor([20, 21])]
    p = prefill_plan([0, 3], [3, 1], slots)
    assert p.rows.tolist() == [[0, 1, 2], [3, 3, 3]]  # padding repeats the last row
    assert p.real.tolist() == [[True, True, True], [True, False, False]]
    assert p.slots.tolist() == [[10, 11, 12, 13, 14], [20, 21, 0, 0, 0]]
    assert p.mask.shape == (2, 1, 3, 5)
    # Sequence 0's queries are positions 2, 3, 4: query i sees keys 0..2+i.
    assert p.mask[0, 0].tolist() == [[True] * 3 + [False] * 2, [True] * 4 + [False], [True] * 5]
    # Sequence 1's query is position 1: keys 0, 1, never the padding keys; its padding
    # queries repeat that query, mask included.
    assert p.mask[1, 0].tolist() == [[True, True, False, False, False]] * 3


@pytest.mark.parametrize(
    ("cached", "new"),
    [
        ([0, 0, 0], [5, 1, 3]),  # plain prompts, one of a single token
        ([7, 0, 12], [2, 6, 1]),  # after a cached prefix (prefix-cache hits), and a decode
        ([0], [9]),  # one sequence
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_batched_equals_per_sequence(cached: list[int], new: list[int], dtype: torch.dtype) -> None:
    if dtype == torch.float16 and DEVICE == "cpu":
        pytest.skip("fp16 attention on CPU is not a configuration TinyServe runs")
    g = torch.Generator().manual_seed(1)
    lens = [c + n for c, n in zip(cached, new, strict=True)]
    k_cache = torch.randn(64, HKV, D, generator=g).to(DEVICE, dtype)
    v_cache = torch.randn(64, HKV, D, generator=g).to(DEVICE, dtype)
    q = torch.randn(sum(new), HQ, D, generator=g).to(DEVICE, dtype)
    slots = scattered_slots(lens, 64)
    starts = [sum(new[:r]) for r in range(len(new))]
    plan = prefill_plan(starts, new, slots)
    res = batched_prefill_attention(q, k_cache, v_cache, plan, D**-0.5)
    got = torch.empty_like(q)
    got[plan.rows[plan.real]] = res[plan.real]
    want = torch.cat(
        [
            paged_gqa_attention(q[s : s + n], k_cache, v_cache, sl, D**-0.5)
            for s, n, sl in zip(starts, new, slots, strict=True)
        ]
    )
    tol = 1e-5 if dtype == torch.float32 else 1e-3
    torch.testing.assert_close(got, want, atol=tol, rtol=tol)


def models(attention: str = "torch") -> tuple[Qwen3, Qwen3]:
    w = {n: t.to(DEVICE) for n, t in tiny_model().w.items()}
    if attention == "triton":
        pytest.importorskip("triton", reason="Triton is not installed (no Windows wheels)")
    return (
        Qwen3(CFG, w, decode_attention=attention),  # type: ignore[arg-type]
        Qwen3(CFG, w, decode_attention=attention, prefill_attention="batched"),  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("attention", ["torch", "triton"])
@pytest.mark.parametrize(
    ("cached", "new"),
    [([0, 0, 0], [3, 5, 2]), ([0, 6, 0, 9], [4, 1, 7, 1]), ([5, 3], [1, 1])],
)
def test_forward_batch_matches_per_sequence_path(
    attention: str, cached: list[int], new: list[int]
) -> None:
    per_seq, batched = models(attention)
    cache = KVCache(CFG, 64, 4, dtype=torch.float32, device=DEVICE)
    g = torch.Generator().manual_seed(0)
    cache.k.copy_(torch.randn(cache.k.shape, generator=g))  # stands in for earlier steps
    cache.v.copy_(torch.randn(cache.v.shape, generator=g))
    alloc = BlockAllocator(cache.num_blocks)
    items = []
    for c, n in zip(cached, new, strict=True):
        t = BlockTable(alloc, 4)
        t.append(c + n)
        items.append((list(range(1, n + 1)), t))
    batch = build_batch(items, device=DEVICE)
    a = per_seq.forward_batch(batch, cache)
    b = batched.forward_batch(batch, cache)
    torch.testing.assert_close(b, a, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("num_blocks", "max_num_seqs", "max_batch_tokens"),
    [(64, 8, 2048), (7, 8, 2048), (64, 8, 17)],  # roomy; preemptions; prompts split by budget
)
def test_engine_with_batched_prefill_matches_m1_greedy(
    num_blocks: int, max_num_seqs: int, max_batch_tokens: int
) -> None:
    _, model = models()
    cache = KVCache(CFG, num_blocks, 4, dtype=torch.float32, device=DEVICE)
    e = Engine(model, cache, max_num_seqs=max_num_seqs, max_batch_tokens=max_batch_tokens)
    assert e.generate(PROMPTS, 10) == [expected(p, 10) for p in PROMPTS]
    if num_blocks == 7:
        assert e.scheduler.num_preemptions > 0


def test_engine_with_batched_prefill_and_prefix_cache_matches_m1_greedy() -> None:
    from tests.test_prefix_scheduler import QUESTIONS, SYSTEM

    _, model = models()
    cache = KVCache(CFG, 64, 4, dtype=torch.float32, device=DEVICE)
    e = Engine(model, cache, max_num_seqs=2, prefix_cache=True)
    prompts = [SYSTEM + q for q in QUESTIONS] + [SYSTEM]
    assert e.generate(prompts, 6) == [expected(p, 6) for p in prompts]
    assert e.scheduler.num_cached_tokens > 0


def test_unknown_prefill_attention_is_rejected() -> None:
    with pytest.raises(ValueError, match="per_seq' or 'batched"):
        Qwen3(CFG, tiny_model().w, prefill_attention="flash")  # type: ignore[arg-type]


@pytest.mark.parametrize("num_blocks", [64, 7])
def test_engine_with_graphs_and_batched_prefill_matches_m1_greedy(num_blocks: int) -> None:
    # The configuration the M8 benchmark runs: kernel, CUDA graphs (captured on a GPU,
    # eager on CPU) and batched prefill.
    from tinyserve.cuda_graphs import DecodeGraphs

    _, model = models("triton")
    cache = KVCache(CFG, num_blocks, 4, dtype=torch.float32, device=DEVICE)
    graphs = DecodeGraphs(model, cache, max_batch=8)
    e = Engine(model, cache, max_num_seqs=8, decode_graphs=graphs)
    assert e.generate(PROMPTS, 10) == [expected(p, 10) for p in PROMPTS]
