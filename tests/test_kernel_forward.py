"""The model with the Triton decode kernel must behave exactly like the PyTorch path.

``Qwen3(..., decode_attention="triton")`` sends every one-token sequence of a batch through
the M4 kernel. On CPU the kernel runs in Triton's interpreter (see conftest.py), so these
tests check the wiring in CI; on a GPU they run the compiled kernel. Tiny random Qwen3.
"""

import pytest
import torch

pytest.importorskip("triton", reason="Triton is not installed (no Windows wheels)")

from tests.test_engine import PROMPTS, expected
from tests.test_model import CFG, tiny_model
from tinyserve.batch import build_batch
from tinyserve.block_allocator import BlockAllocator
from tinyserve.block_table import BlockTable
from tinyserve.engine import Engine
from tinyserve.kv_cache import KVCache
from tinyserve.model import Qwen3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def models() -> tuple[Qwen3, Qwen3]:
    w = {n: t.to(DEVICE) for n, t in tiny_model().w.items()}
    return Qwen3(CFG, w), Qwen3(CFG, w, decode_attention="triton")


def tables(alloc: BlockAllocator, cached: list[int], new: list[int]) -> list[BlockTable]:
    out = []
    for c, n in zip(cached, new, strict=True):
        t = BlockTable(alloc, 4)
        t.append(c)
        t.append(n)
        out.append(t)
    return out


@pytest.mark.parametrize(
    ("cached", "new"),
    [
        ([5, 0, 12, 3], [1, 1, 1, 1]),  # all decode (one 1-token prompt among them)
        ([0, 6, 0, 9], [4, 1, 7, 1]),  # mixed: two prefills, two decodes
        ([0, 0], [3, 5]),  # all prefill: the kernel is never called
    ],
)
def test_forward_batch_matches_torch_path(cached: list[int], new: list[int]) -> None:
    torch_model, triton_model = models()
    cache = KVCache(CFG, 64, 4, dtype=torch.float32, device=DEVICE)
    alloc = BlockAllocator(cache.num_blocks)
    g = torch.Generator().manual_seed(0)
    cache.k.copy_(torch.randn(cache.k.shape, generator=g))  # stands in for earlier steps
    cache.v.copy_(torch.randn(cache.v.shape, generator=g))
    ts = tables(alloc, cached, new)
    items = [(list(range(1, n + 1)), t) for n, t in zip(new, ts, strict=True)]
    batch = build_batch(items, device=DEVICE)
    a = torch_model.forward_batch(batch, cache)
    b = triton_model.forward_batch(batch, cache)
    torch.testing.assert_close(b, a, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("num_blocks", "max_num_seqs"),
    [(64, 8), (64, 1), (7, 8)],  # roomy; one at a time; tight cache (preemptions)
)
def test_engine_with_kernel_matches_m1_greedy(num_blocks: int, max_num_seqs: int) -> None:
    _, triton_model = models()
    cache = KVCache(CFG, num_blocks, 4, dtype=torch.float32, device=DEVICE)
    e = Engine(triton_model, cache, max_num_seqs=max_num_seqs)
    assert e.generate(PROMPTS, 10) == [expected(p, 10) for p in PROMPTS]
    if num_blocks == 7:
        assert e.scheduler.num_preemptions > 0


@pytest.mark.parametrize("num_blocks", [64, 9])
def test_engine_with_kernel_and_prefix_cache_matches_m1_greedy(num_blocks: int) -> None:
    # A cache hit can leave a one-token "prefill" after the cached prefix: it goes through
    # the decode kernel, reading the shared blocks.
    from tests.test_prefix_scheduler import QUESTIONS, SYSTEM

    _, triton_model = models()
    cache = KVCache(CFG, num_blocks, 4, dtype=torch.float32, device=DEVICE)
    e = Engine(triton_model, cache, max_num_seqs=2, prefix_cache=True)
    prompts = [SYSTEM + q for q in QUESTIONS] + [SYSTEM]  # SYSTEM: 8 cached + 1 to run
    assert e.generate(prompts, 6) == [expected(p, 6) for p in prompts]
    assert e.scheduler.num_cached_tokens > 0


def test_unknown_decode_attention_is_rejected() -> None:
    with pytest.raises(ValueError, match="torch' or 'triton"):
        Qwen3(CFG, tiny_model().w, decode_attention="flash")  # type: ignore[arg-type]
