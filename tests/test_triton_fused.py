"""The fused norm / RoPE / cache-write kernels (M9) must equal the PyTorch ops they replace.

References: ``layers.rms_norm``, ``layers.apply_rope`` and ``KVCache.write``. Tolerances
are M4's: 1e-5 in float32, 1e-3 in float16. Then the model with ``fused_kernels=True``
must give the same logits and the same greedy tokens as without.
"""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytest.importorskip("triton", reason="Triton is not installed (no Windows wheels)")

from tests.test_engine import PROMPTS, expected
from tests.test_model import CFG, tiny_model
from tinyserve import triton_fused
from tinyserve.batch import DecodeInputs, build_batch, fill_decode_inputs
from tinyserve.block_allocator import BlockAllocator
from tinyserve.block_table import BlockTable
from tinyserve.cuda_graphs import DecodeGraphs
from tinyserve.engine import Engine
from tinyserve.kv_cache import KVCache
from tinyserve.layers import apply_rope, rms_norm, rope_cos_sin
from tinyserve.model import Qwen3
from tinyserve.triton_fused import qk_norm_rope_store, rms_norm_rows

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOL = {torch.float32: 1e-5, torch.float16: 1e-3}
EPS = 1e-6


def randn(*shape: int, dtype: torch.dtype = torch.float32, scale: float = 1.0) -> torch.Tensor:
    return (torch.randn(shape) * scale).to(DEVICE, dtype)


# --- rms_norm_rows ------------------------------------------------------------------------


@pytest.mark.parametrize("n", [16, 24, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rms_norm_rows_equals_rms_norm(n: int, dtype: torch.dtype) -> None:
    torch.manual_seed(n)
    x = randn(7, n, dtype=dtype, scale=300.0)  # 300**2 overflows fp16: must upcast first
    w = randn(n, dtype=dtype)
    got = rms_norm_rows(x, w, EPS)
    assert got.dtype == dtype and torch.isfinite(got).all()
    torch.testing.assert_close(got, rms_norm(x, w, EPS), atol=TOL[dtype], rtol=TOL[dtype])


def test_rms_norm_rows_of_nothing_and_bad_shapes() -> None:
    assert rms_norm_rows(randn(0, 16), randn(16), EPS).shape == (0, 16)
    with pytest.raises(ValueError):
        rms_norm_rows(randn(2, 3, 16), randn(16), EPS)
    with pytest.raises(ValueError):
        rms_norm_rows(randn(2, 16), randn(8), EPS)


# --- qk_norm_rope_store -------------------------------------------------------------------


def qk_case(t: int, hq: int, hkv: int, d: int, dtype: torch.dtype, seed: int = 0) -> dict[str, Any]:
    torch.manual_seed(seed)
    num_slots = 4 * t + 8
    return {
        "q": randn(t, hq, d, dtype=dtype, scale=30.0),
        "k": randn(t, hkv, d, dtype=dtype, scale=30.0),
        "v": randn(t, hkv, d, dtype=dtype),
        "qw": randn(d, dtype=dtype),
        "kw": randn(d, dtype=dtype),
        "pos": torch.randint(0, 40_000, (t,), device=DEVICE),
        "slots": torch.randperm(num_slots, device=DEVICE)[:t],
        "k_cache": torch.full((num_slots, hkv, d), float("nan"), dtype=dtype, device=DEVICE),
        "v_cache": torch.full((num_slots, hkv, d), float("nan"), dtype=dtype, device=DEVICE),
    }


@pytest.mark.parametrize(("hq", "hkv", "d"), [(4, 2, 8), (4, 4, 16), (16, 8, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_qk_norm_rope_store_equals_the_pytorch_ops(
    hq: int, hkv: int, d: int, dtype: torch.dtype
) -> None:
    c = qk_case(9, hq, hkv, d, dtype, seed=d)
    cos, sin = rope_cos_sin(d, 1e6, c["pos"])
    q = qk_norm_rope_store(c["q"], c["k"], c["v"], c["qw"], c["kw"], cos, sin, c["slots"],
                           c["k_cache"], c["v_cache"], EPS)  # fmt: skip
    tol = TOL[dtype]
    want_q = apply_rope(rms_norm(c["q"], c["qw"], EPS), cos, sin)
    want_k = apply_rope(rms_norm(c["k"], c["kw"], EPS), cos, sin)
    torch.testing.assert_close(q, want_q, atol=tol, rtol=tol)
    torch.testing.assert_close(c["k_cache"][c["slots"]], want_k, atol=tol, rtol=tol)
    torch.testing.assert_close(c["v_cache"][c["slots"]], c["v"], atol=0, rtol=0)
    # Every other slot is untouched (still NaN).
    others = torch.ones(c["k_cache"].shape[0], dtype=torch.bool, device=DEVICE)
    others[c["slots"]] = False
    assert c["k_cache"][others].isnan().all() and c["v_cache"][others].isnan().all()


def test_qk_norm_rope_store_rejects_bad_shapes() -> None:
    c = qk_case(3, 4, 2, 8, torch.float32)
    cos, sin = rope_cos_sin(8, 1e6, c["pos"])
    with pytest.raises(ValueError):  # v with the wrong number of heads
        qk_norm_rope_store(c["q"], c["k"], c["q"], c["qw"], c["kw"], cos, sin, c["slots"],
                           c["k_cache"], c["v_cache"], EPS)  # fmt: skip
    with pytest.raises(ValueError):  # tables for the wrong number of tokens
        qk_norm_rope_store(c["q"], c["k"], c["v"], c["qw"], c["kw"], cos[:2], sin[:2],
                           c["slots"], c["k_cache"], c["v_cache"], EPS)  # fmt: skip


# --- in the model -------------------------------------------------------------------------


def counted(calls: list[str]) -> SimpleNamespace:
    """The triton_fused module, but counting calls (to check the model uses it)."""

    def wrap(name: str) -> Callable[..., torch.Tensor]:
        fn = getattr(triton_fused, name)

        def call(*args: Any) -> torch.Tensor:
            calls.append(name)
            return fn(*args)  # type: ignore[no-any-return]

        return call

    return SimpleNamespace(rms_norm_rows=wrap("rms_norm_rows"),
                           qk_norm_rope_store=wrap("qk_norm_rope_store"))  # fmt: skip


def two_models(decode: str = "triton_grouped") -> tuple[Qwen3, Qwen3]:
    w = {n: t.to(DEVICE) for n, t in tiny_model().w.items()}
    plain = Qwen3(CFG, w, decode_attention=decode)  # type: ignore[arg-type]
    return plain, Qwen3(CFG, w, decode_attention=decode, fused_kernels=True)  # type: ignore[arg-type]


@pytest.mark.parametrize("decode", ["torch", "triton_grouped"])
def test_forward_batch_with_fused_kernels_equals_without(decode: str) -> None:
    plain, fused = two_models(decode)
    calls: list[str] = []
    fused.fused = counted(calls)
    logits = []
    for m in (plain, fused):
        cache = KVCache(CFG, 32, 4, dtype=torch.float32, device=DEVICE)
        alloc = BlockAllocator(32)
        tables = [BlockTable(alloc, 4) for _ in PROMPTS]
        for t, p in zip(tables, PROMPTS, strict=True):
            t.append(len(p))
        m.forward_batch(build_batch(list(zip(PROMPTS, tables, strict=True)), DEVICE), cache)
        for t in tables:  # then one decode token each: a mixed path through the kernel
            t.append(1)
        logits.append(m.forward_batch(build_batch([([3], t) for t in tables], DEVICE), cache))
    torch.testing.assert_close(logits[1], logits[0], atol=1e-5, rtol=1e-5)
    # Per forward: 2 hidden norms per layer and the final one, 1 q/k launch per layer.
    per_forward = 3 * CFG.num_layers + 1
    assert len(calls) == 2 * per_forward
    assert calls.count("qk_norm_rope_store") == 2 * CFG.num_layers


def test_forward_decode_with_fused_kernels_equals_without() -> None:
    plain, fused = two_models()
    outs = []
    for m in (plain, fused):
        cache = KVCache(CFG, 32, 4, dtype=torch.float32, device=DEVICE)
        alloc = BlockAllocator(32)
        tables = [BlockTable(alloc, 4) for _ in PROMPTS]
        for t, p in zip(tables, PROMPTS, strict=True):
            t.append(len(p))
        m.forward_batch(build_batch(list(zip(PROMPTS, tables, strict=True)), DEVICE), cache)
        for t in tables:
            t.append(1)
        inputs = DecodeInputs.empty(8, 8, DEVICE)  # 5 sequences + 3 padding rows
        fill_decode_inputs(inputs, [(3, t) for t in tables], cache.pad_block)
        outs.append(m.forward_decode(inputs, cache)[: len(tables)])
    torch.testing.assert_close(outs[1], outs[0], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("num_blocks", [64, 7])
def test_engine_with_every_m9_option_matches_m1_greedy(num_blocks: int) -> None:
    # Fused kernels, the split-K decode kernel in CUDA graphs (eager on CPU), the varlen
    # prefill kernel, chunked prefill and the prefix cache, all at once.
    w = {n: t.to(DEVICE) for n, t in tiny_model().w.items()}
    model = Qwen3(CFG, w, decode_attention="triton_split", prefill_attention="triton",
                  fused_kernels=True)  # fmt: skip
    cache = KVCache(CFG, num_blocks, 4, dtype=torch.float32, device=DEVICE)
    graphs = DecodeGraphs(model, cache, 4)
    e = Engine(model, cache, max_num_seqs=4, max_batch_tokens=6, prefix_cache=True,
               decode_graphs=graphs, chunked_prefill=True)  # fmt: skip
    prompts = [[7, 3, 9, 1, *p] for p in PROMPTS]
    assert e.generate(prompts, 8) == [expected(p, 8) for p in prompts]
    if num_blocks == 7:
        assert e.scheduler.num_preemptions > 0
