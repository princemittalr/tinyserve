"""Spec for tinyserve.layers. Do not edit these tests to make them pass."""

import math

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from tinyserve.layers import apply_rope, rms_norm, rope_cos_sin

# ---------------------------------------------------------------- RMSNorm


def test_rms_norm_hand_computed() -> None:
    # rms([3, 4]) = sqrt((9 + 16) / 2) = sqrt(12.5)
    x = torch.tensor([[3.0, 4.0]])
    y = rms_norm(x, torch.ones(2), eps=0.0)
    r = math.sqrt(12.5)
    torch.testing.assert_close(y, torch.tensor([[3 / r, 4 / r]]))


def test_rms_norm_applies_weight_per_channel() -> None:
    x = torch.tensor([[3.0, 4.0]])
    y = rms_norm(x, torch.tensor([2.0, -1.0]), eps=0.0)
    r = math.sqrt(12.5)
    torch.testing.assert_close(y, torch.tensor([[6 / r, -4 / r]]))


def test_rms_norm_zero_row_is_zero_not_nan() -> None:
    y = rms_norm(torch.zeros(2, 8), torch.ones(8), eps=1e-6)
    assert torch.equal(y, torch.zeros(2, 8))


def test_rms_norm_normalises_last_dim_only() -> None:
    # [T, H, D]: each head is normalised on its own, like Qwen3's q_norm / k_norm.
    x = torch.randn(3, 4, 16, generator=torch.Generator().manual_seed(0))
    x[:, 1] *= 100.0  # one head much larger than the others
    y = rms_norm(x, torch.ones(16), eps=1e-6)
    rms = y.pow(2).mean(-1).sqrt()
    torch.testing.assert_close(rms, torch.ones(3, 4), atol=1e-4, rtol=0)


def test_rms_norm_preserves_shape_and_dtype() -> None:
    x = torch.randn(5, 32).half()
    y = rms_norm(x, torch.ones(32).half(), eps=1e-6)
    assert y.shape == x.shape
    assert y.dtype == torch.float16


def test_rms_norm_fp16_large_values_do_not_overflow() -> None:
    # 300**2 overflows fp16. A correct implementation squares in fp32.
    x = torch.full((1, 64), 300.0, dtype=torch.float16)
    y = rms_norm(x, torch.ones(64, dtype=torch.float16), eps=1e-6)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y.float(), torch.ones(1, 64), atol=1e-3, rtol=0)


@settings(deadline=None, max_examples=50)
@given(
    scale=st.floats(min_value=1e-2, max_value=1e3),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_rms_norm_is_scale_invariant(scale: float, seed: int) -> None:
    x = torch.randn(4, 64, generator=torch.Generator().manual_seed(seed))
    w = torch.ones(64)
    torch.testing.assert_close(rms_norm(scale * x, w, 1e-12), rms_norm(x, w, 1e-12))


# ---------------------------------------------------------------- RoPE


def test_rope_tables_shape_dtype() -> None:
    cos, sin = rope_cos_sin(8, 10000.0, torch.arange(5))
    assert cos.shape == sin.shape == (5, 8)
    assert cos.dtype == sin.dtype == torch.float32


def test_rope_tables_hand_computed() -> None:
    # D = 4, theta = 10000: inv_freq = [10000**0, 10000**-0.5] = [1, 0.01]
    p = 7
    cos, sin = rope_cos_sin(4, 10000.0, torch.tensor([p]))
    angles = [p * 1.0, p * 0.01, p * 1.0, p * 0.01]
    torch.testing.assert_close(cos[0], torch.tensor([math.cos(a) for a in angles]))
    torch.testing.assert_close(sin[0], torch.tensor([math.sin(a) for a in angles]))


def test_rope_tables_accurate_at_long_positions() -> None:
    cos, _ = rope_cos_sin(128, 1e6, torch.tensor([40000]))
    assert abs(cos[0, 0].item() - math.cos(40000.0)) < 1e-3


def test_rope_position_zero_is_identity() -> None:
    x = torch.randn(1, 2, 8)
    cos, sin = rope_cos_sin(8, 1e6, torch.tensor([0]))
    torch.testing.assert_close(apply_rope(x, cos, sin), x)


def test_rope_pairs_i_with_i_plus_half() -> None:
    # Unit vector on element 0 must rotate into element D/2 = 2, not element 1.
    p = 3
    x = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    cos, sin = rope_cos_sin(4, 10000.0, torch.tensor([p]))
    expected = torch.tensor([[[math.cos(p), 0.0, math.sin(p), 0.0]]])
    torch.testing.assert_close(apply_rope(x, cos, sin), expected)


def test_rope_each_token_uses_its_own_position() -> None:
    x = torch.randn(1, 3, 16).expand(2, 3, 16).clone()  # same vector at two tokens
    cos, sin = rope_cos_sin(16, 1e4, torch.tensor([5, 9]))
    y = apply_rope(x, cos, sin)
    one = rope_cos_sin(16, 1e4, torch.tensor([9]))
    torch.testing.assert_close(y[1:], apply_rope(x[1:], *one))
    assert not torch.allclose(y[0], y[1])


def test_rope_keeps_dtype() -> None:
    x = torch.randn(4, 2, 8).half()
    cos, sin = rope_cos_sin(8, 1e6, torch.arange(4))
    assert apply_rope(x, cos, sin).dtype == torch.float16


@settings(deadline=None, max_examples=50)
@given(pos=st.integers(min_value=0, max_value=40959), seed=st.integers(0, 2**31 - 1))
def test_rope_preserves_norm(pos: int, seed: int) -> None:
    x = torch.randn(1, 4, 64, generator=torch.Generator().manual_seed(seed))
    cos, sin = rope_cos_sin(64, 1e6, torch.tensor([pos]))
    torch.testing.assert_close(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1))


@settings(deadline=None, max_examples=50)
@given(
    m=st.integers(0, 4000),
    n=st.integers(0, 4000),
    shift=st.integers(0, 4000),
    seed=st.integers(0, 2**31 - 1),
)
def test_rope_score_depends_on_relative_position(m: int, n: int, shift: int, seed: int) -> None:
    # The reason RoPE exists: <rope(q, m), rope(k, n)> is a function of m - n.
    g = torch.Generator().manual_seed(seed)
    q, k = torch.randn(1, 1, 32, generator=g), torch.randn(1, 1, 32, generator=g)

    def score(a: int, b: int) -> float:
        qa = apply_rope(q, *rope_cos_sin(32, 1e4, torch.tensor([a])))
        kb = apply_rope(k, *rope_cos_sin(32, 1e4, torch.tensor([b])))
        return float((qa * kb).sum())

    assert math.isclose(score(m, n), score(m + shift, n + shift), rel_tol=1e-3, abs_tol=1e-3)
