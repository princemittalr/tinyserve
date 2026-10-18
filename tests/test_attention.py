"""Spec for tinyserve.attention. Do not edit these tests to make them pass."""

import math

import pytest
import torch
import torch.nn.functional as F

from tinyserve.attention import causal_gqa_attention


def rand(*shape: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def test_first_token_sees_only_itself() -> None:
    q, k, v = rand(5, 4, 8, seed=1), rand(5, 2, 8, seed=2), rand(5, 2, 8, seed=3)
    out = causal_gqa_attention(q, k, v, scale=1 / math.sqrt(8))
    # Token 0 can attend only to token 0, so its output is exactly v[0] for its kv head.
    torch.testing.assert_close(out[0], v[0].repeat_interleave(2, dim=0))


def test_zero_query_averages_the_past() -> None:
    # q = 0 makes every score equal, so each token averages v over tokens 0..i.
    t = 6
    v = rand(t, 1, 4)
    out = causal_gqa_attention(torch.zeros(t, 1, 4), rand(t, 1, 4), v, scale=1.0)
    for i in range(t):
        torch.testing.assert_close(out[i, 0], v[: i + 1, 0].mean(dim=0))


def test_gqa_head_mapping_is_grouped_not_round_robin() -> None:
    # 4 query heads, 2 kv heads: q heads 0,1 -> kv 0 and q heads 2,3 -> kv 1.
    t = 3
    v = torch.stack([torch.full((t, 4), 1.0), torch.full((t, 4), 2.0)], dim=1)  # kv0=1, kv1=2
    out = causal_gqa_attention(torch.zeros(t, 4, 4), torch.zeros(t, 2, 4), v, scale=1.0)
    assert out[:, 0].eq(1.0).all() and out[:, 1].eq(1.0).all()
    assert out[:, 2].eq(2.0).all() and out[:, 3].eq(2.0).all()


def test_future_tokens_do_not_leak() -> None:
    q, k, v = rand(6, 2, 8, seed=1), rand(6, 2, 8, seed=2), rand(6, 2, 8, seed=3)
    out = causal_gqa_attention(q, k, v, scale=0.3)
    v2 = v.clone()
    v2[4:] += 100.0  # changing tokens 4 and 5 must not change outputs 0..3
    k2 = k.clone()
    k2[4:] -= 50.0
    out2 = causal_gqa_attention(q, k2, v2, scale=0.3)
    torch.testing.assert_close(out[:4], out2[:4])


@pytest.mark.parametrize(("t", "hq", "hkv", "d"), [(1, 2, 2, 8), (7, 4, 2, 16), (33, 16, 8, 128)])
def test_matches_pytorch_sdpa(t: int, hq: int, hkv: int, d: int) -> None:
    q, k, v = rand(t, hq, d, seed=1), rand(t, hkv, d, seed=2), rand(t, hkv, d, seed=3)
    ours = causal_gqa_attention(q, k, v, scale=1 / math.sqrt(d))
    # PyTorch's built-in wants [heads, T, D].
    ref = F.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), is_causal=True, enable_gqa=True
    ).transpose(0, 1)
    torch.testing.assert_close(ours, ref, atol=1e-5, rtol=1e-4)


def test_fp16_large_scores_stay_finite() -> None:
    # q.k = 64 * 40 * 40 = 102400 > fp16 max. Softmax must run in fp32.
    q = torch.full((3, 1, 64), 40.0, dtype=torch.float16)
    k = torch.full((3, 1, 64), 40.0, dtype=torch.float16)
    v = rand(3, 1, 64).half()
    out = causal_gqa_attention(q, k, v, scale=1.0)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
