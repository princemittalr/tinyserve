"""Spec for tinyserve.layers.swiglu_mlp. Do not edit these tests to make them pass."""

import torch
import torch.nn.functional as F

from tinyserve.layers import swiglu_mlp


def test_identity_weights() -> None:
    # With identity weights every linear layer is a no-op, so y = silu(x) * x.
    x = torch.tensor([[-2.0, 0.0, 1.0, 3.0]])
    eye = torch.eye(4)
    torch.testing.assert_close(swiglu_mlp(x, eye, eye, eye), F.silu(x) * x)


def test_gate_and_up_are_not_swapped() -> None:
    # gate = 2x, up = x  ->  silu(2x) * x.  Swapping them gives silu(x) * 2x instead.
    x = torch.tensor([[-1.0, 0.5, 2.0]])
    eye = torch.eye(3)
    y = swiglu_mlp(x, 2 * eye, eye, eye)
    torch.testing.assert_close(y, F.silu(2 * x) * x)


def test_zero_gate_closes_everything() -> None:
    # silu(0) = 0, so a zero gate weight blocks all output.
    x = torch.randn(5, 8)
    y = swiglu_mlp(x, torch.zeros(16, 8), torch.randn(16, 8), torch.randn(8, 16))
    assert torch.equal(y, torch.zeros(5, 8))


def test_shapes_follow_qwen3_layout() -> None:
    # hidden 8 -> intermediate 24 -> hidden 8, weights stored [out, in].
    x = torch.randn(3, 8)
    y = swiglu_mlp(x, torch.randn(24, 8), torch.randn(24, 8), torch.randn(8, 24))
    assert y.shape == (3, 8)


def test_keeps_dtype() -> None:
    x = torch.randn(2, 4).half()
    w = torch.randn(4, 4).half()
    assert swiglu_mlp(x, w, w, w).dtype == torch.float16
