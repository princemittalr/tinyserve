"""Building blocks of the Qwen3 forward pass: RMSNorm and rotary position embeddings.

YOU implement the three functions below (ownership rule). ``tests/test_layers.py``
is the spec: make it pass without editing it.

Shape conventions used throughout TinyServe:
    T = number of tokens in the batch (all sequences flattened together)
    H = number of heads, D = head_dim
"""

import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dimension (Zhang & Sennrich, 2019).

        y = x / sqrt(mean(x**2, over last dim) + eps) * weight

    Args:
        x: ``[..., N]``, any leading shape. Qwen3 uses this on ``[T, hidden]`` and,
           for q_norm / k_norm, on ``[T, H, D]`` (normalising each head separately).
        weight: ``[N]``, the learned per-channel scale.
        eps: added inside the square root, so an all-zero row gives 0, not nan.

    Returns:
        Same shape and dtype as ``x``.

    Numerics contract (this is what the HF reference does; matching it matters for
    token-identical output in M1):
        1. upcast x to float32 and normalise in float32
           (in fp16, 300**2 = 90000 already overflows: fp16 max is 65504);
        2. cast the normalised value back to x.dtype;
        3. multiply by weight.
    """
    # TODO(you): about 4 lines. Useful: Tensor.float(), .pow(2), .mean(dim=-1, keepdim=True),
    # torch.rsqrt, .to(dtype).
    raise NotImplementedError("rms_norm")


def rope_cos_sin(
    head_dim: int, theta: float, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the RoPE rotation tables for the given token positions (Su et al., 2021).

    Dimension pair ``i`` (for i in 0 .. head_dim/2 - 1) rotates at frequency
        inv_freq[i] = theta ** (-2i / head_dim)
    so a token at position p rotates pair i by the angle  p * inv_freq[i].

    Args:
        head_dim: D, must be even.
        theta: the base (``rope_theta`` in the config; 1e6 for Qwen3).
        positions: ``[T]`` integer tensor. Positions are passed explicitly, not assumed
            to be 0..T-1, because in a batch every sequence is at a different position.

    Returns:
        ``(cos, sin)``, each ``[T, D]`` float32, laid out as ``cat([angles, angles], -1)``,
        i.e. the first half and second half of the last dim hold the same angles.
        Compute angles in float32 or better: positions reach 40960.
    """
    # TODO(you): about 5 lines. Useful: torch.arange, torch.outer, torch.cat, .cos(), .sin().
    raise NotImplementedError("rope_cos_sin")


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate q or k by position, using the "rotate half" layout Qwen3 was trained with.

    Pair i is made of elements ``i`` and ``i + D/2`` (NOT neighbours 2i and 2i+1;
    that "interleaved" layout is a different model convention and gives wrong tokens).
    For each pair (a, b) and its angle t:
        a' = a * cos(t) - b * sin(t)
        b' = b * cos(t) + a * sin(t)

    Args:
        x: ``[T, H, D]`` queries or keys, any float dtype.
        cos, sin: ``[T, D]`` float32 from :func:`rope_cos_sin`.

    Returns:
        ``[T, H, D]`` in x's dtype. Do the math in float32, then cast back.
    """
    # TODO(you): about 6 lines. cos/sin need a head axis to broadcast against [T, H, D]
    # (Tensor.unsqueeze). Splitting x in halves: x[..., :half], x[..., half:].
    raise NotImplementedError("apply_rope")
