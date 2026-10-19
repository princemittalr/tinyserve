"""Fused Triton kernels for the small per-layer ops around attention (M9).

M8 step 3 measured a C=32 decode step with CUDA graphs at ~17 ms of GPU time, ~5.7 ms of
it in small elementwise kernels: RMSNorm, the q/k head norms, RoPE and the k/v copies
into the cache. Each is several PyTorch kernels (an upcast, a square, a mean, an rsqrt,
a multiply, a downcast, then RoPE's slices, products, cat and another downcast), and
each kernel reads and writes its whole tensor from memory again. These two kernels do
the same math in one pass each:

``rms_norm_rows``: ``layers.rms_norm`` over the rows of ``[T, N]`` (the hidden-state
    norms: input, post-attention and final), one program per row.

``qk_norm_rope_store``: for every token, q_norm + RoPE on each query head (returned),
    and k_norm + RoPE on each key head, written straight into the KV cache together
    with the value head: what ``rms_norm``, ``apply_rope`` and ``KVCache.write`` do for
    q, k and v in ``Qwen3._layer``. One program per (token, head).

Both follow ``layers.py``'s numerics contract step by step: normalise in float32, cast
to the input dtype, multiply by the weight in that dtype, then rotate in float32 and
cast back once. Only the order of the float32 sum inside the mean differs from
PyTorch's, so float32 results agree within 1e-5 and float16 within one fp16 rounding.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_rows_kernel(x_ptr, w_ptr, out_ptr, eps, stride_x, stride_o, N: tl.constexpr,
                          N_PAD: tl.constexpr):  # fmt: skip
    row = tl.program_id(0)
    i = tl.arange(0, N_PAD)
    ok = i < N
    x = tl.load(x_ptr + row * stride_x + i, mask=ok, other=0.0)
    xf = x.to(tl.float32)
    inv = 1.0 / tl.sqrt_rn(tl.sum(xf * xf, axis=0) / N + eps)
    w = tl.load(w_ptr + i, mask=ok, other=0.0)
    y = (xf * inv).to(x.dtype) * w  # cast, then the weight in the input dtype
    tl.store(out_ptr + row * stride_o + i, y, mask=ok)


@triton.jit
def _norm_rope(base, w_ptr, cos, sin, i, eps, HEAD_DIM: tl.constexpr):
    """One head vector at ``base``: RMSNorm with weight ``w``, then RoPE (rotate half).
    Returns the two rotated halves (a', b') in float32, rounded like ``apply_rope``'s
    input (to the input dtype after the norm)."""
    half: tl.constexpr = HEAD_DIM // 2
    xa = tl.load(base + i)
    xb = tl.load(base + half + i)
    fa, fb = xa.to(tl.float32), xb.to(tl.float32)
    inv = 1.0 / tl.sqrt_rn((tl.sum(fa * fa, axis=0) + tl.sum(fb * fb, axis=0)) / HEAD_DIM + eps)
    a = ((fa * inv).to(xa.dtype) * tl.load(w_ptr + i)).to(tl.float32)
    b = ((fb * inv).to(xa.dtype) * tl.load(w_ptr + half + i)).to(tl.float32)
    return a * cos - b * sin, b * cos + a * sin


@triton.jit
def _qk_norm_rope_store_kernel(
    q_ptr,  # [T, Hq, D]
    k_ptr,  # [T, Hkv, D]
    v_ptr,  # [T, Hkv, D]
    q_out_ptr,  # [T, Hq, D]
    q_w_ptr,  # [D] q_norm weight
    k_w_ptr,  # [D] k_norm weight
    cos_ptr,  # [T, D] float32 (rope_cos_sin)
    sin_ptr,
    slots_ptr,  # [T] int64: the cache slot of each token
    k_cache_ptr,  # [num_slots, Hkv, D], one layer
    v_cache_ptr,
    eps,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_ot,
    stride_oh,
    stride_ct,
    stride_cs,  # cache: per slot
    stride_ch,  # cache: per kv head
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)  # 0 .. Hq - 1: a query head; Hq .. Hq + Hkv - 1: a kv head
    half: tl.constexpr = HEAD_DIM // 2
    i = tl.arange(0, half)
    cos = tl.load(cos_ptr + t * stride_ct + i)  # both halves hold the same angles
    sin = tl.load(sin_ptr + t * stride_ct + i)
    if h < NUM_Q_HEADS:
        a, b = _norm_rope(q_ptr + t * stride_qt + h * stride_qh, q_w_ptr, cos, sin, i, eps,
                          HEAD_DIM)  # fmt: skip
        out = q_out_ptr + t * stride_ot + h * stride_oh
        tl.store(out + i, a.to(q_out_ptr.dtype.element_ty))
        tl.store(out + half + i, b.to(q_out_ptr.dtype.element_ty))
    else:
        # Names of their own: a variable set in both branches of an ``if`` must have one
        # type, and these are fp16 when the cache is (the interpreter does not check).
        kh = h - NUM_Q_HEADS
        ka, kb = _norm_rope(k_ptr + t * stride_kt + kh * stride_kh, k_w_ptr, cos, sin, i, eps,
                            HEAD_DIM)  # fmt: skip
        # rope's cast to k's dtype, then KVCache.write's cast to the cache dtype
        k_a = ka.to(k_ptr.dtype.element_ty).to(k_cache_ptr.dtype.element_ty)
        k_b = kb.to(k_ptr.dtype.element_ty).to(k_cache_ptr.dtype.element_ty)
        slot = tl.load(slots_ptr + t)
        dst = slot * stride_cs + kh * stride_ch
        tl.store(k_cache_ptr + dst + i, k_a)
        tl.store(k_cache_ptr + dst + half + i, k_b)
        d = tl.arange(0, HEAD_DIM)
        v = tl.load(v_ptr + t * stride_vt + kh * stride_vh + d)
        tl.store(v_cache_ptr + dst + d, v.to(v_cache_ptr.dtype.element_ty))


def rms_norm_rows(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``layers.rms_norm`` for a ``[T, N]`` tensor in one launch (one program per row)."""
    if x.dim() != 2 or x.stride(1) != 1 or weight.shape != (x.shape[1],):
        raise ValueError(f"need a row-contiguous [T, N] tensor and an [N] weight: {x.shape}")
    out = torch.empty_like(x)
    if x.shape[0]:
        n = x.shape[1]
        _rms_norm_rows_kernel[(x.shape[0],)](
            x, weight, out, eps, x.stride(0), out.stride(0), N=n, N_PAD=triton.next_power_of_2(n)
        )
    return out


def qk_norm_rope_store(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    slots: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """q/k head norms, RoPE and the cache write of one layer, in one launch.

    Equals ``apply_rope(rms_norm(q, q_weight, eps), cos, sin)``, returned, while
    ``apply_rope(rms_norm(k, k_weight, eps), cos, sin)`` and ``v`` are written to
    ``k_cache[slots]`` / ``v_cache[slots]`` like ``KVCache.write``. No host sync, so CUDA
    graphs can capture it.

    Args:
        q: ``[T, Hq, D]``; k, v: ``[T, Hkv, D]``. The head dimension must be contiguous.
        q_weight, k_weight: ``[D]``.
        cos, sin: ``[T, D]`` float32 from ``rope_cos_sin``.
        slots: ``[T]`` int64 cache slots.
        k_cache, v_cache: ``[num_slots, Hkv, D]``, one layer of the cache.
    """
    t, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    if head_dim % 2 or (head_dim // 2) & (head_dim // 2 - 1):
        raise ValueError(f"head_dim must be twice a power of 2, got {head_dim}")
    if k.shape != v.shape or k.shape[0] != t or k_cache.shape[1:] != k.shape[1:]:
        raise ValueError(f"shapes q {q.shape}, k {k.shape}, v {v.shape}, cache {k_cache.shape}")
    if any(x.stride(2) != 1 for x in (q, k, v, k_cache)) or v_cache.stride() != k_cache.stride():
        raise ValueError("the head dimension must be contiguous, and k/v caches laid out alike")
    if cos.shape != (t, head_dim) or cos.stride() != sin.stride() or cos.stride(1) != 1:
        raise ValueError(f"cos and sin must be [T, D] = {(t, head_dim)}, laid out alike")
    out = torch.empty_like(q)
    if t:
        _qk_norm_rope_store_kernel[(t, num_q_heads + num_kv_heads)](
            q, k, v, out, q_weight, k_weight, cos, sin, slots, k_cache, v_cache,
            eps,
            q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
            out.stride(0), out.stride(1), cos.stride(0),
            k_cache.stride(0), k_cache.stride(1),
            NUM_Q_HEADS=num_q_heads,
            HEAD_DIM=head_dim,
        )  # fmt: skip
    return out
