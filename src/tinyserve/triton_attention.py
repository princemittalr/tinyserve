"""Paged decode attention as one Triton kernel launch per layer (M4).

M4 step 1 measured the PyTorch path on the T4: at 64 sequences, attention was 94% of a
forward pass, and the GPU was busy only 18% of the time. ``forward_batch`` calls
``paged_gqa_attention`` once per sequence per layer, and each call launches about 17 small
kernels (gather, repeat, einsum, mask, softmax, ...): ~30,600 launches per step, so the
GPU mostly waits for Python. This kernel does a whole decode batch in ONE launch, and
reads each sequence's keys and values straight from its cache blocks, with no gather copy.

Decode only: every sequence brings exactly one new token (one query row). Prefill keeps
the PyTorch path.

One program instance per (sequence, query head). It walks the sequence's blocks in
order and keeps a running ("online") softmax, so it never holds all the scores at once:

    m       = largest score seen so far                   (starts at -inf)
    sum_exp = sum of exp(score - m) over the scores seen  (starts at 0)
    acc     = sum of exp(score - m) * v over the same     (starts at 0, [D])

When a block raises the maximum from m to m_new, everything summed so far was scaled by
exp(-m) and must be rescaled by alpha = exp(m - m_new). At the end, out = acc / sum_exp,
which equals softmax(scores) @ v exactly (up to rounding).

M8 adds ``paged_decode_attention_grouped``: the same math with one program per
(sequence, kv head), so the query heads of a GQA group share each k/v load. M7 measured
the per-head kernel loading every k/v twice for Qwen3-0.6B (16 query heads, 8 kv heads).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_kernel(
    q_ptr,  # [B, Hq, D]
    k_ptr,  # [num_slots, Hkv, D], one layer of the cache
    v_ptr,  # [num_slots, Hkv, D]
    out_ptr,  # [B, Hq, D]
    block_tables_ptr,  # [B, max_blocks] int32: block ids of each sequence, in order
    context_lens_ptr,  # [B] int32: tokens of each sequence, the new one included
    scale,
    stride_qb,
    stride_qh,
    stride_ob,
    stride_oh,
    stride_ks,
    stride_kh,
    stride_bt,
    GROUP: tl.constexpr,  # query heads per kv head
    BLOCK_SIZE: tl.constexpr,  # tokens per cache block
    HEAD_DIM: tl.constexpr,
):
    seq = tl.program_id(0)
    head = tl.program_id(1)
    # TODO(you) 1: the kv head this query head reads (heads 0,1 -> kv 0; 2,3 -> kv 1; ...).
    kv_head = head // GROUP
    ctx_len = tl.load(context_lens_ptr + seq)

    d = tl.arange(0, HEAD_DIM)  # [D]: offsets inside one head vector
    t = tl.arange(0, BLOCK_SIZE)  # [BLOCK_SIZE]: token offsets inside one block
    q = tl.load(q_ptr + seq * stride_qb + head * stride_qh + d).to(tl.float32)

    m = tl.full([], float("-inf"), tl.float32)
    sum_exp = tl.full([], 0.0, tl.float32)
    acc = tl.zeros([HEAD_DIM], tl.float32)

    # TODO(you) 2: how many blocks hold this sequence's ctx_len tokens? (round up: tl.cdiv)
    num_blocks = tl.cdiv(ctx_len, BLOCK_SIZE)
    for j in range(0, num_blocks):
        block = tl.load(block_tables_ptr + seq * stride_bt + j)
        # TODO(you) 3: [BLOCK_SIZE] bool, True where the token's position is inside the
        # sequence. Block j holds positions j * BLOCK_SIZE + t; the last block may be partial.
        valid = j * BLOCK_SIZE + t < ctx_len
        # TODO(you) 4: [BLOCK_SIZE] cache slots of this block's tokens (as in BlockTable.slot).
        slot = block * BLOCK_SIZE + t
        off = slot[:, None] * stride_ks + kv_head * stride_kh + d[None, :]  # [BLOCK_SIZE, D]
        k = tl.load(k_ptr + off, mask=valid[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_ptr + off, mask=valid[:, None], other=0.0).to(tl.float32)

        # TODO(you) 5: [BLOCK_SIZE] scores: dot product of q with each row of k, times scale.
        # (No tl.dot: it needs >= 16 rows on each side, and decode has one query.)
        s = tl.sum(k * q[None, :], axis=1) * scale
        s = tl.where(valid, s, float("-inf"))
        # TODO(you) 6: the new running maximum: the old one or this block's largest score.
        m_new = tl.maximum(m, tl.max(s, axis=0))
        # TODO(you) 7: rescale factor for everything summed so far (see the module docstring).
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new)
        # TODO(you) 8: rescale the old sum, then add this block's exp(score - m_new).
        sum_exp = sum_exp * alpha + tl.sum(p, axis=0)
        # TODO(you) 9: rescale the old acc, then add this block's p-weighted rows of v ([D]).
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m = m_new

    # TODO(you) 10: normalise: softmax weights must sum to 1.
    out = acc / sum_exp
    tl.store(out_ptr + seq * stride_ob + head * stride_oh + d, out.to(out_ptr.dtype.element_ty))


@triton.jit
def _paged_decode_grouped_kernel(
    q_ptr,  # [B, Hq, D]
    k_ptr,  # [num_slots, Hkv, D], one layer of the cache
    v_ptr,  # [num_slots, Hkv, D]
    out_ptr,  # [B, Hq, D]
    block_tables_ptr,  # [B, max_blocks] int32
    context_lens_ptr,  # [B] int32
    scale,
    stride_qb,
    stride_qh,
    stride_ob,
    stride_oh,
    stride_ks,
    stride_kh,
    stride_bt,
    GROUP: tl.constexpr,  # query heads per kv head (a power of 2)
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """M8: the same online softmax, one program per (sequence, KV head) instead of per
    query head. The GROUP query heads that share a kv head are handled together: each
    block of k/v is loaded once and used for all of them, where ``_paged_decode_kernel``
    loads it GROUP times (once per query-head program). Everything per head (m, sum_exp,
    acc, the scores) gains a leading [GROUP] axis."""
    seq = tl.program_id(0)
    kv_head = tl.program_id(1)
    ctx_len = tl.load(context_lens_ptr + seq)

    g = tl.arange(0, GROUP)
    heads = kv_head * GROUP + g  # [GROUP]: the query heads that read this kv head
    d = tl.arange(0, HEAD_DIM)
    t = tl.arange(0, BLOCK_SIZE)
    q_off = seq * stride_qb + heads[:, None] * stride_qh + d[None, :]  # [GROUP, D]
    q = tl.load(q_ptr + q_off).to(tl.float32)

    m = tl.full([GROUP], float("-inf"), tl.float32)
    sum_exp = tl.zeros([GROUP], tl.float32)
    acc = tl.zeros([GROUP, HEAD_DIM], tl.float32)

    num_blocks = tl.cdiv(ctx_len, BLOCK_SIZE)
    for j in range(0, num_blocks):
        block = tl.load(block_tables_ptr + seq * stride_bt + j)
        valid = j * BLOCK_SIZE + t < ctx_len
        slot = block * BLOCK_SIZE + t
        off = slot[:, None] * stride_ks + kv_head * stride_kh + d[None, :]  # [BLOCK_SIZE, D]
        k = tl.load(k_ptr + off, mask=valid[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_ptr + off, mask=valid[:, None], other=0.0).to(tl.float32)

        # [GROUP, BLOCK_SIZE]: each query head's dot product with each key of the block.
        s = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * scale
        s = tl.where(valid[None, :], s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.exp(m - m_new)  # [GROUP]
        p = tl.exp(s - m_new[:, None])  # [GROUP, BLOCK_SIZE]
        sum_exp = sum_exp * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v[None, :, :], axis=1)
        m = m_new

    out = acc / sum_exp[:, None]
    o_off = seq * stride_ob + heads[:, None] * stride_oh + d[None, :]
    tl.store(out_ptr + o_off, out.to(out_ptr.dtype.element_ty))


def paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    block_size: int,
) -> torch.Tensor:
    """Decode attention for B sequences in one kernel launch.

    Same result as ``paged_decode_attention_ref`` (attention.py) within the tolerance in
    tests/test_triton_attention.py. Checks its arguments, then calls ``launch_paged_decode``.

    Args:
        q: ``[B, Hq, D]``, each sequence's one new query, RoPE-rotated.
        k_cache, v_cache: ``[num_slots, Hkv, D]``, one layer of the KV cache; the new
            tokens' k/v must already be written.
        block_tables: ``[B, max_blocks]`` int32; row b lists sequence b's blocks in
            position order. Entries past its last block are never read.
        context_lens: ``[B]`` int32, tokens of each sequence including the new one (>= 1).
        scale: multiplies the scores; Qwen3 uses ``1 / sqrt(D)``.
        block_size: tokens per cache block.

    Returns:
        ``[B, Hq, D]`` in q's dtype.
    """
    check_decode_args(q, k_cache, v_cache, block_tables, context_lens, block_size)
    return launch_paged_decode(q, k_cache, v_cache, block_tables, context_lens, scale, block_size)


def check_decode_args(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
) -> None:
    """Raise ValueError if the kernel's arguments don't fit together.

    The last check reads ``context_lens`` back to the host, which waits for the GPU (a
    host sync). That is why it is separate from the launch: a CUDA graph (M8) cannot
    capture a sync, and the graph path fills ``context_lens`` itself (batch.py).
    """
    b, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    if num_q_heads % num_kv_heads:
        raise ValueError(f"{num_q_heads} query heads do not split into {num_kv_heads} groups")
    for name, n in (("head_dim", head_dim), ("block_size", block_size)):
        if n & (n - 1):
            raise ValueError(f"{name} must be a power of 2 for tl.arange, got {n}")
    if k_cache.shape != v_cache.shape or k_cache.shape[2] != head_dim:
        raise ValueError(f"cache shapes {k_cache.shape} / {v_cache.shape} vs q {q.shape}")
    if block_tables.dtype != torch.int32 or context_lens.dtype != torch.int32:
        raise ValueError("block_tables and context_lens must be int32")
    if context_lens.shape != (b,) or block_tables.shape[0] != b:
        raise ValueError(f"need one block-table row and one length per sequence ({b})")
    if q.stride(2) != 1 or k_cache.stride(2) != 1 or v_cache.stride() != k_cache.stride():
        raise ValueError("the head dimension must be contiguous, and k/v laid out alike")
    if bool((context_lens < 1).any()):
        raise ValueError("every sequence needs at least one token")


def launch_paged_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    block_size: int,
) -> torch.Tensor:
    """The kernel launch alone: no checks and no host sync, so a CUDA graph can capture it.

    Same arguments and result as ``paged_decode_attention``; the caller guarantees what
    ``check_decode_args`` would check.
    """
    b, num_q_heads, head_dim = q.shape
    out = torch.empty_like(q)
    _paged_decode_kernel[(b, num_q_heads)](
        q, k_cache, v_cache, out, block_tables, context_lens,
        scale,
        q.stride(0), q.stride(1),
        out.stride(0), out.stride(1),
        k_cache.stride(0), k_cache.stride(1),
        block_tables.stride(0),
        GROUP=num_q_heads // k_cache.shape[1],
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
    )  # fmt: skip
    return out


def paged_decode_attention_grouped(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    block_size: int,
) -> torch.Tensor:
    """``paged_decode_attention`` with the grouped kernel (M8): same arguments, same result.

    One program per (sequence, kv head), so each k/v block is loaded once per head group
    instead of once per query head. Needs ``Hq / Hkv`` to be a power of 2 (for tl.arange).
    """
    check_decode_args(q, k_cache, v_cache, block_tables, context_lens, block_size)
    group = q.shape[1] // k_cache.shape[1]
    if group & (group - 1):
        raise ValueError(f"the grouped kernel needs a power-of-2 head group, got {group}")
    return launch_paged_decode_grouped(
        q, k_cache, v_cache, block_tables, context_lens, scale, block_size
    )


def launch_paged_decode_grouped(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    block_size: int,
) -> torch.Tensor:
    """The grouped kernel's launch alone: no checks, no host sync (CUDA-graph safe)."""
    b, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    out = torch.empty_like(q)
    _paged_decode_grouped_kernel[(b, num_kv_heads)](
        q, k_cache, v_cache, out, block_tables, context_lens,
        scale,
        q.stride(0), q.stride(1),
        out.stride(0), out.stride(1),
        k_cache.stride(0), k_cache.stride(1),
        block_tables.stride(0),
        GROUP=num_q_heads // num_kv_heads,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
    )  # fmt: skip
    return out
