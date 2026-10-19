"""Paged prefill attention as one Triton kernel launch per layer (M9).

M8 batched prefill attention into one PyTorch call per layer, but that call first
gathers every prefilling sequence's keys and values out of the cache into a padded
``[R, H, L, D]`` float32 tensor (a copy of the whole context, with each kv head repeated
for its query heads), then runs SDPA over the padded rectangle. This kernel reads the
keys and values straight from the cache blocks, like the decode kernels, and skips the
padding: each sequence only does the work of its own queries and keys.

"Varlen": the sequences bring different numbers of queries (a whole prompt, the part
after a cached prefix, or one chunk of a prompt) and attend to different numbers of
keys. Sequence r's queries are flat rows ``q_starts[r] ..`` of the batch, its last
``q_lens[r]`` positions ``context_lens[r] - q_lens[r] .. context_lens[r] - 1``.

One program per (sequence, tile of queries, kv head). Like the grouped decode kernel,
the GROUP query heads that share a kv head are handled together, so each key and value
is loaded once for all of them. The program's rows are (query, head) pairs:

    row = q * GROUP + g   ->   query q of the tile, query head kv_head * GROUP + g

It walks the keys in chunks of BLOCK_N positions, from position 0 up to the tile's last
query position (causal: later keys are never needed), with the online softmax of
triton_attention.py: running max m, running sum l, running output acc, each per row.
Scores are ``tl.dot`` (q [M, D] times k^T [D, BLOCK_N]), so on a GPU they run on tensor
cores in fp16. Each key's cache slot comes from the block table, position by position
(``table[pos // BLOCK_SIZE] * BLOCK_SIZE + pos % BLOCK_SIZE``), so a chunk of keys may
span several cache blocks and the cache block size can be anything.

Precision: q.k is computed from the cache dtype with float32 accumulation. In float32
the dots are exact IEEE float32 (no TF32). In float16, p is rounded to float16 before
p @ v (as FlashAttention does), so fp16 results differ from the float32 reference by
about one fp16 rounding (tested at 1e-3, the kernel tolerance since M4).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_prefill_kernel(
    q_ptr,  # [T, Hq, D]: the whole batch's queries
    k_ptr,  # [num_slots, Hkv, D], one layer of the cache
    v_ptr,  # [num_slots, Hkv, D]
    out_ptr,  # [T, Hq, D]: rows of this kernel's sequences are written, others untouched
    block_tables_ptr,  # [R, max_blocks] int32
    q_starts_ptr,  # [R] int32: flat row of each sequence's first query
    q_lens_ptr,  # [R] int32: queries of each sequence (its newest positions)
    context_lens_ptr,  # [R] int32: tokens of each sequence so far, new ones included
    scale,
    stride_qt,
    stride_qh,
    stride_ot,
    stride_oh,
    stride_ks,
    stride_kh,
    stride_bt,
    GROUP: tl.constexpr,  # query heads per kv head (a power of 2)
    BLOCK_SIZE: tl.constexpr,  # tokens per cache block
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PAD: tl.constexpr,  # HEAD_DIM rounded up to a power of 2, at least 16
    BLOCK_M: tl.constexpr,  # queries per tile; BLOCK_M * GROUP rows per program
    BLOCK_N: tl.constexpr,  # keys per inner step
):
    seq = tl.program_id(0)
    tile = tl.program_id(1)
    kv_head = tl.program_id(2)
    q_len = tl.load(q_lens_ptr + seq)
    if tile * BLOCK_M >= q_len:
        return  # the grid has tiles for the longest sequence; this one is shorter
    q_start = tl.load(q_starts_ptr + seq)
    ctx_len = tl.load(context_lens_ptr + seq)

    r = tl.arange(0, BLOCK_M * GROUP)
    qi = tile * BLOCK_M + r // GROUP  # [M]: which of the sequence's queries
    head = kv_head * GROUP + r % GROUP  # [M]: which query head
    row_ok = qi < q_len
    pos = ctx_len - q_len + qi  # [M]: the query's position in its sequence
    d = tl.arange(0, HEAD_DIM_PAD)
    d_ok = d < HEAD_DIM
    q_off = (q_start + qi)[:, None] * stride_qt + head[:, None] * stride_qh + d[None, :]
    q = tl.load(q_ptr + q_off, mask=row_ok[:, None] & d_ok[None, :], other=0.0)

    m = tl.full([BLOCK_M * GROUP], float("-inf"), tl.float32)
    l = tl.zeros([BLOCK_M * GROUP], tl.float32)  # noqa: E741 (the usual name in the papers)
    acc = tl.zeros([BLOCK_M * GROUP, HEAD_DIM_PAD], tl.float32)

    # Keys up to the tile's last query position: 0 .. key_end - 1.
    key_end = ctx_len - q_len + tl.minimum(tile * BLOCK_M + BLOCK_M, q_len)
    for n0 in range(0, key_end, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)  # [N] key positions
        n_ok = n < key_end
        block = tl.load(block_tables_ptr + seq * stride_bt + n // BLOCK_SIZE, mask=n_ok, other=0)
        slot = block * BLOCK_SIZE + n % BLOCK_SIZE  # [N], as in BlockTable.slot
        kt_off = slot[None, :] * stride_ks + kv_head * stride_kh + d[:, None]  # [D, N]
        kt = tl.load(k_ptr + kt_off, mask=d_ok[:, None] & n_ok[None, :], other=0.0)
        s = tl.dot(q, kt, input_precision="ieee") * scale  # [M, N] float32
        # Causal: key n only up to the row's own position. Key 0 is always allowed, so
        # after the first step every row's m is finite and alpha below is never nan.
        s = tl.where((n[None, :] <= pos[:, None]) & n_ok[None, :], s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new[:, None])  # [M, N]
        l = l * alpha + tl.sum(p, axis=1)  # noqa: E741
        v_off = slot[:, None] * stride_ks + kv_head * stride_kh + d[None, :]  # [N, D]
        v = tl.load(v_ptr + v_off, mask=n_ok[:, None] & d_ok[None, :], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee")
        m = m_new

    out = acc / l[:, None]
    o_off = (q_start + qi)[:, None] * stride_ot + head[:, None] * stride_oh + d[None, :]
    tl.store(
        out_ptr + o_off, out.to(out_ptr.dtype.element_ty), mask=row_ok[:, None] & d_ok[None, :]
    )


def paged_prefill_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    q_starts: torch.Tensor,
    q_lens: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    block_size: int,
    max_q_len: int,
    out: torch.Tensor | None = None,
    rows_per_tile: int | None = None,
    block_n: int = 32,
) -> torch.Tensor:
    """Attention for R prefilling sequences in one launch, reading k/v from the cache.

    Sequence r's queries are rows ``q_starts[r] .. q_starts[r] + q_lens[r] - 1`` of ``q``
    and are its newest positions; the result for them is what ``paged_gqa_attention``
    returns for that sequence (attention.py, the reference in the tests).

    Args:
        q: ``[T, Hq, D]``, the batch's queries, RoPE-rotated. Rows of other sequences are
            ignored.
        k_cache, v_cache: ``[num_slots, Hkv, D]``, one layer; the new k/v already written.
        block_tables: ``[R, max_blocks]`` int32, each sequence's blocks in order.
        q_starts, q_lens, context_lens: ``[R]`` int32 (see the module docstring).
        max_q_len: the largest ``q_lens`` entry, from the host (sizes the grid, no sync).
        out: ``[T, Hq, D]`` to write into (other rows are left as they are); default a
            new ``empty_like(q)``.
        rows_per_tile: (query, head) rows per program, a power of 2 >= 16 and >= GROUP.
            Default 64 for 16-bit inputs, 32 for float32, whose tiles take twice the
            T4's 64 KiB of shared memory per row.
        block_n: keys per inner step, a power of 2 >= 16.

    Returns:
        ``out``.
    """
    t, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    group = num_q_heads // num_kv_heads
    if rows_per_tile is None:
        rows_per_tile = max(32 if q.dtype == torch.float32 else 64, group)
    if num_q_heads % num_kv_heads or group & (group - 1):
        raise ValueError(f"need a power-of-2 head group, got {num_q_heads} / {num_kv_heads}")
    for name, n in (("block_size", block_size), ("rows_per_tile", rows_per_tile),
                    ("block_n", block_n)):  # fmt: skip
        if n & (n - 1):
            raise ValueError(f"{name} must be a power of 2, got {n}")
    if rows_per_tile < max(16, group) or block_n < 16:
        raise ValueError("tl.dot needs at least 16 rows and 16 keys per tile")
    if k_cache.shape != v_cache.shape or k_cache.shape[2] != head_dim:
        raise ValueError(f"cache shapes {k_cache.shape} / {v_cache.shape} vs q {q.shape}")
    if q.stride(2) != 1 or k_cache.stride(2) != 1 or v_cache.stride() != k_cache.stride():
        raise ValueError("the head dimension must be contiguous, and k/v laid out alike")
    r = block_tables.shape[0]
    for name, x in (("q_starts", q_starts), ("q_lens", q_lens), ("context_lens", context_lens)):
        if x.dtype != torch.int32 or x.shape != (r,):
            raise ValueError(f"{name} must be int32 with one entry per sequence ({r})")
    if block_tables.dtype != torch.int32:
        raise ValueError("block_tables must be int32")
    if out is None:
        out = torch.empty_like(q)
    if r == 0 or t == 0:
        return out
    block_m = rows_per_tile // group
    _paged_prefill_kernel[(r, triton.cdiv(max_q_len, block_m), num_kv_heads)](
        q, k_cache, v_cache, out, block_tables, q_starts, q_lens, context_lens,
        scale,
        q.stride(0), q.stride(1),
        out.stride(0), out.stride(1),
        k_cache.stride(0), k_cache.stride(1),
        block_tables.stride(0),
        GROUP=group,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        HEAD_DIM_PAD=max(16, triton.next_power_of_2(head_dim)),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )  # fmt: skip
    return out
