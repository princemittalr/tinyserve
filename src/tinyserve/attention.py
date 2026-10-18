"""Reference attention in plain PyTorch: one sequence, causal, grouped-query (GQA).

This is the slow, obviously-correct version. The paged KV cache (M2) and the Triton
kernel (M4) will be tested against it, so correctness here matters more than speed.
"""

import torch


def causal_gqa_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> torch.Tensor:
    """softmax(q k^T * scale + causal mask) v, for a single sequence.

    Grouped-query attention: there are fewer key/value heads than query heads
    (Qwen3-0.6B: 16 query heads, 8 kv heads). Query heads are split into
    ``Hq // Hkv`` consecutive groups, and every head in group g reads kv head g.
    So query head h uses kv head ``h // (Hq // Hkv)``: heads 0,1 -> kv 0; heads 2,3 -> kv 1.

    Args:
        q: ``[T, Hq, D]`` queries, already RoPE-rotated.
        k: ``[T, Hkv, D]`` keys, already RoPE-rotated.
        v: ``[T, Hkv, D]`` values.
        scale: multiplies the scores; Qwen3 uses ``1 / sqrt(D)``.

    Returns:
        ``[T, Hq, D]`` in q's dtype. Scores and softmax are computed in float32:
        in fp16 a score above 65504 becomes inf and softmax returns nan.
    """
    num_q_heads, num_kv_heads = q.shape[1], k.shape[1]
    t = q.shape[0]

    # TODO(you) 1: how many query heads share one kv head?
    group = num_q_heads // num_kv_heads
    # TODO(you) 2: copy each kv head `group` times along the head axis, so [T, Hkv, D]
    # becomes [T, Hq, D] in the order 0,0,1,1,... (repeat_interleave, not repeat).
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)

    # scores[h, i, j] = how much token i attends to token j, in head h.
    # TODO(you) 3: multiply by the scale.
    scores = torch.einsum("ihd,jhd->hij", q.float(), k.float()) * scale

    # Causal mask: token i may only look at tokens j <= i. True marks the forbidden
    # entries above the diagonal. TODO(you) 4: which diagonal starts the forbidden part?
    future = torch.ones(t, t, dtype=torch.bool, device=q.device).triu(diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))

    # TODO(you) 5: softmax over j, the tokens being attended to: which dim of [h, i, j]?
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hij,jhd->ihd", probs, v.float())
    # TODO(you) 6: return in the same dtype as the input queries.
    return out.to(q.dtype)


def paged_gqa_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slots: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Attention for the NEWEST ``n`` tokens of one sequence, reading k/v from the cache.

    The sequence has ``total = len(slots)`` tokens so far; ``slots[p]`` is where token
    position p's key and value live in the cache. The queries are the last ``n`` positions,
    ``total - n .. total - 1``. Prefill passes all positions (n == total); a decode step
    passes one query (n == 1). The new tokens' k/v must already be written to the cache.

    The result must equal rows ``total - n ..`` of ``causal_gqa_attention`` on the
    contiguous k/v; ``tests/test_paged_attention.py`` checks exactly that.

    Args:
        q: ``[n, Hq, D]``, RoPE-rotated.
        k_cache, v_cache: ``[num_slots, Hkv, D]``, one layer of the KV cache.
        slots: ``[total]`` int64, from the sequence's block table, in position order.
        scale: as in ``causal_gqa_attention``.

    Returns:
        ``[n, Hq, D]`` in q's dtype.
    """
    n, total = q.shape[0], slots.shape[0]
    group = q.shape[1] // k_cache.shape[1]

    # TODO(you) 1-2: gather this sequence's keys and values from the cache, in position
    # order. Indexing a tensor with a tensor of indices picks those rows: [total, Hkv, D].
    k = k_cache[slots]
    v = v_cache[slots]
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)

    # Absolute positions: query row i is token position start + i.
    # TODO(you) 3: position of the first query (the queries are the LAST n positions).
    start = total - n
    q_pos = torch.arange(start, total, device=q.device)
    k_pos = torch.arange(total, device=q.device)
    # TODO(you) 4: forbidden = key position j is after query position i. Compare
    # k_pos[None, :] with q_pos[:, None] so the result is [n, total].
    future = k_pos[None, :] > q_pos[:, None]

    scores = torch.einsum("ihd,jhd->hij", q.float(), k.float()) * scale
    scores = scores.masked_fill(future, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hij,jhd->ihd", probs, v.float())
    return out.to(q.dtype)


def paged_decode_attention_ref(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    block_size: int,
) -> torch.Tensor:
    """The reference for the M4 Triton kernel: same arguments, same result.

    One decode query per sequence. Turns each block-table row into that sequence's slots
    and runs ``paged_gqa_attention`` on it, one sequence at a time (the slow path that
    ``forward_batch`` uses in M3).

    Args:
        q: ``[B, Hq, D]``, one new query per sequence.
        block_tables: ``[B, max_blocks]``; row b lists sequence b's blocks in order.
        context_lens: ``[B]``, tokens of each sequence including the new one.
        Others as in ``paged_gqa_attention``.
    """
    outs = []
    for b, n in enumerate(context_lens.tolist()):
        pos = torch.arange(n, device=q.device)
        blocks = block_tables[b].to(torch.int64)[pos // block_size]
        slots = blocks * block_size + pos % block_size
        outs.append(paged_gqa_attention(q[b : b + 1], k_cache, v_cache, slots, scale))
    return torch.cat(outs)
