"""Reference attention in plain PyTorch: one sequence, causal, grouped-query (GQA).

This is the slow, obviously-correct version. The paged KV cache (M2) and the Triton
kernel (M4) will be tested against it, so correctness here matters more than speed.

``batched_prefill_attention`` (M8) is the exception: the same math for many prefilling
sequences in one call, tested against ``paged_gqa_attention`` run one sequence at a time.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


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


@dataclass(frozen=True)
class PrefillPlan:
    """Where R prefill sequences' queries and keys are, padded to one rectangle (M8).

    ``batched_prefill_attention`` attends all R sequences in one call instead of one
    ``paged_gqa_attention`` per sequence. Sequence r brings ``n_r`` queries (its last
    ``n_r`` positions) and attends ``total_r`` keys; both are padded to the longest.

    Attributes:
        rows: ``[R, Lq]`` flat batch row of query i of sequence r. Padding repeats the
            sequence's last row, so every index is valid; its result is dropped.
        slots: ``[R, Lk]`` cache slot of key j of sequence r; padding is slot 0, masked.
        mask: ``[R, 1, Lq, Lk]`` True where query i may attend key j: j is not after
            the query's position (causal), hence one of the sequence's own keys.
        real: ``[R, Lq]`` True for real queries, False for padding.
    """

    rows: torch.Tensor
    slots: torch.Tensor
    mask: torch.Tensor
    real: torch.Tensor


def prefill_plan(
    starts: list[int], query_lens: list[int], seq_slots: list[torch.Tensor]
) -> PrefillPlan:
    """Build the plan for sequences whose queries start at flat row ``starts[r]``.

    ``seq_slots[r]`` lists the slots of all of sequence r's tokens in position order, the
    new ones included (as in ``Batch.seq_slots``). Built once per step, used by every layer.
    """
    device = seq_slots[0].device
    lq = max(query_lens)
    n = torch.tensor(query_lens, device=device)[:, None]  # [R, 1]
    total = torch.tensor([s.shape[0] for s in seq_slots], device=device)[:, None]
    i = torch.arange(lq, device=device)[None, :]  # [1, Lq]
    last = torch.minimum(i, n - 1)  # padding queries repeat the sequence's last query
    rows = torch.tensor(starts, device=device)[:, None] + last
    slots = torch.nn.utils.rnn.pad_sequence(seq_slots, batch_first=True, padding_value=0)
    q_pos = total - n + last  # [R, Lq]: query i is the sequence's position total - n + i
    k_pos = torch.arange(slots.shape[1], device=device)[None, None, :]  # [1, 1, Lk]
    # Causal: key j only up to the query's position. Every position is < total, so this
    # also keeps every query off the padding keys.
    mask = k_pos <= q_pos[:, :, None]
    return PrefillPlan(rows=rows, slots=slots, mask=mask[:, None], real=i < n)


def batched_prefill_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    plan: PrefillPlan,
    scale: float,
) -> torch.Tensor:
    """``paged_gqa_attention`` for every sequence of ``plan`` in one call (M8).

    Gathers each sequence's queries and keys into padded ``[R, H, L, D]`` tensors and runs
    ``F.scaled_dot_product_attention`` once with ``plan.mask``. M8 step 3 measured why: a
    C=32 step that prefills ~11 prompts took 224 ms, mostly per-sequence attention calls of
    ~17 small kernels each, per layer. Like the reference, it computes in float32.

    Args:
        q: ``[T, Hq, D]``, the whole batch's queries (RoPE-rotated).
        k_cache, v_cache: ``[num_slots, Hkv, D]``, one layer; new k/v already written.
        plan: from ``prefill_plan``.
        scale: as in ``causal_gqa_attention``.

    Returns:
        ``[R, Lq, Hq, D]`` in q's dtype; row ``plan.rows[r, i]`` of the batch gets
        ``[r, i]`` where ``plan.real[r, i]``.
    """
    group = q.shape[1] // k_cache.shape[1]
    qb = q[plan.rows].float().transpose(1, 2)  # [R, Hq, Lq, D]
    k = k_cache[plan.slots].float().repeat_interleave(group, dim=2).transpose(1, 2)
    v = v_cache[plan.slots].float().repeat_interleave(group, dim=2).transpose(1, 2)
    out = F.scaled_dot_product_attention(qb, k, v, attn_mask=plan.mask, scale=scale)
    return out.transpose(1, 2).to(q.dtype)
