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
