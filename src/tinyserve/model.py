"""The Qwen3 forward pass, built from the pieces in layers.py and attention.py.

Two paths share the same layers and differ only in attention:
    ``forward`` (M1): no KV cache. Every generation step re-runs the whole sequence,
        O(n^2) work per token. Kept as the simplest correct baseline.
    ``forward_paged`` (M2): runs only the NEW tokens. Their keys and values are written
        into the paged KV cache, and attention reads the whole history back from it.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F

from tinyserve.attention import causal_gqa_attention, paged_gqa_attention
from tinyserve.block_allocator import BlockAllocator
from tinyserve.block_table import BlockTable
from tinyserve.config import ModelConfig
from tinyserve.kv_cache import KVCache
from tinyserve.layers import apply_rope, rms_norm, rope_cos_sin, swiglu_mlp

# (layer index, q [T, Hq, D], k [T, Hkv, D], v [T, Hkv, D]) -> attention output [T, Hq, D]
AttentionFn = Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


class Qwen3:
    def __init__(self, cfg: ModelConfig, weights: dict[str, torch.Tensor]) -> None:
        self.cfg = cfg
        self.w = weights
        # Tied embeddings: the output projection reuses the input embedding matrix.
        key = "model.embed_tokens.weight" if cfg.tie_word_embeddings else "lm_head.weight"
        self.lm_head = weights[key]

    def _layer(
        self, i: int, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attend: AttentionFn
    ) -> torch.Tensor:
        cfg, p = self.cfg, f"model.layers.{i}."
        w = self.w
        t = x.shape[0]

        # Attention block: pre-norm, then a residual add.
        h = rms_norm(x, w[p + "input_layernorm.weight"], cfg.rms_norm_eps)
        q = F.linear(h, w[p + "self_attn.q_proj.weight"]).view(t, cfg.num_heads, cfg.head_dim)
        k = F.linear(h, w[p + "self_attn.k_proj.weight"]).view(t, cfg.num_kv_heads, cfg.head_dim)
        v = F.linear(h, w[p + "self_attn.v_proj.weight"]).view(t, cfg.num_kv_heads, cfg.head_dim)
        # Qwen3-specific: normalise each head of q and k BEFORE applying RoPE.
        q = rms_norm(q, w[p + "self_attn.q_norm.weight"], cfg.rms_norm_eps)
        k = rms_norm(k, w[p + "self_attn.k_norm.weight"], cfg.rms_norm_eps)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        attn = attend(i, q, k, v)
        x = x + F.linear(attn.reshape(t, cfg.q_size), w[p + "self_attn.o_proj.weight"])

        # MLP block: pre-norm, then a residual add.
        h = rms_norm(x, w[p + "post_attention_layernorm.weight"], cfg.rms_norm_eps)
        mlp = swiglu_mlp(
            h,
            w[p + "mlp.gate_proj.weight"],
            w[p + "mlp.up_proj.weight"],
            w[p + "mlp.down_proj.weight"],
        )
        return x + mlp

    def _run(
        self, token_ids: torch.Tensor, positions: torch.Tensor, attend: AttentionFn
    ) -> torch.Tensor:
        cos, sin = rope_cos_sin(self.cfg.head_dim, self.cfg.rope_theta, positions)
        x = self.w["model.embed_tokens.weight"][token_ids]  # embedding = row lookup
        for i in range(self.cfg.num_layers):
            x = self._layer(i, x, cos, sin, attend)
        x = rms_norm(x, self.w["model.norm.weight"], self.cfg.rms_norm_eps)
        return F.linear(x, self.lm_head).float()

    @torch.inference_mode()
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """``[T]`` token ids of one sequence -> ``[T, vocab]`` float32 logits. No cache."""
        positions = torch.arange(token_ids.shape[0], device=token_ids.device)
        scale = self.cfg.head_dim**-0.5
        return self._run(
            token_ids, positions, lambda i, q, k, v: causal_gqa_attention(q, k, v, scale)
        )

    @torch.inference_mode()
    def forward_paged(
        self,
        token_ids: torch.Tensor,
        cache: KVCache,
        new_slots: torch.Tensor,
        all_slots: torch.Tensor,
    ) -> torch.Tensor:
        """Run the newest ``n`` tokens of one sequence through the model, using the KV cache.

        Args:
            token_ids: ``[n]`` the new tokens: the whole prompt for prefill, one token
                for a decode step.
            cache: the shared KV cache; the new tokens' k/v are written into it.
            new_slots: ``[n]`` cache slots for the new tokens (``BlockTable.append``).
            all_slots: ``[total]`` slots of every token of the sequence so far, new ones
                included, in position order.

        Returns:
            ``[n, vocab]`` float32 logits for the new tokens.
        """
        n, total = token_ids.shape[0], all_slots.shape[0]
        positions = torch.arange(total - n, total, device=token_ids.device)
        scale = self.cfg.head_dim**-0.5

        def attend(i: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            cache.write(i, new_slots, k, v)
            return paged_gqa_attention(q, cache.k[i], cache.v[i], all_slots, scale)

        return self._run(token_ids, positions, attend)


def greedy_generate(
    model: Qwen3, prompt_ids: list[int], max_new_tokens: int, eos_ids: frozenset[int] = frozenset()
) -> list[int]:
    """Append the highest-scoring next token until max_new_tokens or an EOS token.

    Returns only the new tokens (the EOS token is included if one was produced).
    """
    device = model.lm_head.device
    ids = list(prompt_ids)
    new: list[int] = []
    for _ in range(max_new_tokens):
        logits = model.forward(torch.tensor(ids, device=device))
        nxt = int(logits[-1].argmax())  # only the last position predicts the next token
        new.append(nxt)
        ids.append(nxt)
        if nxt in eos_ids:
            break
    return new


def greedy_generate_paged(
    model: Qwen3,
    prompt_ids: list[int],
    max_new_tokens: int,
    cache: KVCache,
    allocator: BlockAllocator,
    eos_ids: frozenset[int] = frozenset(),
) -> list[int]:
    """Same output as :func:`greedy_generate`, but with the paged KV cache.

    Prefill runs the whole prompt once; each decode step then runs only the newest token.
    The sequence's blocks are always given back to the allocator, even on an error.
    """
    device = cache.k.device
    table = BlockTable(allocator, cache.block_size)
    new: list[int] = []
    pending = list(prompt_ids)  # tokens whose k/v are not in the cache yet
    try:
        for _ in range(max_new_tokens):
            new_slots = table.append(len(pending))
            all_slots = [table.slot(p) for p in range(table.num_tokens)]
            logits = model.forward_paged(
                torch.tensor(pending, device=device),
                cache,
                torch.tensor(new_slots, device=device),
                torch.tensor(all_slots, device=device),
            )
            nxt = int(logits[-1].argmax())
            new.append(nxt)
            if nxt in eos_ids:
                break
            pending = [nxt]
    finally:
        table.release()
    return new
