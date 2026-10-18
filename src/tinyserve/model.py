"""The Qwen3 forward pass, built from the pieces in layers.py and attention.py.

M1 version: one sequence, no KV cache. Every generation step re-runs the whole
sequence through the model, which is O(n^2) work per token. That is slow on
purpose: it is the simplest correct baseline. M2 adds the paged KV cache.
"""

import torch
import torch.nn.functional as F

from tinyserve.attention import causal_gqa_attention
from tinyserve.config import ModelConfig
from tinyserve.layers import apply_rope, rms_norm, rope_cos_sin, swiglu_mlp


class Qwen3:
    def __init__(self, cfg: ModelConfig, weights: dict[str, torch.Tensor]) -> None:
        self.cfg = cfg
        self.w = weights
        # Tied embeddings: the output projection reuses the input embedding matrix.
        key = "model.embed_tokens.weight" if cfg.tie_word_embeddings else "lm_head.weight"
        self.lm_head = weights[key]

    def _layer(self, i: int, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
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
        attn = causal_gqa_attention(q, k, v, scale=cfg.head_dim**-0.5)
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

    @torch.inference_mode()
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """``[T]`` token ids of one sequence -> ``[T, vocab]`` float32 logits."""
        positions = torch.arange(token_ids.shape[0], device=token_ids.device)
        cos, sin = rope_cos_sin(self.cfg.head_dim, self.cfg.rope_theta, positions)
        x = self.w["model.embed_tokens.weight"][token_ids]  # embedding = row lookup
        for i in range(self.cfg.num_layers):
            x = self._layer(i, x, cos, sin)
        x = rms_norm(x, self.w["model.norm.weight"], self.cfg.rms_norm_eps)
        return F.linear(x, self.lm_head).float()


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
