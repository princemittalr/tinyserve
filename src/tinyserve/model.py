"""The Qwen3 forward pass, built from the pieces in layers.py and attention.py.

Two paths share the same layers and differ only in attention:
    ``forward`` (M1): no KV cache. Every generation step re-runs the whole sequence,
        O(n^2) work per token. Kept as the simplest correct baseline.
    ``forward_paged`` (M2): runs only the NEW tokens. Their keys and values are written
        into the paged KV cache, and attention reads the whole history back from it.
    ``forward_batch`` (M3): ``forward_paged`` for many sequences in one pass. Their new
        tokens are flattened into one batch (see batch.py); only attention is per sequence.
        With ``decode_attention="triton"`` (M4), every sequence that brings one token is
        handled by one Triton kernel launch per layer instead of one PyTorch call each.
    ``forward_decode`` (M8): a decode step with fixed shapes and no host syncs, from
        buffers filled in advance (``DecodeInputs``), so a CUDA graph can capture it
        (cuda_graphs.py). Same layers, same kernel; only how the inputs arrive differs.
"""

from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F

from tinyserve.attention import causal_gqa_attention, paged_gqa_attention
from tinyserve.batch import Batch, DecodeInputs
from tinyserve.block_allocator import BlockAllocator
from tinyserve.block_table import BlockTable
from tinyserve.config import ModelConfig
from tinyserve.kv_cache import KVCache
from tinyserve.layers import apply_rope, rms_norm, rope_cos_sin, swiglu_mlp

# (layer index, q [T, Hq, D], k [T, Hkv, D], v [T, Hkv, D]) -> attention output [T, Hq, D]
AttentionFn = Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
# paged_decode_attention's signature (triton_attention.py); kept here so the model does
# not import Triton unless asked to.
DecodeFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, int],
    torch.Tensor,
]


class Qwen3:
    def __init__(
        self,
        cfg: ModelConfig,
        weights: dict[str, torch.Tensor],
        decode_attention: Literal["torch", "triton"] = "torch",
    ) -> None:
        """``decode_attention`` picks how ``forward_batch`` attends for one-token sequences:
        "torch" runs ``paged_gqa_attention`` per sequence, "triton" runs the M4 kernel once
        for all of them. Triton is imported only when chosen (it has no Windows wheels)."""
        self.cfg = cfg
        self.w = weights
        self.decode_kernel: DecodeFn | None = None
        self.decode_launch: DecodeFn | None = None  # the same kernel without its checks
        if decode_attention == "triton":
            from tinyserve.triton_attention import launch_paged_decode, paged_decode_attention

            self.decode_kernel = paged_decode_attention
            self.decode_launch = launch_paged_decode
        elif decode_attention != "torch":
            raise ValueError(f"decode_attention must be 'torch' or 'triton': {decode_attention!r}")
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
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        attend: AttentionFn,
        rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Logits for every token, or only for ``rows`` if given."""
        cos, sin = rope_cos_sin(self.cfg.head_dim, self.cfg.rope_theta, positions)
        x = self.w["model.embed_tokens.weight"][token_ids]  # embedding = row lookup
        for i in range(self.cfg.num_layers):
            x = self._layer(i, x, cos, sin, attend)
        if rows is not None:
            # The output head is [vocab, hidden] = 151,936 x 1,024 for Qwen3-0.6B: the
            # largest matmul in the model. Skip it for rows nobody samples from.
            x = x[rows]
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

    @torch.inference_mode()
    def forward_batch(self, batch: Batch, cache: KVCache) -> torch.Tensor:
        """One step for B sequences: ``[B, vocab]`` float32 logits of each one's last token.

        Every new token's k/v is written to the cache in one scatter per layer. With the
        Triton kernel, all sequences that bring one token (decode) attend in one launch;
        the rest (prefill) attend one by one in PyTorch, on their slice of the queries.
        Row b of the result predicts sequence b's next token.
        """
        scale = self.cfg.head_dim**-0.5
        bounds = [0, *torch.tensor(batch.query_lens).cumsum(0).tolist()]
        kernel = self.decode_kernel
        one = [b for b, n in enumerate(batch.query_lens) if kernel and n == 1]
        rest = [b for b, n in enumerate(batch.query_lens) if not (kernel and n == 1)]
        rows, tables, lens = batch.last_rows, batch.block_tables, batch.context_lens
        if one and rest:  # mixed batch: pick out the one-token sequences
            idx = torch.tensor(one, device=batch.token_ids.device)
            # A one-token sequence's only query row is its last row.
            rows, tables, lens = rows[idx], tables[idx], lens[idx]

        def attend(i: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            cache.write(i, batch.new_slots, k, v)
            ki, vi, bs = cache.k[i], cache.v[i], cache.block_size
            if kernel and not rest:  # all decode: q's rows are the B sequences, in order
                return kernel(q, ki, vi, tables, lens, scale, bs)
            out = torch.empty_like(q)
            if kernel and one:
                out[rows] = kernel(q[rows], ki, vi, tables, lens, scale, bs)
            for b in rest:
                s, e = bounds[b], bounds[b + 1]
                out[s:e] = paged_gqa_attention(q[s:e], ki, vi, batch.seq_slots[b], scale)
            return out

        return self._run(batch.token_ids, batch.positions, attend, rows=batch.last_rows)

    @torch.inference_mode()
    def forward_decode(self, inputs: DecodeInputs, cache: KVCache) -> torch.Tensor:
        """One decode step for every row of ``inputs``: ``[N, vocab]`` float32 logits.

        Row r predicts the next token of the sequence in row r; padding rows give logits
        nobody reads. Nothing here waits for the GPU or depends on a Python value that
        changes between steps: the shapes are fixed by the buffers, and every per-step value
        (tokens, positions, slots, block tables, lengths) is read from them on the device.
        That is what lets a CUDA graph record this once and replay it (cuda_graphs.py).
        Needs ``decode_attention="triton"``; ``fill_decode_inputs`` guarantees what the
        kernel's skipped checks would check.
        """
        launch = self.decode_launch
        if launch is None:
            raise ValueError('forward_decode needs decode_attention="triton"')
        scale = self.cfg.head_dim**-0.5

        def attend(i: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            cache.write(i, inputs.slots, k, v)
            ki, vi, bs = cache.k[i], cache.v[i], cache.block_size
            return launch(q, ki, vi, inputs.block_tables, inputs.context_lens, scale, bs)

        return self._run(inputs.token_ids, inputs.positions, attend)


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
