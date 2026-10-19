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
        With ``prefill_attention="batched"`` (M8), the other sequences attend in one
        padded call per layer (``batched_prefill_attention``) instead of one call each;
        with ``"triton"`` (M9), in one launch of the varlen kernel in triton_prefill.py,
        which reads their keys and values straight from the cache blocks.
    ``forward_decode`` (M8): a decode step with fixed shapes and no host syncs, from
        buffers filled in advance (``DecodeInputs``), so a CUDA graph can capture it
        (cuda_graphs.py). Same layers, same kernel; only how the inputs arrive differs.

With ``fused_kernels=True`` (M9, triton_fused.py) the paged paths run the hidden-state
RMSNorms as one Triton launch each, and the q/k head norms, RoPE and the k/v cache write
as one launch per layer, instead of several PyTorch kernels each.

Since M10 every path multiplies by q/k/v as one matrix and by gate/up as one
(``weights.merge_projections``), and the output head always gets at least
``HEAD_MIN_ROWS`` rows: with fewer, cuBLAS runs a matrix-vector kernel that read the
head's 311 MB at 138 GB/s on the T4, against 239 GB/s for 8 rows (M10 run 1).
"""

from collections.abc import Callable
from typing import Any, Literal

import torch
import torch.nn.functional as F

from tinyserve.attention import (
    batched_prefill_attention,
    causal_gqa_attention,
    paged_gqa_attention,
    prefill_plan,
)
from tinyserve.batch import Batch, DecodeInputs
from tinyserve.block_allocator import BlockAllocator
from tinyserve.block_table import BlockTable
from tinyserve.config import ModelConfig
from tinyserve.kv_cache import KVCache
from tinyserve.layers import apply_rope, rms_norm, rope_cos_sin
from tinyserve.weights import GATE_UP, QKV, merge_projections

# decode_attention choices: per-sequence PyTorch, the M4 kernel, the M8 grouped kernel,
# the grouped kernel with split-K for small batches (M9).
DECODE_ATTENTIONS = ("torch", "triton", "triton_grouped", "triton_split")
# prefill_attention choices: per-sequence PyTorch, one padded SDPA call (M8), the M9 kernel.
PREFILL_ATTENTIONS = ("per_seq", "batched", "triton")
# (layer index, q [T, Hq, D], k [T, Hkv, D], v [T, Hkv, D]) -> attention output [T, Hq, D].
# k and v are None when the fused kernel (M9) already wrote them into the cache.
AttentionFn = Callable[[int, torch.Tensor, torch.Tensor | None, torch.Tensor | None], torch.Tensor]
# paged_decode_attention's signature (triton_attention.py); kept here so the model does
# not import Triton unless asked to.
DecodeFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, int],
    torch.Tensor,
]
# The output head runs on at least this many rows (zero rows are added, then dropped).
HEAD_MIN_ROWS = 8


class Qwen3:
    def __init__(
        self,
        cfg: ModelConfig,
        weights: dict[str, torch.Tensor],
        decode_attention: Literal["torch", "triton", "triton_grouped", "triton_split"] = "torch",
        prefill_attention: Literal["per_seq", "batched", "triton"] = "per_seq",
        fused_kernels: bool = False,
    ) -> None:
        """``decode_attention`` picks how ``forward_batch`` attends for one-token sequences:
        "torch" runs ``paged_gqa_attention`` per sequence, "triton" runs the M4 kernel once
        for all of them, "triton_grouped" the M8 kernel that shares k/v loads across a
        head group, "triton_split" (M9) that kernel with split-K when the batch is small.
        Triton is imported only when chosen (it has no Windows wheels).
        ``prefill_attention`` picks how the other sequences attend: "per_seq" runs
        ``paged_gqa_attention`` for each, "batched" (M8) runs ``batched_prefill_attention``
        once for all of them, "triton" (M9) the varlen prefill kernel once for all.
        ``fused_kernels`` (M9) turns on the fused norm / RoPE / cache-write kernels for
        ``forward_batch`` and ``forward_decode``.
        ``weights`` gets the merged q/k/v and gate/up matrices added (M10,
        ``merge_projections``); its other entries keep their values."""
        self.cfg = cfg
        merge_projections(weights, cfg)
        self.w = weights
        self.decode_kernel: DecodeFn | None = None
        self.decode_launch: DecodeFn | None = None  # the same kernel without its checks
        if decode_attention == "triton":
            from tinyserve.triton_attention import launch_paged_decode, paged_decode_attention

            self.decode_kernel = paged_decode_attention
            self.decode_launch = launch_paged_decode
        elif decode_attention == "triton_grouped":  # M8: one program per kv head
            from tinyserve import triton_attention as ta

            self.decode_kernel = ta.paged_decode_attention_grouped
            self.decode_launch = ta.launch_paged_decode_grouped
        elif decode_attention == "triton_split":  # M9: grouped, split-K at small batches
            from tinyserve import triton_attention as ta

            self.decode_kernel = ta.paged_decode_attention_split
            self.decode_launch = ta.launch_paged_decode_split
        elif decode_attention != "torch":
            raise ValueError(
                f"decode_attention must be one of {DECODE_ATTENTIONS}: {decode_attention!r}"
            )
        if prefill_attention not in PREFILL_ATTENTIONS:
            raise ValueError(
                f"prefill_attention must be one of {PREFILL_ATTENTIONS}: {prefill_attention!r}"
            )
        self.batched_prefill = prefill_attention == "batched"
        self.prefill_kernel: Callable[..., torch.Tensor] | None = None
        if prefill_attention == "triton":
            from tinyserve.triton_prefill import paged_prefill_attention

            self.prefill_kernel = paged_prefill_attention
        self.fused: Any = None  # the triton_fused module, imported only when asked for
        if fused_kernels:
            from tinyserve import triton_fused

            self.fused = triton_fused
        # Tied embeddings: the output projection reuses the input embedding matrix.
        key = "model.embed_tokens.weight" if cfg.tie_word_embeddings else "lm_head.weight"
        self.lm_head = weights[key]

    def _norm(self, x: torch.Tensor, weight: torch.Tensor, fused: bool) -> torch.Tensor:
        if fused:
            return self.fused.rms_norm_rows(x, weight, self.cfg.rms_norm_eps)  # type: ignore[no-any-return]
        return rms_norm(x, weight, self.cfg.rms_norm_eps)

    def _layer(
        self,
        i: int,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attend: AttentionFn,
        store: tuple[KVCache, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """One decoder layer. With ``store`` = (cache, slots) and the fused kernels on,
        the new k/v go into the cache here, and ``attend`` gets k = v = None."""
        cfg, p = self.cfg, f"model.layers.{i}."
        w = self.w
        t = x.shape[0]
        fused = store is not None and self.fused is not None

        # Attention block: pre-norm, then a residual add.
        h = self._norm(x, w[p + "input_layernorm.weight"], fused)
        # One GEMM for q, k and v (M10); the three are column ranges of its output.
        qkv = F.linear(h, w[p + QKV])
        q, k, v = torch.split(qkv, [cfg.q_size, cfg.kv_size, cfg.kv_size], dim=1)
        q = q.view(t, cfg.num_heads, cfg.head_dim)
        k = k.view(t, cfg.num_kv_heads, cfg.head_dim)
        v = v.view(t, cfg.num_kv_heads, cfg.head_dim)
        qn, kn = w[p + "self_attn.q_norm.weight"], w[p + "self_attn.k_norm.weight"]
        if fused and store is not None:  # M9: norms, RoPE and the cache write in one launch
            cache, slots = store
            q = self.fused.qk_norm_rope_store(
                q, k, v, qn, kn, cos, sin, slots, cache.k[i], cache.v[i], cfg.rms_norm_eps
            )
            attn = attend(i, q, None, None)
        else:
            # Qwen3-specific: normalise each head of q and k BEFORE applying RoPE.
            q, k = rms_norm(q, qn, cfg.rms_norm_eps), rms_norm(k, kn, cfg.rms_norm_eps)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            attn = attend(i, q, k, v)
        x = x + F.linear(attn.reshape(t, cfg.q_size), w[p + "self_attn.o_proj.weight"])

        # MLP block: pre-norm, then a residual add.
        h = self._norm(x, w[p + "post_attention_layernorm.weight"], fused)
        # swiglu_mlp (layers.py) with gate and up from one GEMM (M10).
        gate, up = F.linear(h, w[p + GATE_UP]).chunk(2, dim=1)
        return x + F.linear(F.silu(gate) * up, w[p + "mlp.down_proj.weight"])

    def _run(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        attend: AttentionFn,
        rows: torch.Tensor | None = None,
        store: tuple[KVCache, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Logits for every token, or only for ``rows`` if given. ``store`` = (cache, the
        new tokens' slots) lets the fused kernels write k/v (see ``_layer``)."""
        cos, sin = rope_cos_sin(self.cfg.head_dim, self.cfg.rope_theta, positions)
        x = self.w["model.embed_tokens.weight"][token_ids]  # embedding = row lookup
        for i in range(self.cfg.num_layers):
            x = self._layer(i, x, cos, sin, attend, store)
        if rows is not None:
            # The output head is [vocab, hidden] = 151,936 x 1,024 for Qwen3-0.6B: the
            # largest matmul in the model. Skip it for rows nobody samples from.
            x = x[rows]
        x = self._norm(x, self.w["model.norm.weight"], store is not None and bool(self.fused))
        return self.head(x)

    def head(self, x: torch.Tensor) -> torch.Tensor:
        """``[N, hidden]`` -> ``[N, vocab]`` float32 logits, padded to ``HEAD_MIN_ROWS``
        rows for the matrix multiply (M10)."""
        n = x.shape[0]
        if n < HEAD_MIN_ROWS:
            x = F.pad(x, (0, 0, 0, HEAD_MIN_ROWS - n))
        return F.linear(x, self.lm_head)[:n].float()

    @torch.inference_mode()
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """``[T]`` token ids of one sequence -> ``[T, vocab]`` float32 logits. No cache."""
        positions = torch.arange(token_ids.shape[0], device=token_ids.device)
        scale = self.cfg.head_dim**-0.5

        def attend(
            i: int, q: torch.Tensor, k: torch.Tensor | None, v: torch.Tensor | None
        ) -> torch.Tensor:
            assert k is not None and v is not None  # no cache, so never fused
            return causal_gqa_attention(q, k, v, scale)

        return self._run(token_ids, positions, attend)

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

        def attend(
            i: int, q: torch.Tensor, k: torch.Tensor | None, v: torch.Tensor | None
        ) -> torch.Tensor:
            assert k is not None and v is not None  # the M2 path is never fused
            cache.write(i, new_slots, k, v)
            return paged_gqa_attention(q, cache.k[i], cache.v[i], all_slots, scale)

        return self._run(token_ids, positions, attend)

    @torch.inference_mode()
    def forward_batch(self, batch: Batch, cache: KVCache) -> torch.Tensor:
        """One step for B sequences: ``[B, vocab]`` float32 logits of each one's last token.

        Every new token's k/v is written to the cache in one scatter per layer. With the
        Triton kernel, all sequences that bring one token (decode) attend in one launch;
        the rest (prefill) attend one by one in PyTorch, on their slice of the queries, or
        all together in one call per layer with ``prefill_attention="batched"``.
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

        plan = None
        if self.batched_prefill and rest:  # built once per step, used by every layer
            plan = prefill_plan(
                [bounds[b] for b in rest],
                [batch.query_lens[b] for b in rest],
                [batch.seq_slots[b] for b in rest],
            )
        prefill = self.prefill_kernel if rest else None
        if prefill is not None:  # the varlen kernel's per-sequence inputs, once per step
            dev = batch.token_ids.device
            sel = torch.tensor(rest, device=dev)

            def ints(xs: list[int]) -> torch.Tensor:
                return torch.tensor(xs, dtype=torch.int32, device=dev)

            starts, qlens = (
                ints([bounds[b] for b in rest]),
                ints([batch.query_lens[b] for b in rest]),
            )
            p_tables, p_lens = batch.block_tables[sel], batch.context_lens[sel]
            max_q = max(batch.query_lens[b] for b in rest)

        def attend(
            i: int, q: torch.Tensor, k: torch.Tensor | None, v: torch.Tensor | None
        ) -> torch.Tensor:
            if k is not None and v is not None:  # else the fused kernel wrote them
                cache.write(i, batch.new_slots, k, v)
            ki, vi, bs = cache.k[i], cache.v[i], cache.block_size
            if kernel and not rest:  # all decode: q's rows are the B sequences, in order
                return kernel(q, ki, vi, tables, lens, scale, bs)
            out = torch.empty_like(q)
            if kernel and one:
                out[rows] = kernel(q[rows], ki, vi, tables, lens, scale, bs)
            if prefill is not None:  # writes the rest's rows of out in place
                return prefill(q, ki, vi, p_tables, starts, qlens, p_lens, scale, bs, max_q, out)
            if plan is not None:
                res = batched_prefill_attention(q, ki, vi, plan, scale)
                out[plan.rows[plan.real]] = res[plan.real]
                return out
            for b in rest:
                s, e = bounds[b], bounds[b + 1]
                out[s:e] = paged_gqa_attention(q[s:e], ki, vi, batch.seq_slots[b], scale)
            return out

        return self._run(
            batch.token_ids,
            batch.positions,
            attend,
            rows=batch.last_rows,
            store=(cache, batch.new_slots),
        )

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
            raise ValueError('forward_decode needs a Triton decode kernel ("triton..." attention)')
        scale = self.cfg.head_dim**-0.5

        def attend(
            i: int, q: torch.Tensor, k: torch.Tensor | None, v: torch.Tensor | None
        ) -> torch.Tensor:
            if k is not None and v is not None:  # else the fused kernel wrote them
                cache.write(i, inputs.slots, k, v)
            ki, vi, bs = cache.k[i], cache.v[i], cache.block_size
            return launch(q, ki, vi, inputs.block_tables, inputs.context_lens, scale, bs)

        return self._run(inputs.token_ids, inputs.positions, attend, store=(cache, inputs.slots))


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
