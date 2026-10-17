"""Load a Qwen3 checkpoint from safetensors into a flat ``name -> tensor`` dict.

The loader is strict: every expected tensor must be present with the exact
shape, and unknown tensors are an error. A silent mismatch here would only show
up much later as wrong tokens, which is far harder to debug.
"""

from pathlib import Path

import torch
from safetensors import safe_open

from tinyserve.config import ModelConfig


class WeightLoadError(RuntimeError):
    """The checkpoint does not match what the config says it should contain."""


def expected_shapes(cfg: ModelConfig) -> dict[str, tuple[int, ...]]:
    """Every tensor TinyServe needs, with its shape as stored on disk ([out, in] for Linear)."""
    h, inter, d = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (cfg.vocab_size, h),
        "model.norm.weight": (h,),
    }
    if not cfg.tie_word_embeddings:
        shapes["lm_head.weight"] = (cfg.vocab_size, h)
    for i in range(cfg.num_layers):
        p = f"model.layers.{i}."
        shapes |= {
            p + "input_layernorm.weight": (h,),
            p + "self_attn.q_proj.weight": (cfg.q_size, h),
            p + "self_attn.k_proj.weight": (cfg.kv_size, h),
            p + "self_attn.v_proj.weight": (cfg.kv_size, h),
            p + "self_attn.o_proj.weight": (h, cfg.q_size),
            # Qwen3-specific: RMSNorm applied per head to q and k before RoPE.
            p + "self_attn.q_norm.weight": (d,),
            p + "self_attn.k_norm.weight": (d,),
            p + "post_attention_layernorm.weight": (h,),
            p + "mlp.gate_proj.weight": (inter, h),
            p + "mlp.up_proj.weight": (inter, h),
            p + "mlp.down_proj.weight": (h, inter),
        }
    return shapes


def load_weights(
    model_dir: Path,
    cfg: ModelConfig,
    dtype: torch.dtype = torch.float16,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Read all ``*.safetensors`` files in ``model_dir``, validate, cast to ``dtype``.

    The default is float16 because the T4 (Turing) has no fast bfloat16 path.
    Casting bf16 -> fp16 can overflow (fp16 max is 65504), so we check for inf/nan.
    """
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise WeightLoadError(f"no *.safetensors files in {model_dir}")

    expected = expected_shapes(cfg)
    # A tied checkpoint may still ship lm_head.weight; it duplicates the embedding, so skip it.
    ignorable = {"lm_head.weight"} if cfg.tie_word_embeddings else set()

    weights: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as f:
            for name in f.keys():  # noqa: SIM118  (safe_open is not a dict)
                if name in ignorable:
                    continue
                if name not in expected:
                    unexpected.append(name)
                    continue
                weights[name] = f.get_tensor(name)

    problems: list[str] = []
    missing = sorted(expected.keys() - weights.keys())
    if missing:
        problems.append(f"{len(missing)} missing, e.g. {missing[:3]}")
    if unexpected:
        problems.append(f"{len(unexpected)} unexpected, e.g. {sorted(unexpected)[:3]}")
    for name, t in weights.items():
        if tuple(t.shape) != expected[name]:
            problems.append(f"{name}: shape {tuple(t.shape)}, expected {expected[name]}")
    if problems:
        raise WeightLoadError("checkpoint does not match config:\n  " + "\n  ".join(problems))

    out: dict[str, torch.Tensor] = {}
    for name, t in weights.items():
        cast = t.to(device=device, dtype=dtype)
        if not torch.isfinite(cast).all():
            raise WeightLoadError(f"{name} has inf/nan after casting to {dtype}")
        out[name] = cast
    return out
