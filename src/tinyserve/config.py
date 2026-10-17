"""Model hyperparameters, parsed from a Hugging Face ``config.json``.

We only read the JSON; no Hugging Face code is involved. Anything we don't
implement (attention bias, RoPE scaling, other activations) is rejected loudly
instead of being silently ignored.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class UnsupportedConfigError(ValueError):
    """The checkpoint uses a feature TinyServe does not implement."""


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool

    @property
    def q_size(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_kv_heads * self.head_dim

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        if "Qwen3ForCausalLM" not in d.get("architectures", []):
            raise UnsupportedConfigError(f"expected Qwen3ForCausalLM, got {d.get('architectures')}")
        if d.get("hidden_act", "silu") != "silu":
            raise UnsupportedConfigError(
                f"hidden_act={d['hidden_act']!r}, only 'silu' is supported"
            )
        if d.get("attention_bias", False):
            raise UnsupportedConfigError("attention_bias=True is not supported")
        if d.get("rope_scaling") is not None:
            raise UnsupportedConfigError(f"rope_scaling={d['rope_scaling']!r} is not supported")

        cfg = cls(
            vocab_size=int(d["vocab_size"]),
            hidden_size=int(d["hidden_size"]),
            intermediate_size=int(d["intermediate_size"]),
            num_layers=int(d["num_hidden_layers"]),
            num_heads=int(d["num_attention_heads"]),
            num_kv_heads=int(d["num_key_value_heads"]),
            # Qwen3 sets head_dim explicitly; it is NOT hidden_size // num_heads in general.
            head_dim=int(d.get("head_dim", d["hidden_size"] // d["num_attention_heads"])),
            rms_norm_eps=float(d["rms_norm_eps"]),
            rope_theta=float(d["rope_theta"]),
            max_position_embeddings=int(d["max_position_embeddings"]),
            tie_word_embeddings=bool(d.get("tie_word_embeddings", False)),
        )
        if cfg.num_heads % cfg.num_kv_heads != 0:
            raise UnsupportedConfigError(
                f"num_heads={cfg.num_heads} is not a multiple of num_kv_heads={cfg.num_kv_heads}"
            )
        return cfg

    @classmethod
    def from_file(cls, path: Path) -> "ModelConfig":
        with path.open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
