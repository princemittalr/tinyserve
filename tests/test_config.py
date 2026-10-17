from pathlib import Path
from typing import Any

import pytest

from tinyserve.config import ModelConfig, UnsupportedConfigError

# Same keys and layout as Qwen3-0.6B's config.json, with tiny sizes.
TINY: dict[str, Any] = {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "head_dim": 8,
    "hidden_act": "silu",
    "hidden_size": 16,
    "intermediate_size": 32,
    "max_position_embeddings": 128,
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-6,
    "rope_scaling": None,
    "rope_theta": 1000000,
    "tie_word_embeddings": True,
    "vocab_size": 64,
}


def test_parses_tiny_config() -> None:
    cfg = ModelConfig.from_dict(TINY)
    assert cfg.num_layers == 2
    assert cfg.q_size == 4 * 8
    assert cfg.kv_size == 2 * 8
    assert cfg.tie_word_embeddings


def test_head_dim_is_read_not_derived() -> None:
    # hidden_size // num_heads would be 4 here; Qwen3 says 8 explicitly.
    assert ModelConfig.from_dict(TINY).head_dim == 8


def test_from_file(tmp_path: Path) -> None:
    import json

    path = tmp_path / "config.json"
    path.write_text(json.dumps(TINY), encoding="utf-8")
    assert ModelConfig.from_file(path) == ModelConfig.from_dict(TINY)


@pytest.mark.parametrize(
    "override",
    [
        {"architectures": ["LlamaForCausalLM"]},
        {"hidden_act": "gelu"},
        {"attention_bias": True},
        {"rope_scaling": {"type": "yarn", "factor": 4.0}},
        {"num_key_value_heads": 3},
    ],
)
def test_rejects_unsupported(override: dict[str, Any]) -> None:
    with pytest.raises(UnsupportedConfigError):
        ModelConfig.from_dict(TINY | override)
