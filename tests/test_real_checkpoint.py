"""Checks against the real Qwen3-0.6B files. Skipped unless TINYSERVE_MODEL_DIR is set.

Run ``python scripts/inspect_weights.py`` once to download the model, then:
``$env:TINYSERVE_MODEL_DIR = "checkpoints/Qwen3-0.6B"; pytest tests/test_real_checkpoint.py``
"""

import os
from pathlib import Path

import pytest
import torch

from tinyserve.config import ModelConfig
from tinyserve.weights import load_weights

MODEL_DIR = os.environ.get("TINYSERVE_MODEL_DIR")
pytestmark = pytest.mark.skipif(MODEL_DIR is None, reason="TINYSERVE_MODEL_DIR not set")


@pytest.fixture(scope="module")
def cfg() -> ModelConfig:
    assert MODEL_DIR is not None
    return ModelConfig.from_file(Path(MODEL_DIR) / "config.json")


def test_config_matches_qwen3_0_6b(cfg: ModelConfig) -> None:
    assert (cfg.num_layers, cfg.hidden_size, cfg.intermediate_size) == (28, 1024, 3072)
    assert (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim) == (16, 8, 128)
    assert cfg.vocab_size == 151936
    assert cfg.tie_word_embeddings


def test_loads_in_fp16(cfg: ModelConfig) -> None:
    assert MODEL_DIR is not None
    w = load_weights(Path(MODEL_DIR), cfg, dtype=torch.float16)
    assert len(w) == 11 * cfg.num_layers + 2
