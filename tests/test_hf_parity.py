"""TinyServe vs Hugging Face transformers on a tiny, randomly initialised Qwen3.

HF is the reference only; it is never imported by tinyserve itself. The tiny model is
saved with save_pretrained and loaded back through OUR config parser and weight loader,
so this also tests the loading path. The real-model check is scripts/compare_hf.py.
"""

from pathlib import Path
from typing import Any

import pytest
import torch

from tinyserve.config import ModelConfig
from tinyserve.model import Qwen3, greedy_generate
from tinyserve.weights import load_weights

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def pair(tmp_path_factory: pytest.TempPathFactory) -> tuple[Qwen3, Any]:
    torch.manual_seed(0)
    hf_cfg = transformers.Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,  # deliberately != hidden_size // num_heads, like the real model
        max_position_embeddings=256,
        rope_theta=1e6,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        initializer_range=0.2,  # larger than default so greedy output is not one repeated token
    )
    hf = transformers.Qwen3ForCausalLM(hf_cfg).eval()
    with torch.no_grad():  # norm weights init to 1.0; randomise so misplaced norms show up
        for name, param in hf.named_parameters():
            if "norm" in name:
                param.uniform_(0.5, 1.5)

    model_dir: Path = tmp_path_factory.mktemp("tiny_qwen3")
    hf.save_pretrained(model_dir)
    cfg = ModelConfig.from_file(model_dir / "config.json")
    ours = Qwen3(cfg, load_weights(model_dir, cfg, dtype=torch.float32))
    return ours, hf


@pytest.mark.parametrize("seed", range(5))
def test_prefill_logits_match(pair: tuple[Qwen3, Any], seed: int) -> None:
    ours, hf = pair
    ids = torch.randint(0, 128, (17,), generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        ref = hf(ids[None]).logits[0].float()
    torch.testing.assert_close(ours.forward(ids), ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("seed", range(5))
def test_greedy_tokens_identical(pair: tuple[Qwen3, Any], seed: int) -> None:
    ours, hf = pair
    prompt = torch.randint(0, 128, (9,), generator=torch.Generator().manual_seed(100 + seed))
    with torch.no_grad():
        full = hf.generate(prompt[None], max_new_tokens=24, do_sample=False, min_new_tokens=24)
    expected = full[0, prompt.shape[0] :].tolist()
    assert len(set(expected)) > 1, "degenerate reference; the test would prove little"
    assert greedy_generate(ours, prompt.tolist(), max_new_tokens=24) == expected
