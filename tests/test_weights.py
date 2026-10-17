from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tests.test_config import TINY
from tinyserve.config import ModelConfig
from tinyserve.weights import WeightLoadError, expected_shapes, load_weights

CFG = ModelConfig.from_dict(TINY)


def fake_checkpoint(cfg: ModelConfig) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    shapes = expected_shapes(cfg)
    return {n: torch.randn(s, generator=g).to(torch.bfloat16) for n, s in shapes.items()}


def write(tmp_path: Path, tensors: dict[str, torch.Tensor]) -> Path:
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


def test_expected_shapes_count() -> None:
    # 11 tensors per layer + embedding + final norm; lm_head is tied.
    assert len(expected_shapes(CFG)) == 11 * CFG.num_layers + 2


def test_untied_adds_lm_head() -> None:
    cfg = ModelConfig.from_dict(TINY | {"tie_word_embeddings": False})
    assert expected_shapes(cfg)["lm_head.weight"] == (cfg.vocab_size, cfg.hidden_size)


def test_loads_and_casts(tmp_path: Path) -> None:
    ckpt = fake_checkpoint(CFG)
    w = load_weights(write(tmp_path, ckpt), CFG, dtype=torch.float16)
    assert w.keys() == ckpt.keys()
    for name, t in w.items():
        assert t.dtype == torch.float16
        torch.testing.assert_close(t, ckpt[name].to(torch.float16))


def test_sharded_checkpoint(tmp_path: Path) -> None:
    ckpt = fake_checkpoint(CFG)
    names = sorted(ckpt)
    half = len(names) // 2
    save_file(
        {n: ckpt[n] for n in names[:half]}, str(tmp_path / "model-00001-of-00002.safetensors")
    )
    save_file(
        {n: ckpt[n] for n in names[half:]}, str(tmp_path / "model-00002-of-00002.safetensors")
    )
    assert load_weights(tmp_path, CFG).keys() == ckpt.keys()


def test_tied_checkpoint_may_ship_lm_head(tmp_path: Path) -> None:
    ckpt = fake_checkpoint(CFG)
    ckpt["lm_head.weight"] = ckpt["model.embed_tokens.weight"].clone()
    assert "lm_head.weight" not in load_weights(write(tmp_path, ckpt), CFG)


def test_missing_tensor(tmp_path: Path) -> None:
    ckpt = fake_checkpoint(CFG)
    del ckpt["model.layers.1.self_attn.k_norm.weight"]
    with pytest.raises(WeightLoadError, match="1 missing"):
        load_weights(write(tmp_path, ckpt), CFG)


def test_unexpected_tensor(tmp_path: Path) -> None:
    ckpt = fake_checkpoint(CFG)
    ckpt["model.layers.0.self_attn.q_proj.bias"] = torch.zeros(CFG.q_size)
    with pytest.raises(WeightLoadError, match="1 unexpected"):
        load_weights(write(tmp_path, ckpt), CFG)


def test_wrong_shape(tmp_path: Path) -> None:
    ckpt = fake_checkpoint(CFG)
    ckpt["model.layers.0.mlp.up_proj.weight"] = torch.zeros(CFG.hidden_size, CFG.intermediate_size)
    with pytest.raises(WeightLoadError, match="up_proj"):
        load_weights(write(tmp_path, ckpt), CFG)


def test_fp16_overflow_is_caught(tmp_path: Path) -> None:
    ckpt = fake_checkpoint(CFG)
    ckpt["model.norm.weight"][0] = 1e5  # fine in bf16, inf in fp16
    with pytest.raises(WeightLoadError, match="inf/nan"):
        load_weights(write(tmp_path, ckpt), CFG, dtype=torch.float16)


def test_empty_dir(tmp_path: Path) -> None:
    with pytest.raises(WeightLoadError, match=r"no \*\.safetensors"):
        load_weights(tmp_path, CFG)
