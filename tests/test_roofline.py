"""The roofline model (M7): byte and FLOP counts checked against the real Qwen3-0.6B
sizes and against the tiny test checkpoint's actual tensors."""

import math

import pytest

from tests.test_config import TINY
from tests.test_model import CFG
from tinyserve.config import ModelConfig
from tinyserve.roofline import (
    T4,
    GPUSpec,
    StepCost,
    decode_step_cost,
    kv_bytes_per_token,
    layer_matmul_params,
    weight_bytes,
)
from tinyserve.weights import expected_shapes

# Qwen3-0.6B's config.json sizes (the real file is checked in test_real_checkpoint.py).
QWEN3_06B = ModelConfig.from_dict(
    {
        **TINY,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151_936,
        "max_position_embeddings": 40_960,
    }
)


def test_weight_bytes_counts_every_tensor_of_the_checkpoint_once() -> None:
    # Tied embeddings: the output head is the embedding matrix, stored (and read) once.
    numel = sum(math.prod(s) for s in expected_shapes(CFG).values())
    assert weight_bytes(CFG, 1) == numel
    assert weight_bytes(CFG, 2) == 2 * numel


def test_qwen3_06b_sizes_match_the_real_checkpoint() -> None:
    # M1 step 2 counted 596,049,920 parameters, 440,467,456 without the embedding.
    assert weight_bytes(QWEN3_06B, 1) == 596_049_920
    assert 28 * layer_matmul_params(QWEN3_06B) == 440_401_920  # the rest are norms
    assert kv_bytes_per_token(QWEN3_06B, 2) == 2 * 28 * 8 * 128 * 2  # 112 KiB


def test_t4_ridge_point() -> None:
    assert T4.ridge(2) == pytest.approx(65e12 / 320e9)
    assert T4.ridge(4) == pytest.approx(8.1e12 / 320e9)


def test_floor_is_the_slower_of_memory_and_compute() -> None:
    gpu = GPUSpec("toy", mem_gbps=1.0, fp16_tflops=1e-3, fp32_tflops=1e-3)  # 1e9 FLOP/s
    memory_heavy = StepCost(flops=1e6, weight_bytes=2e9, kv_bytes=0.0)
    assert memory_heavy.floor_ms(gpu, 2) == pytest.approx(2000.0)
    assert memory_heavy.memory_bound(gpu, 2)
    compute_heavy = StepCost(flops=5e9, weight_bytes=1e9, kv_bytes=0.0)
    assert compute_heavy.floor_ms(gpu, 2) == pytest.approx(5000.0)
    assert not compute_heavy.memory_bound(gpu, 2)


def test_decode_step_flops_and_bytes() -> None:
    c = decode_step_cost(QWEN3_06B, num_seqs=1, context_tokens=0, elem_bytes=2)
    # One token through every weight: 2 FLOPs per parameter (the embedding lookup is free,
    # but the tied head multiplies by the same matrix).
    assert c.flops == 2 * 596_049_920 - 2 * (28 * (2 * 1024 + 2 * 128) + 1024)
    assert c.kv_bytes == 0
    two = decode_step_cost(QWEN3_06B, num_seqs=2, context_tokens=300, elem_bytes=2)
    assert two.kv_bytes == 300 * kv_bytes_per_token(QWEN3_06B, 2)
    attn = 28 * 4 * 16 * 128 * 300
    assert two.flops == pytest.approx(2 * c.flops + attn)


def test_decode_is_memory_bound_on_a_t4_at_every_benchmarked_batch() -> None:
    for b in (1, 8, 32, 64):
        c = decode_step_cost(QWEN3_06B, num_seqs=b, context_tokens=b * 192, elem_bytes=2)
        assert c.memory_bound(T4, 2)
        # 1.19 GB of weights at 320 GB/s is about 3.7 ms before any k/v is read.
        assert c.floor_ms(T4, 2) > 3.7
    huge = decode_step_cost(QWEN3_06B, num_seqs=1024, context_tokens=0, elem_bytes=2)
    assert not huge.memory_bound(T4, 2)
