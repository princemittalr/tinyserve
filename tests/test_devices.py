"""Every tensor a forward pass creates must live on the model's device.

A tensor created without ``device=`` lands on the CPU. That is invisible in CPU-only
tests and crashes on the GPU ("Expected all tensors to be on the same device"); the
first T4 run hit exactly this in ``rope_cos_sin``. PyTorch's "meta" device stores shapes
only and refuses to mix with CPU tensors, so running the model on it catches a missing
``device=`` on a laptop. The ``gpu`` test checks the real thing on a T4.
"""

import pytest
import torch

from tests.test_model import CFG, tiny_model
from tinyserve.batch import build_batch
from tinyserve.block_allocator import BlockAllocator
from tinyserve.block_table import BlockTable
from tinyserve.engine import Engine
from tinyserve.kv_cache import KVCache
from tinyserve.layers import rope_cos_sin
from tinyserve.model import Qwen3, greedy_generate


def on(device: str) -> tuple[Qwen3, KVCache]:
    m = tiny_model()
    m = Qwen3(CFG, {name: w.to(device) for name, w in m.w.items()})
    return m, KVCache(CFG, 16, 4, dtype=torch.float32, device=device)


def test_rope_tables_follow_the_positions_device() -> None:
    cos, sin = rope_cos_sin(8, 1e6, torch.arange(5, device="meta"))
    assert cos.device.type == sin.device.type == "meta"


def test_forward_paths_create_no_cpu_tensors() -> None:
    m, cache = on("meta")
    assert m.forward(torch.tensor([1, 2, 3], device="meta")).device.type == "meta"

    alloc = BlockAllocator(cache.num_blocks)
    a, b = BlockTable(alloc, cache.block_size), BlockTable(alloc, cache.block_size)
    a.append(3)
    b.append(1)
    batch = build_batch([([1, 2, 3], a), ([4], b)], device="meta")
    logits = m.forward_batch(batch, cache)
    assert logits.device.type == "meta"
    assert logits.shape == (2, CFG.vocab_size)


@pytest.mark.gpu
def test_engine_on_cuda_matches_cpu_greedy() -> None:
    m, cache = on("cuda")
    prompts = [[1], [5, 9, 2], [3, 8, 11, 2, 7, 7, 1]]
    out = Engine(m, cache).generate(prompts, 8)
    cpu = tiny_model()
    assert out == [greedy_generate(cpu, p, 8) for p in prompts]
