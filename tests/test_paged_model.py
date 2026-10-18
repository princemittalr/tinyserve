"""The model with the paged KV cache must produce the same logits and tokens as M1's
no-cache path, on a tiny random Qwen3. The real-model check is scripts/compare_hf.py."""

import pytest
import torch

from tests.test_model import CFG, tiny_model
from tinyserve.block_allocator import BlockAllocator, OutOfBlocksError
from tinyserve.block_table import BlockTable
from tinyserve.kv_cache import KVCache
from tinyserve.model import greedy_generate, greedy_generate_paged

PROMPTS = [[1], [5, 9, 2], [3, 8, 11, 2, 7, 7, 1, 4, 9, 30, 12, 5, 6, 2, 8, 1, 3]]


def make_cache(num_blocks: int = 32, block_size: int = 4) -> tuple[KVCache, BlockAllocator]:
    return (
        KVCache(CFG, num_blocks, block_size, dtype=torch.float32),
        BlockAllocator(num_blocks),
    )


def test_prefill_logits_match_no_cache_forward() -> None:
    m = tiny_model()
    cache, alloc = make_cache()
    ids = torch.tensor(PROMPTS[2])
    table = BlockTable(alloc, cache.block_size)
    slots = torch.tensor(table.append(len(ids)))
    paged = m.forward_paged(ids, cache, slots, slots)
    torch.testing.assert_close(paged, m.forward(ids), atol=1e-5, rtol=1e-5)


def test_each_decode_step_matches_no_cache_forward() -> None:
    m = tiny_model()
    cache, alloc = make_cache(block_size=2)
    ids = [5, 9, 2, 7, 1, 4, 13, 6]
    table = BlockTable(alloc, cache.block_size)
    s = torch.tensor(table.append(3))
    m.forward_paged(torch.tensor(ids[:3]), cache, s, s)
    for t in range(3, len(ids)):
        new = torch.tensor(table.append(1))
        every = torch.tensor([table.slot(p) for p in range(t + 1)])
        got = m.forward_paged(torch.tensor(ids[t : t + 1]), cache, new, every)
        expected = m.forward(torch.tensor(ids[: t + 1]))[-1:]
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("block_size", [1, 4, 16])
@pytest.mark.parametrize("prompt", PROMPTS)
def test_greedy_tokens_identical_to_m1(block_size: int, prompt: list[int]) -> None:
    m = tiny_model()
    cache, alloc = make_cache(num_blocks=64, block_size=block_size)
    expected = greedy_generate(m, prompt, max_new_tokens=12)
    assert greedy_generate_paged(m, prompt, 12, cache, alloc) == expected


def test_stops_at_eos_like_m1() -> None:
    m = tiny_model()
    cache, alloc = make_cache()
    first = greedy_generate(m, [1, 2, 3], max_new_tokens=1)[0]
    out = greedy_generate_paged(m, [1, 2, 3], 10, cache, alloc, eos_ids=frozenset({first}))
    assert out == [first]


def test_blocks_are_returned_after_generation() -> None:
    m = tiny_model()
    cache, alloc = make_cache(num_blocks=8)
    greedy_generate_paged(m, PROMPTS[2], 6, cache, alloc)
    assert alloc.num_free == 8


def test_blocks_are_returned_when_the_cache_runs_out() -> None:
    m = tiny_model()
    cache, alloc = make_cache(num_blocks=3, block_size=4)  # room for 12 tokens
    with pytest.raises(OutOfBlocksError):
        greedy_generate_paged(m, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 8, cache, alloc)
    assert alloc.num_free == 3


def test_reused_blocks_with_stale_data_do_not_leak_into_the_next_sequence() -> None:
    # The second sequence gets the first one's freed blocks, still full of its k/v.
    m = tiny_model()
    cache, alloc = make_cache(num_blocks=16)
    greedy_generate_paged(m, PROMPTS[2], 10, cache, alloc)
    out = greedy_generate_paged(m, PROMPTS[1], 10, cache, alloc)
    assert out == greedy_generate(m, PROMPTS[1], 10)
