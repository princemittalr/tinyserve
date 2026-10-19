"""The Triton paged decode-attention kernel must match the PyTorch reference.

The reference is ``paged_decode_attention_ref``: M2's ``paged_gqa_attention`` run one
sequence at a time, itself tested against M1's attention. On a GPU these tests run the
compiled kernel; without one they run it in Triton's interpreter (see conftest.py).

Tolerance policy (decided in M4 step 2). The kernel and the reference both compute
scores, softmax and the weighted sum in float32 and round to the input dtype once, at the
end. They differ only in summation order, so:
    float32 inputs: atol = rtol = 1e-5.
    float16 inputs: atol = rtol = 1e-3, about one fp16 rounding step for values near 1.

Every cache slot no sequence owns is filled with NaN, and block-table padding points at
a NaN block. A kernel that reads one slot too many returns NaN and fails the test.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("triton", reason="Triton is not installed (no Windows wheels)")

from tinyserve.attention import paged_decode_attention_ref, paged_gqa_attention
from tinyserve.triton_attention import (
    MAX_SPLITS,
    launch_paged_decode_split,
    num_splits_for,
    paged_decode_attention,
    paged_decode_attention_grouped,
    paged_decode_attention_split,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOL = {torch.float32: 1e-5, torch.float16: 1e-3}
# The M4 kernel (one program per query head), the M8 grouped one (one per kv head) and
# the M9 split-K one (grouped, each sequence's blocks split into ranges at small batches):
# every check below runs all three against the same reference.
KERNELS = [paged_decode_attention, paged_decode_attention_grouped, paged_decode_attention_split]


@dataclass
class Case:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_tables: torch.Tensor
    context_lens: torch.Tensor
    scale: float
    block_size: int

    def args(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.q, self.k_cache, self.v_cache, self.block_tables, self.context_lens


def make_case(
    lens: list[int],
    num_q_heads: int = 16,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    block_size: int = 16,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    magnitude: float = 1.0,
) -> Case:
    """Sequences of ``lens`` tokens on randomly scattered blocks of one cache layer."""
    g = torch.Generator().manual_seed(seed)
    per_seq = [math.ceil(n / block_size) for n in lens]
    num_blocks = sum(per_seq) + 3  # spare blocks stay NaN
    order = torch.randperm(num_blocks, generator=g).tolist()
    nan_block = order.pop()  # block-table padding points here
    shape = (num_blocks * block_size, num_kv_heads, head_dim)
    k = torch.full(shape, float("nan"))
    v = torch.full(shape, float("nan"))
    tables = torch.full((len(lens), max(per_seq) + 2), nan_block, dtype=torch.int32)
    for b, (n, nb) in enumerate(zip(lens, per_seq, strict=True)):
        blocks = [order.pop() for _ in range(nb)]
        tables[b, :nb] = torch.tensor(blocks, dtype=torch.int32)
        slots = [blocks[p // block_size] * block_size + p % block_size for p in range(n)]
        k[slots] = torch.randn(n, num_kv_heads, head_dim, generator=g) * magnitude
        v[slots] = torch.randn(n, num_kv_heads, head_dim, generator=g)
    q = torch.randn(len(lens), num_q_heads, head_dim, generator=g) * magnitude
    return Case(
        q.to(DEVICE, dtype),
        k.to(DEVICE, dtype),
        v.to(DEVICE, dtype),
        tables.to(DEVICE),
        torch.tensor(lens, dtype=torch.int32, device=DEVICE),
        head_dim**-0.5,
        block_size,
    )


def check(case: Case) -> torch.Tensor:
    ref = paged_decode_attention_ref(*case.args(), case.scale, case.block_size)
    for kernel in KERNELS:
        out = kernel(*case.args(), case.scale, case.block_size)
        assert out.shape == ref.shape
        assert out.dtype == case.q.dtype
        assert torch.isfinite(out).all(), "read a slot outside the sequence (NaN leaked in)"
        tol = TOL[case.q.dtype]
        torch.testing.assert_close(out, ref, atol=tol, rtol=tol, msg=kernel.__name__)
    return ref


# --- the reference itself ---------------------------------------------------------------


def test_reference_equals_paged_gqa_attention_per_sequence() -> None:
    c = make_case([5, 17, 32], head_dim=32, block_size=4)
    ref = paged_decode_attention_ref(*c.args(), c.scale, c.block_size)
    for b, n in enumerate(c.context_lens.tolist()):
        blocks = c.block_tables[b].tolist()
        slots = torch.tensor([blocks[p // 4] * 4 + p % 4 for p in range(n)], device=DEVICE)
        one = paged_gqa_attention(c.q[b : b + 1], c.k_cache, c.v_cache, slots, c.scale)
        torch.testing.assert_close(ref[b : b + 1], one, atol=0, rtol=0)


# --- the kernel -------------------------------------------------------------------------


def test_one_token_returns_its_value() -> None:
    # softmax over a single score is 1, so the output is that token's v (per kv head).
    c = make_case([1], num_q_heads=4, num_kv_heads=2, head_dim=32)
    out = check(c)
    slot = int(c.block_tables[0, 0]) * c.block_size
    expected = c.v_cache[slot].repeat_interleave(2, dim=0)
    torch.testing.assert_close(out[0], expected, atol=1e-6, rtol=1e-6)


def test_qwen3_shapes_across_block_boundaries() -> None:
    # 16 query heads, 8 kv heads, head_dim 128, block size 16: Qwen3-0.6B in the engine.
    # Lengths just below, at and above block boundaries, plus the benchmark's 192 + 1.
    check(make_case([1, 15, 16, 17, 31, 32, 33, 193]))


@pytest.mark.parametrize(("num_q_heads", "num_kv_heads"), [(4, 4), (8, 2), (8, 1)])
def test_head_groupings(num_q_heads: int, num_kv_heads: int) -> None:
    # Plain multi-head (group 1), GQA with group 4, and multi-query (one kv head).
    check(make_case([3, 20, 47], num_q_heads, num_kv_heads, head_dim=64))


@pytest.mark.parametrize("block_size", [1, 8, 32])
def test_block_sizes(block_size: int) -> None:
    check(make_case([1, 9, 40], 4, 2, head_dim=32, block_size=block_size))


def test_float16() -> None:
    check(make_case([1, 16, 100, 193], dtype=torch.float16))


def test_large_scores_do_not_overflow() -> None:
    # Scores in the hundreds: exp(100) already overflows float32, so the kernel must subtract
    # the running maximum (and rescale) rather than exponentiate raw scores.
    c = make_case([40, 70], 4, 2, head_dim=64, dtype=torch.float16, magnitude=12.0)
    check(c)


@pytest.mark.parametrize("kernel", KERNELS)
def test_each_sequence_only_sees_its_own_blocks(kernel: Callable[..., torch.Tensor]) -> None:
    # Running sequences together or alone must give the same rows.
    c = make_case([7, 30, 2], 4, 2, head_dim=32, block_size=4)
    together = kernel(*c.args(), c.scale, c.block_size)
    for b in range(3):
        alone = kernel(
            c.q[b : b + 1],
            c.k_cache,
            c.v_cache,
            c.block_tables[b : b + 1],
            c.context_lens[b : b + 1],
            c.scale,
            c.block_size,
        )
        torch.testing.assert_close(alone, together[b : b + 1], atol=0, rtol=0)


@settings(max_examples=15, deadline=None)
@given(
    lens=st.lists(st.integers(1, 70), min_size=1, max_size=4),
    heads=st.sampled_from([(2, 1), (4, 2), (4, 4)]),
    head_dim=st.sampled_from([16, 32]),
    block_size=st.sampled_from([2, 4, 16]),
    seed=st.integers(0, 2**16),
)
def test_random_shapes(
    lens: list[int], heads: tuple[int, int], head_dim: int, block_size: int, seed: int
) -> None:
    check(make_case(lens, *heads, head_dim=head_dim, block_size=block_size, seed=seed))


# --- argument checks (in the launcher, before the kernel runs) --------------------------


def test_rejects_empty_sequence() -> None:
    c = make_case([3, 5], 4, 2, head_dim=32)
    c.context_lens[1] = 0
    with pytest.raises(ValueError, match="at least one token"):
        paged_decode_attention(*c.args(), c.scale, c.block_size)


def test_rejects_non_power_of_two_head_dim() -> None:
    c = make_case([3], 4, 2, head_dim=24)
    with pytest.raises(ValueError, match="power of 2"):
        paged_decode_attention(*c.args(), c.scale, c.block_size)


def test_rejects_int64_block_tables() -> None:
    c = make_case([3], 4, 2, head_dim=32)
    with pytest.raises(ValueError, match="int32"):
        paged_decode_attention(
            c.q, c.k_cache, c.v_cache, c.block_tables.long(), c.context_lens, c.scale, 16
        )


def test_rejects_heads_that_do_not_group() -> None:
    c = make_case([3], 6, 4, head_dim=32)
    with pytest.raises(ValueError, match="groups"):
        paged_decode_attention(*c.args(), c.scale, c.block_size)


def test_grouped_kernel_rejects_a_group_that_is_not_a_power_of_2() -> None:
    c = make_case([3], 12, 4, head_dim=32)  # 3 query heads per kv head
    paged_decode_attention(*c.args(), c.scale, c.block_size)  # the per-head kernel is fine
    with pytest.raises(ValueError, match="power-of-2 head group"):
        paged_decode_attention_grouped(*c.args(), c.scale, c.block_size)
    with pytest.raises(ValueError, match="power-of-2 head group"):
        paged_decode_attention_split(*c.args(), c.scale, c.block_size)


# --- split-K (M9) ---------------------------------------------------------------------------


def test_number_of_splits_fills_the_gpu_at_small_batches_only() -> None:
    # Qwen3-0.6B has 8 kv heads. M10 tuning on the T4: about 1,024 programs while the
    # batch alone gives fewer than 256 (batch 32), so batch 8 gets 16 ranges, 16 gets 8,
    # 24 gets 6 and 31 gets 5; batch 32 and up none. Never more ranges than blocks in a
    # table, or MAX_SPLITS.
    batches = (1, 2, 4, 8, 10, 16, 24, 31, 32, 64)
    assert [num_splits_for(b, 8, 64) for b in batches] == [16, 16, 16, 16, 13, 8, 6, 5, 1, 1]
    assert num_splits_for(1, 8, 4) == 4
    assert num_splits_for(1, 1, 64) == MAX_SPLITS
    assert num_splits_for(256, 1, 64) == 1


@pytest.mark.parametrize("splits", [1, 2, 3, 5, 16])
@pytest.mark.parametrize("lens", [[1], [17, 300, 33], [64, 2]])
def test_any_number_of_splits_gives_the_same_result(splits: int, lens: list[int]) -> None:
    # More ranges than a sequence has blocks leaves some empty: they must not count.
    c = make_case(lens, 4, 2, head_dim=32, block_size=4, seed=splits)
    ref = paged_decode_attention_ref(*c.args(), c.scale, c.block_size)
    out = launch_paged_decode_split(*c.args(), c.scale, c.block_size, num_splits=splits)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_one_long_sequence_on_gpu() -> None:
    # Split-K's case: batch 1 with a long context (2,048 tokens), fp16, Qwen3 shapes.
    check(make_case([2048], dtype=torch.float16, seed=2))


@pytest.mark.gpu
def test_benchmark_sized_batch_on_gpu() -> None:
    # The M3 benchmark's decode step at concurrency 64: 64 sequences of 193 tokens, fp16.
    check(make_case([193] * 64, dtype=torch.float16, seed=1))
