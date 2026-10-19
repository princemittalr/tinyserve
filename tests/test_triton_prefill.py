"""The Triton varlen prefill kernel (M9) must match the PyTorch reference.

The reference is ``paged_gqa_attention`` (attention.py) run one sequence at a time on
the sequence's slots: the M2-M7 prefill path. On a GPU these tests run the compiled
kernel; without one, Triton's interpreter (see conftest.py). Tolerances are M4's:
1e-5 in float32, 1e-3 in float16. As in test_triton_attention.py, every cache slot no
sequence owns is NaN and block-table padding points at a NaN block, so a read of one
slot too many fails the test.
"""

import math
from dataclasses import dataclass

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("triton", reason="Triton is not installed (no Windows wheels)")

from tests.test_engine import PROMPTS, expected
from tests.test_model import CFG, tiny_model
from tinyserve.attention import paged_gqa_attention
from tinyserve.batch import build_batch
from tinyserve.block_allocator import BlockAllocator
from tinyserve.block_table import BlockTable
from tinyserve.engine import Engine
from tinyserve.kv_cache import KVCache
from tinyserve.model import Qwen3
from tinyserve.triton_prefill import paged_prefill_attention

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOL = {torch.float32: 1e-5, torch.float16: 1e-3}


@dataclass
class Case:
    q: torch.Tensor  # [T, Hq, D]
    k: torch.Tensor
    v: torch.Tensor
    tables: torch.Tensor
    starts: torch.Tensor
    q_lens: torch.Tensor
    ctx_lens: torch.Tensor
    slots: list[torch.Tensor]  # per sequence, every token's slot (for the reference)
    block_size: int

    def run(
        self, out: torch.Tensor | None = None, rows_per_tile: int | None = None, block_n: int = 32
    ) -> torch.Tensor:
        scale = self.q.shape[2] ** -0.5
        return paged_prefill_attention(
            self.q, self.k, self.v, self.tables, self.starts, self.q_lens, self.ctx_lens,
            scale, self.block_size, int(self.q_lens.max()), out, rows_per_tile, block_n,
        )  # fmt: skip

    def reference(self, r: int) -> torch.Tensor:
        s, n = int(self.starts[r]), int(self.q_lens[r])
        return paged_gqa_attention(
            self.q[s : s + n], self.k, self.v, self.slots[r], self.q.shape[2] ** -0.5
        )


def make_case(
    specs: list[tuple[int, int]],  # per sequence: (cached tokens, new tokens)
    num_q_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 8,
    block_size: int = 4,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    gap: int = 0,  # extra batch rows between sequences (rows of other sequences)
) -> Case:
    g = torch.Generator().manual_seed(seed)
    totals = [c + n for c, n in specs]
    per_seq = [math.ceil(t / block_size) for t in totals]
    num_blocks = sum(per_seq) + 3
    order = torch.randperm(num_blocks, generator=g).tolist()
    nan_block = order.pop()
    shape = (num_blocks * block_size, num_kv_heads, head_dim)
    k, v = torch.full(shape, float("nan")), torch.full(shape, float("nan"))
    tables = torch.full((len(specs), max(per_seq) + 2), nan_block, dtype=torch.int32)
    slots, starts, row = [], [], 0
    for r, (total, nb) in enumerate(zip(totals, per_seq, strict=True)):
        blocks = [order.pop() for _ in range(nb)]
        tables[r, :nb] = torch.tensor(blocks, dtype=torch.int32)
        s = torch.tensor([blocks[p // block_size] * block_size + p % block_size
                          for p in range(total)])  # fmt: skip
        k[s] = torch.randn(total, num_kv_heads, head_dim, generator=g)
        v[s] = torch.randn(total, num_kv_heads, head_dim, generator=g)
        slots.append(s.to(DEVICE))
        starts.append(row)
        row += specs[r][1] + gap
    q = torch.randn(row, num_q_heads, head_dim, generator=g)

    def i32(xs: list[int]) -> torch.Tensor:
        return torch.tensor(xs, dtype=torch.int32, device=DEVICE)

    return Case(
        q.to(DEVICE, dtype), k.to(DEVICE, dtype), v.to(DEVICE, dtype), tables.to(DEVICE),
        i32(starts), i32([n for _, n in specs]), i32(totals), slots, block_size,
    )  # fmt: skip


def check(case: Case, out: torch.Tensor) -> None:
    tol = TOL[case.q.dtype]
    for r in range(len(case.slots)):
        s, n = int(case.starts[r]), int(case.q_lens[r])
        torch.testing.assert_close(out[s : s + n], case.reference(r), atol=tol, rtol=tol)


SHAPES = [
    (4, 4, 8),  # MHA
    (4, 2, 8),  # GQA, the tiny test model
    (4, 1, 8),  # MQA
    (16, 8, 128),  # Qwen3-0.6B
]
SPECS = [
    [(0, 5)],  # one plain prompt
    [(0, 37), (0, 1), (0, 9)],  # several tiles, a one-token prompt
    [(12, 4), (40, 7), (3, 1)],  # after cached prefixes (hits), chunks, a decode
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("specs", SPECS)
def test_matches_the_reference(shape: tuple[int, int, int], specs: list[tuple[int, int]]) -> None:
    hq, hkv, d = shape
    case = make_case(specs, hq, hkv, d, block_size=16 if d == 128 else 4)
    check(case, case.run())


@pytest.mark.parametrize("block_size", [1, 2, 4, 16, 32])
@pytest.mark.parametrize(("rows_per_tile", "block_n"), [(16, 16), (64, 32), (32, 64)])
def test_any_block_size_and_tile_shape(block_size: int, rows_per_tile: int, block_n: int) -> None:
    case = make_case([(9, 21), (0, 3), (30, 2)], block_size=block_size, seed=block_size)
    check(case, case.run(rows_per_tile=rows_per_tile, block_n=block_n))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_dtypes(dtype: torch.dtype) -> None:
    case = make_case(SPECS[2], 16, 8, 128, block_size=16, dtype=dtype)
    out = case.run()
    assert out.dtype == dtype
    check(case, out)


def test_rows_of_other_sequences_are_left_alone() -> None:
    case = make_case(SPECS[2], gap=2)  # 2 rows of someone else after each sequence
    out = torch.full_like(case.q, 7.0)
    case.run(out=out)
    check(case, out)
    mine = torch.zeros(case.q.shape[0], dtype=torch.bool)
    for s, n in zip(case.starts.tolist(), case.q_lens.tolist(), strict=True):
        mine[s : s + n] = True
    assert bool((out[~mine.to(out.device)] == 7.0).all())


@settings(max_examples=40, deadline=None)
@given(
    specs=st.lists(st.tuples(st.integers(0, 40), st.integers(1, 40)), min_size=1, max_size=4),
    shape=st.sampled_from(SHAPES[:3]),
    block_size=st.sampled_from([1, 2, 4, 8]),
    seed=st.integers(0, 2**16),
)
def test_random_batches(
    specs: list[tuple[int, int]], shape: tuple[int, int, int], block_size: int, seed: int
) -> None:
    case = make_case(specs, *shape, block_size=block_size, seed=seed)
    check(case, case.run(rows_per_tile=16, block_n=16))


def test_rejects_bad_arguments() -> None:
    case = make_case(SPECS[0])
    with pytest.raises(ValueError):
        case.run(rows_per_tile=8)  # under tl.dot's 16 rows
    with pytest.raises(ValueError):
        case.run(block_n=24)  # not a power of 2
    q3 = case.q[:, :3].contiguous()  # 3 query heads on 2 kv heads
    with pytest.raises(ValueError):
        paged_prefill_attention(q3, case.k, case.v, case.tables, case.starts, case.q_lens,
                                case.ctx_lens, 1.0, 4, 5)  # fmt: skip
    with pytest.raises(ValueError):
        paged_prefill_attention(case.q, case.k, case.v, case.tables, case.starts.long(),
                                case.q_lens, case.ctx_lens, 1.0, 4, 5)  # fmt: skip


# --- in the model and the engine ----------------------------------------------------------


def models(decode: str) -> tuple[Qwen3, Qwen3]:
    m = tiny_model()
    ref = Qwen3(CFG, m.w, decode_attention=decode)  # type: ignore[arg-type]
    return ref, Qwen3(CFG, m.w, decode_attention=decode, prefill_attention="triton")  # type: ignore[arg-type]


@pytest.mark.parametrize("decode", ["torch", "triton_grouped"])
def test_forward_batch_equals_the_per_sequence_path(decode: str) -> None:
    ref, ours = models(decode)
    cache = KVCache(CFG, 64, 4, dtype=torch.float32)
    alloc = BlockAllocator(64)
    tables = [BlockTable(alloc, 4) for _ in range(4)]
    # 0: a prompt, 1: a decode after 6 cached tokens, 2: 3 new after 9 cached, 3: 1 token.
    news = [[5, 9, 2, 7, 1], [4], [3, 8, 11], [6]]
    for t, cached in zip(tables, [0, 6, 9, 0], strict=True):
        t.append(cached)
    for t, n in zip(tables, news, strict=True):
        t.append(len(n))
    items = list(zip(news, tables, strict=True))
    # Fill the cached part of the cache identically for both runs: random k/v there.
    g = torch.Generator().manual_seed(1)
    cache.k.copy_(torch.randn(cache.k.shape, generator=g))
    cache.v.copy_(torch.randn(cache.v.shape, generator=g))
    saved = cache.k.clone(), cache.v.clone()
    want = ref.forward_batch(build_batch(items), cache)
    cache.k.copy_(saved[0])
    cache.v.copy_(saved[1])
    kernel, calls = ours.prefill_kernel, []
    assert kernel is not None

    def counted(*args: object) -> torch.Tensor:
        calls.append(1)
        return kernel(*args)

    ours.prefill_kernel = counted
    got = ours.forward_batch(build_batch(items), cache)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
    assert len(calls) == CFG.num_layers  # one launch per layer, for every prefill


@pytest.mark.parametrize(
    ("prefix_cache", "chunked", "num_blocks"),
    [(False, False, 64), (True, False, 64), (False, True, 64), (False, False, 7)],
)
def test_engine_matches_m1_greedy(prefix_cache: bool, chunked: bool, num_blocks: int) -> None:
    _, ours = models("triton_grouped")
    cache = KVCache(CFG, num_blocks, 4, dtype=torch.float32)
    e = Engine(ours, cache, max_num_seqs=4, max_batch_tokens=5 if chunked else 2048,
               prefix_cache=prefix_cache, chunked_prefill=chunked)  # fmt: skip
    prompts = [[7, 3, 9, 1, *p] for p in PROMPTS] if prefix_cache else PROMPTS
    assert e.generate(prompts, 8) == [expected(p, 8) for p in prompts]
    if num_blocks == 7:
        assert e.scheduler.num_preemptions > 0


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_qwen3_shapes_on_the_gpu(dtype: torch.dtype) -> None:
    # The M5 prefix workload's shape: 1,024 cached + 64 new tokens, a few of them at once,
    # next to a plain 128-token prompt. Compiled for the T4, fp16 and fp32 tiles.
    case = make_case([(1024, 64)] * 4 + [(0, 128)], 16, 8, 128, 16, dtype)
    check(case, case.run())
