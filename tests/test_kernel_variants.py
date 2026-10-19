"""Each Triton kernel compiles once per dtype, whatever the batch looks like (M10).

Triton compiles a new variant of a kernel for each value of a constexpr and for each
int argument that is 1, a multiple of 16, or neither. Before M10 the block-table width
and the split count reached the decode and prefill kernels that way, so a step with a
width or a decode count not seen before waited for the compiler mid-run. These tests
launch every attention kernel over widths 1, 2, 9, 16 and 32 and every split count, and
count the variants Triton holds: on a GPU it must stay at one per kernel. Triton's
interpreter (CPU) compiles nothing, so the counting test needs a GPU; the CPU checks
are that the results are right at every width and that a bad split count is refused.
"""

from typing import Any

import pytest
import torch

pytest.importorskip("triton", reason="Triton is not installed (no Windows wheels)")

from tests.test_triton_attention import DEVICE, TOL, make_case
from tests.test_triton_prefill import check as check_prefill
from tests.test_triton_prefill import make_case as make_prefill_case
from tinyserve import triton_attention as ta
from tinyserve import triton_prefill as tp
from tinyserve.attention import paged_decode_attention_ref

WIDTHS = [1, 2, 9, 16, 32]


def widened(tables: torch.Tensor, width: int) -> torch.Tensor:
    """``tables`` cut or padded to ``width`` columns (padding repeats the last column,
    which make_case fills with its NaN block), as a new tensor: its row stride is width."""
    if width <= tables.shape[1]:
        return tables[:, :width].contiguous()
    pad = tables[:, -1:].expand(-1, width - tables.shape[1])
    return torch.cat([tables, pad], dim=1).contiguous()


def variants(kernel: Any) -> int:
    """How many compiled variants Triton holds for ``kernel`` in this process."""
    return sum(len(cache[0]) for cache in kernel.device_caches.values())


def decode_launches(dtype: torch.dtype) -> list[torch.Tensor]:
    """Every decode kernel at every width (and the split kernel at every split count),
    each result checked against the reference."""
    c = make_case([5, 12, 16], dtype=dtype, seed=7)  # one block each: width 1 works
    ref = paged_decode_attention_ref(*c.args(), c.scale, c.block_size)
    outs = []
    for w in WIDTHS:
        args = (c.q, c.k_cache, c.v_cache, widened(c.block_tables, w), c.context_lens)
        for launch in (ta.launch_paged_decode, ta.launch_paged_decode_grouped):
            outs.append(launch(*args, c.scale, c.block_size))
        for s in range(2, ta.MAX_SPLITS + 1):
            outs.append(ta.launch_paged_decode_split(*args, c.scale, c.block_size, num_splits=s))
    for out in outs:
        torch.testing.assert_close(out, ref, atol=TOL[dtype], rtol=TOL[dtype])
    return outs


def prefill_launches(dtype: torch.dtype) -> None:
    c = make_prefill_case([(3, 5), (0, 9), (10, 6)], 16, 8, 128, block_size=16, dtype=dtype)
    for w in WIDTHS:
        c.tables = widened(c.tables, w)
        check_prefill(c, c.run())


def test_every_width_and_split_count_gives_the_reference() -> None:
    decode_launches(torch.float32)
    prefill_launches(torch.float32)


def test_split_count_out_of_range_is_refused() -> None:
    c = make_case([20], head_dim=32, block_size=4)
    for bad in (-1, ta.MAX_SPLITS + 1):
        with pytest.raises(ValueError, match="num_splits"):
            ta.launch_paged_decode_split(*c.args(), c.scale, c.block_size, num_splits=bad)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_one_compiled_variant_per_kernel_and_dtype(dtype: torch.dtype) -> None:
    assert DEVICE == "cuda"
    kernels = {
        "decode": ta._paged_decode_kernel,
        "grouped": ta._paged_decode_grouped_kernel,
        "split": ta._paged_decode_split_kernel,
        "merge": ta._merge_splits_kernel,
        "prefill": tp._paged_prefill_kernel,
    }
    before = {name: variants(k) for name, k in kernels.items()}
    decode_launches(dtype)
    prefill_launches(dtype)
    # Other tests in this process may have compiled the variant already (then 0 new).
    new = {name: variants(k) - before[name] for name, k in kernels.items()}
    assert all(n <= 1 for n in new.values()), new
