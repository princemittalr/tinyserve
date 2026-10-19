"""Compile every Triton kernel for the T4 (sm_75) on any machine, no GPU needed (M9).

The CPU tests run the kernels in Triton's interpreter, which executes the Python and so
misses what only the compiler sees: M9 step 5's fused kernel ran in every CPU test but
did not compile for fp16 on the T4. Triton can compile for a GPU it does not have, down
to PTX, and the ptxas Triton ships with reports each kernel's registers and spills.

For each kernel, at Qwen3-0.6B's shapes and the launch options its wrapper uses, in
float16 and float32, this prints registers, bytes spilled to local memory and the
number of tensor-core (mma) instructions, and fails on a compile error or a spill of
more than ``MAX_SPILL`` bytes. It also fails if a kernel specializes on an argument that
changes from step to step (``PER_STEP``, M10).
What it found on the T4 (M9 step 8): Triton emits no tensor-core instructions for
sm_75 (``tl.dot`` becomes FMA loops), and the prefill kernel's first tiles spilled
about 10 KB per thread, which is why it ran ~6x slower than batched SDPA. A small spill
can be the right trade: the fastest prefill tile on the T4 spills 936 bytes (step 9).

``--arch 80`` (M10) compiles for another GPU instead, here an A100 (sm_80), to see what
Triton would emit where it has tensor cores; the spill limit applies to the T4 only.
Compiling is all it shows: nothing here ran on such a GPU.

Usage:
    python scripts/check_t4_compile.py          # exit code 1 if any kernel fails
    python scripts/check_t4_compile.py --arch 80
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from typing import Any

if os.environ.get("TRITON_INTERPRET"):  # interpreted kernels cannot be compiled
    sys.exit("unset TRITON_INTERPRET first")

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from tinyserve import triton_attention as ta
from tinyserve import triton_fused as tf
from tinyserve import triton_prefill as tp

PTXAS = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "ptxas")
HQ, HKV, D, BS = 16, 8, 128, 16  # Qwen3-0.6B; the engine's default block size
MAX_SPILL = 1024  # bytes per thread


def specs(dt: str) -> list[tuple[str, Any, dict[str, str], dict[str, Any], int]]:
    """(name, kernel, argument types other than i32, constexprs, num_warps) per kernel."""
    p = "*" + dt
    decode = {"q_ptr": p, "k_ptr": p, "v_ptr": p, "out_ptr": p, "block_tables_ptr": "*i32",
              "context_lens_ptr": "*i32", "scale": "fp32"}  # fmt: skip
    group = {"GROUP": HQ // HKV, "BLOCK_SIZE": BS, "HEAD_DIM": D}
    split = {**decode, "m_ptr": "*fp32", "l_ptr": "*fp32", "acc_ptr": "*fp32"}
    del split["out_ptr"]
    rows, block_n, warps = tp.DEFAULT_TILE
    return [
        ("decode (M4)", ta._paged_decode_kernel, decode, group, 4),
        ("decode grouped (M8)", ta._paged_decode_grouped_kernel, decode, group, 4),
        ("decode split (M9)", ta._paged_decode_split_kernel, split,
         {"NUM_Q_HEADS": HQ, **group}, 4),
        ("merge splits (M9)", ta._merge_splits_kernel,
         {"m_ptr": "*fp32", "l_ptr": "*fp32", "acc_ptr": "*fp32", "out_ptr": p},
         {"NUM_Q_HEADS": HQ, "SPLITS_PAD": ta.MAX_SPLITS, "HEAD_DIM": D}, 4),
        ("prefill (M9)", tp._paged_prefill_kernel,
         {**decode, "q_starts_ptr": "*i32", "q_lens_ptr": "*i32"},
         {**group, "HEAD_DIM_PAD": D, "BLOCK_M": rows // (HQ // HKV), "BLOCK_N": block_n,
          "PRECISION": "ieee" if dt == "fp32" else "tf32"}, warps),
        ("rms_norm_rows (M9)", tf._rms_norm_rows_kernel,
         {"x_ptr": p, "w_ptr": p, "out_ptr": p, "eps": "fp32"}, {"N": 1024, "N_PAD": 1024}, 4),
        ("qk_norm_rope_store (M9)", tf._qk_norm_rope_store_kernel,
         {"q_ptr": p, "k_ptr": p, "v_ptr": p, "q_out_ptr": p, "q_w_ptr": p, "k_w_ptr": p,
          "cos_ptr": "*fp32", "sin_ptr": "*fp32", "slots_ptr": "*i64", "k_cache_ptr": p,
          "v_cache_ptr": p, "eps": "fp32"}, {"NUM_Q_HEADS": HQ, "HEAD_DIM": D}, 4),
    ]  # fmt: skip


def check(kernel: Any, types: dict[str, str], constexprs: dict[str, Any], warps: int,
          arch: int = 75) -> str:  # fmt: skip
    """Compile for sm_<arch>; returns ptxas's report. Raises on a compile error, or on the
    T4 (sm_75) on a spill over ``MAX_SPILL``."""
    sig = {a: "constexpr" if a in constexprs else types.get(a, "i32") for a in kernel.arg_names}
    target = GPUTarget("cuda", arch, 32)
    k = triton.compile(ASTSource(kernel, sig, constexprs=constexprs), target=target,
                       options={"num_warps": warps})  # fmt: skip
    ptx = k.asm["ptx"]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "k.ptx")
        with open(path, "w") as f:
            f.write(ptx)
        cmd = [PTXAS, "-v", "--gpu-name", f"sm_{arch}", path, "-o", path + ".o"]
        report = subprocess.run(cmd, capture_output=True, text=True, check=True).stderr
    regs = re.search(r"Used (\d+) registers", report)
    spills = re.search(r"(\d+) bytes spill stores", report)
    if regs is None or spills is None:
        raise RuntimeError(f"unexpected ptxas output: {report}")
    line = f"{regs[1]:>3} registers, {spills[1]:>5} bytes spilled, {ptx.count('mma.sync')} mma"
    if arch == 75 and int(spills[1]) > MAX_SPILL:
        raise RuntimeError(line)
    return line


# Kernel arguments whose value changes from step to step (M10): listed here, not taken
# from the kernels' own NO_SPECIALIZE, so that dropping one there fails this check.
PER_STEP = ("stride_bt", "num_splits")


def specialized_per_step(kernel: Any) -> list[str]:
    """Per-step arguments of ``kernel`` that Triton still specializes on (each new value
    compiled a new variant mid-run). Empty when all is well."""
    return [p.name for p in kernel.params
            if p.name in PER_STEP and not p.do_not_specialize]  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=int, default=75, help="compute capability, e.g. 80")
    arch = parser.parse_args().arch
    failed = 0
    for name, kernel, *_ in specs("fp16"):
        bad = specialized_per_step(kernel)
        failed += bool(bad)
        print(f"{name:<29} {'FAILED: specializes on ' + ', '.join(bad) if bad else 'one variant'}")
    for dt in ("fp16", "fp32"):
        for name, kernel, types, constexprs, warps in specs(dt):
            try:
                line = check(kernel, types, constexprs, warps, arch)
                print(f"{dt} {name:<24} ok: {line}")
            except Exception as e:  # report every kernel, then fail
                failed += 1
                print(f"{dt} {name:<24} FAILED: {str(e).splitlines()[-1]}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
