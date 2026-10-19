"""Every Triton kernel compiles for the T4 with no register spills (M9 step 8).

The other kernel tests run in Triton's interpreter on CPU, which cannot see compile
errors or spills; ``scripts/check_t4_compile.py`` compiles for sm_75 without a GPU.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("triton", reason="Triton is not installed (no Windows wheels)")

SCRIPT = Path(__file__).parent.parent / "scripts" / "check_t4_compile.py"


def test_every_kernel_compiles_for_the_t4_without_spills() -> None:
    env = {k: v for k, v in os.environ.items() if k != "TRITON_INTERPRET"}
    r = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.count(" ok: ") == 14
