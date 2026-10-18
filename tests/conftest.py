"""Shared pytest configuration.

Tests marked ``@pytest.mark.gpu`` are skipped unless a CUDA device is visible,
so the same suite runs on a laptop, in CI (CPU only) and on a Kaggle T4.

Without a GPU, Triton kernels run in Triton's interpreter (``TRITON_INTERPRET=1``): the
same kernel code, executed with NumPy on CPU tensors. It is slow and says nothing about
speed, but it checks the kernel's results in CI. It must be set before any kernel is
defined, i.e. before the test modules are imported, so it is set here.
"""

import importlib.util
import os

import pytest


def _cuda_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return bool(torch.cuda.is_available())


if not _cuda_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _cuda_available():
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
