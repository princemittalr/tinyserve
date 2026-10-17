"""Shared pytest configuration.

Tests marked ``@pytest.mark.gpu`` are skipped unless a CUDA device is visible,
so the same suite runs on a laptop, in CI (CPU only) and on a Kaggle T4.
"""

import importlib.util

import pytest


def _cuda_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return bool(torch.cuda.is_available())


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _cuda_available():
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
