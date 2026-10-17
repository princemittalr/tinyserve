"""M0 smoke tests: the package imports and the test tooling itself works."""

import re

from hypothesis import given
from hypothesis import strategies as st

import tinyserve


def test_version_is_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", tinyserve.__version__)


@given(st.lists(st.integers()))
def test_hypothesis_runs(xs: list[int]) -> None:
    # Placeholder property so we know Hypothesis is wired up before M2 needs it.
    assert sorted(sorted(xs)) == sorted(xs)
