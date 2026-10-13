import subprocess
import sys

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rubberruler._seeding import derive_seed


@pytest.mark.parametrize(
    ("domain", "keys", "expected"),
    [
        ("transform", ("bbob", 1, 1, 10), 2715082314),
        ("fmax", ("bbob", 1, 1, 10), 1678112591),
        ("features", ("bbob", 1, 1, 10), 1617016386),
        ("run", ("bbob", 1, 1, 10, "cma-es", 0), 374886278),
        ("transform", ("objective",), 1546156558),
        ("fmax", ("objective",), 3206986475),
    ],
)
def test_derive_seed_returns_the_fixed_values(
    domain: str, keys: tuple[str | int, ...], expected: int
) -> None:
    """derive_seed returns these fixed values."""
    assert derive_seed(domain, 42, *keys) == expected


@given(
    domain=st.text(),
    seed=st.integers(0, 2**32 - 1),
    keys=st.lists(st.one_of(st.text(), st.integers())),
)
def test_derive_seed_is_a_32_bit_seed(
    domain: str, seed: int, keys: list[str | int]
) -> None:
    """derive_seed returns a value in [0, 2**32 - 1]."""
    assert 0 <= derive_seed(domain, seed, *keys) <= 2**32 - 1


def test_derive_seed_is_the_same_in_another_process() -> None:
    """derive_seed does not depend on the process (no hash())."""
    code = (
        "from rubberruler._seeding import derive_seed; "
        "print(derive_seed('run', 7, 'x', 3))"
    )
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert int(output) == derive_seed("run", 7, "x", 3)


def test_derive_seed_separates_domains_and_keys() -> None:
    """Each use of randomness gets its own seed."""
    seeds = {
        derive_seed("transform", 42, "bbob", 1, 1, 10),
        derive_seed("fmax", 42, "bbob", 1, 1, 10),
        derive_seed("transform", 43, "bbob", 1, 1, 10),
        derive_seed("transform", 42, "bbob", 1, 2, 10),
        derive_seed("run", 42, "bbob", 1, 1, 10, "a", 0),
        derive_seed("run", 42, "bbob", 1, 1, 10, "a", 1),
        derive_seed("transform", 42, "objective"),
        derive_seed("fmax", 42, "objective"),
    }
    assert len(seeds) == 8
