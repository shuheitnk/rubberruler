import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rubberruler._transformations import (
    StrictlyIncreasingTransformation,
    log_density,
    sample_psi,
)

instance_keys = st.tuples(
    st.integers(0, 2**32 - 1),
    st.integers(1, 24),
    st.integers(1, 10_000),
    st.integers(2, 40),
)


def test_knot_count_matches_the_fixed_value() -> None:
    """m is 5 for seed=42, function 1, instance number 1, dimension 10."""
    assert len(log_density(42, 1, 1, 10).x) == 5


@pytest.mark.parametrize(
    ("z", "expected"),
    [
        (0.25, 0.14828703445665872),
        (0.5, 0.21005289625892246),
        (0.75, 0.35689515337847727),
    ],
)
def test_psi_matches_the_fixed_values(z: float, expected: float) -> None:
    """psi matches these fixed values within 1e-9."""
    assert float(sample_psi(42, 1, 1, 10)(z)) == pytest.approx(expected, abs=1e-9)


@settings(deadline=None)
@given(keys=instance_keys)
def test_psi_maps_0_to_0_and_1_to_1(keys: tuple[int, int, int, int]) -> None:
    """psi(0) = 0 and psi(1) = 1 (up to rounding at the last knot)."""
    psi = sample_psi(*keys)
    assert float(psi(0.0)) == 0.0
    assert float(psi(1.0)) == pytest.approx(1.0, abs=1e-12)


@settings(deadline=None)
@given(keys=instance_keys)
def test_psi_is_strictly_increasing(keys: tuple[int, int, int, int]) -> None:
    """psi is strictly increasing on [0, 1]."""
    values = sample_psi(*keys)(np.linspace(0, 1, 1001))
    assert np.all(np.diff(values) > 0)


@settings(deadline=None)
@given(keys=instance_keys, above=st.floats(1.0, 1e6, exclude_min=True))
def test_g_is_the_identity_above_1(
    keys: tuple[int, int, int, int], above: float
) -> None:
    """g(y) = z for z > 1, so g is continuous at 1 and the identity above."""
    g = StrictlyIncreasingTransformation(sample_psi(*keys), f_min=0.0, f_max=1.0)
    assert g(np.array([above]))[0] == above
    assert g(np.array([1.0]))[0] == pytest.approx(1.0, abs=1e-12)


@settings(deadline=None)
@given(keys=instance_keys)
def test_g_rescales_with_f_min_and_f_max(keys: tuple[int, int, int, int]) -> None:
    """z = max((y - f_min) / (f_max - f_min), 0) and g = psi(z) on [0, 1]."""
    psi = sample_psi(*keys)
    g = StrictlyIncreasingTransformation(psi, f_min=10.0, f_max=30.0)
    y = np.array([0.0, 10.0, 15.0, 20.0, 30.0, 50.0])
    expected = np.array([0.0, 0.0, float(psi(0.25)), float(psi(0.5)), 1.0, 2.0])
    np.testing.assert_allclose(g(y), expected, rtol=0, atol=1e-15)


@settings(deadline=None)
@given(keys=instance_keys)
def test_g_is_increasing_on_the_objective_range(
    keys: tuple[int, int, int, int],
) -> None:
    """g is strictly increasing from f_min upwards."""
    g = StrictlyIncreasingTransformation(sample_psi(*keys), f_min=-3.0, f_max=5.0)
    values = g(np.linspace(-3.0, 20.0, 2001))
    assert np.all(np.diff(values) > 0)
