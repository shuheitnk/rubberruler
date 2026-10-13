import pickle

import numpy as np
import pytest
import scipy.stats
from helpers import assert_three_lines
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

import rubberruler as rr
from rubberruler._features import preprocess
from rubberruler._objective import TransformedObjective
from rubberruler._seeding import generator
from rubberruler._transformations import draw_log_density, psi_from

seeds = st.integers(0, 2**32 - 1)


def sphere(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sum(np.asarray(x) ** 2, axis=-1)


def rastrigin(x: NDArray[np.float64]) -> NDArray[np.float64]:
    x = np.asarray(x)
    return 10 * x.shape[-1] + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=-1)


def first_coordinate(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(x)[..., 0]


def infinite_on_the_right(x: NDArray[np.float64]) -> float:
    return np.inf if x[0] > 0 else float(x[0])


def transformed(
    objective: object, dimension: int = 2, seed: int = 0, n_samples: int = 10_000
) -> TransformedObjective:
    g = rr.transform_objective(
        objective,  # pyright: ignore[reportArgumentType]
        np.full(dimension, -5.0),
        np.full(dimension, 5.0),
        seed=seed,
        n_samples=n_samples,
    )
    assert isinstance(g, TransformedObjective)
    return g


def test_sphere_matches_the_fixed_values() -> None:
    """R-AUG-1: for the sphere at d = 2 on [-5, 5]^2 and seed = 0, f_min, f_max,
    and g at (0, 0), (1, 2), and (5, 5) match the values of spec 6.7 within 1e-9."""
    g = transformed(sphere)
    assert g.transformation.f_min == pytest.approx(0.0026513298523528042, abs=1e-9)
    assert g.transformation.f_max == pytest.approx(49.88435942364988, abs=1e-9)
    assert g(np.array([0.0, 0.0])) == pytest.approx(-5.315234689572469e-05, abs=1e-9)
    assert g(np.array([1.0, 2.0])) == pytest.approx(0.13974027323189725, abs=1e-9)
    assert g(np.array([5.0, 5.0])) == pytest.approx(1.0023182962406312, abs=1e-9)


def test_psi_draws_from_the_objective_keys() -> None:
    """R-AUG-1: psi is drawn by the steps of spec 6.3 from the "transform" seed
    with the key "objective", and the sample from the "fmax" seed with it."""
    g = transformed(sphere)
    z = np.linspace(0, 1, 101)
    psi = psi_from(draw_log_density(generator("transform", 0, "objective")))
    np.testing.assert_array_equal(g.transformation.psi(z), psi(z))
    rng = generator("fmax", 0, "objective")
    sample = sphere(rng.uniform(-5.0, 5.0, size=(10_000, 2)))
    assert g.transformation.f_min == np.min(sample)
    assert g.transformation.f_max == np.max(sample)


@settings(deadline=None, max_examples=25)
@given(seed=seeds)
def test_g_is_strictly_increasing_on_both_sides(seed: int) -> None:
    """R-AUG-1: g is strictly increasing over values below f_min, between f_min
    and f_max, and above f_max, and continuous at z = 0 and z = 1."""
    g = transformed(sphere, seed=seed, n_samples=100).transformation
    low, high = g.f_min, g.f_max
    width = high - low
    y = np.linspace(low - width, high + width, 30_001)
    assert np.all(np.diff(g(y)) > 0)
    edges = np.array([low, high])
    step = 1e-9 * width
    np.testing.assert_allclose(g(edges - step), [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(g(edges + step), [0.0, 1.0], atol=1e-6)
    assert g(np.array([low]))[0] == 0.0


def test_g_keeps_the_order_and_the_ranks() -> None:
    """R-AUG-1: on a uniform sample, the transformed values have the same ranks as
    the objective values, so the rank preprocessing of spec 6.4 gives the same y,
    while the value preprocessing does not."""
    rng = np.random.default_rng(1)
    points = rng.uniform(-5.0, 5.0, size=(500, 5))
    y = rastrigin(points)
    g = transformed(rastrigin, dimension=5)
    stretched = np.asarray(g(points))
    np.testing.assert_array_equal(
        scipy.stats.rankdata(y), scipy.stats.rankdata(stretched)
    )
    np.testing.assert_array_equal(
        preprocess(y, "rank")[0], preprocess(stretched, "rank")[0]
    )
    assert not np.allclose(preprocess(y, "value")[0], preprocess(stretched, "value")[0])


def test_the_same_seed_gives_the_same_g() -> None:
    """R-AUG-1: the same seed gives the same values, and the same psi for any
    objective; another seed gives another psi."""
    points = np.random.default_rng(2).uniform(-5.0, 5.0, size=(50, 2))
    np.testing.assert_array_equal(
        np.asarray(transformed(sphere)(points)), np.asarray(transformed(sphere)(points))
    )
    z = np.linspace(0, 1, 101)
    psi = transformed(sphere).transformation.psi(z)
    np.testing.assert_array_equal(psi, transformed(rastrigin).transformation.psi(z))
    other = transformed(sphere, seed=1).transformation.psi(z)
    assert np.max(np.abs(psi - other)) > 1e-3


def test_many_points_at_once_match_one_at_a_time() -> None:
    """R-AUG-1: an objective that takes (n, d) gives a g that takes (n, d) with the
    same values as one point at a time, and a float for one point."""
    g = transformed(rastrigin)
    points = np.random.default_rng(3).uniform(-5.0, 5.0, size=(20, 2))
    many = g(points)
    assert isinstance(many, np.ndarray)
    assert many.shape == (20,)
    one = [g(point) for point in points]
    assert all(type(value) is float for value in one)
    np.testing.assert_array_equal(many, one)


def test_g_survives_a_pickle_round_trip() -> None:
    """R-AUG-1: the returned function pickles with its objective and gives the
    same values after the round trip."""
    g = transformed(rastrigin)
    points = np.random.default_rng(4).uniform(-5.0, 5.0, size=(20, 2))
    restored = pickle.loads(pickle.dumps(g))
    np.testing.assert_array_equal(np.asarray(restored(points)), np.asarray(g(points)))


def test_nan_and_infinity_pass_through() -> None:
    """R-AUG-1: g maps NaN to NaN, +inf to +inf, and -inf to -inf."""
    g = transformed(first_coordinate)
    values = np.asarray(g(np.array([[np.nan, 0.0], [np.inf, 0.0], [-np.inf, 0.0]])))
    assert np.isnan(values[0])
    assert values[1] == np.inf
    assert values[2] == -np.inf


def test_infinite_values_in_the_sample_are_left_out() -> None:
    """R-AUG-1: f_min and f_max are the smallest and largest finite values."""
    g = transformed(infinite_on_the_right, n_samples=1000)
    assert -0.1 < g.transformation.f_max <= 0.0
    assert g(np.array([4.5, 0.0])) == np.inf


@pytest.mark.parametrize(
    ("lower_bounds", "upper_bounds", "first_line"),
    [
        (
            [-5.0, -5.0],
            [5.0, 5.0, 5.0],
            "lower_bounds has shape (2,) and upper_bounds has shape (3,).",
        ),
        ([[-5.0, -5.0]], [[5.0, 5.0]], "lower_bounds has shape (1, 2)."),
        ([-5.0, -5.0], 5.0, "upper_bounds has shape ()."),
        ([], [], "lower_bounds has shape (0,)."),
        (["a", "b"], [5.0, 5.0], "lower_bounds is not an array of numbers."),
        ([-np.inf, -5.0], [5.0, 5.0], "lower_bounds contains -inf."),
        ([-5.0, -5.0], [5.0, np.nan], "upper_bounds contains nan."),
        (
            [-5.0, 5.0],
            [5.0, 5.0],
            "lower_bounds[1]=5.0 is not below upper_bounds[1]=5.0.",
        ),
    ],
)
def test_wrong_bounds_stop_with_three_lines(
    lower_bounds: object, upper_bounds: object, first_line: str
) -> None:
    """R-AUG-1, R-ERR-1: bounds of different shapes, not 1-D, not finite, or with
    lower >= upper stop with a RubberRulerError of three lines."""
    with pytest.raises(rr.RubberRulerError) as raised:
        rr.transform_objective(
            sphere,  # pyright: ignore[reportArgumentType]
            lower_bounds,  # pyright: ignore[reportArgumentType]
            upper_bounds,  # pyright: ignore[reportArgumentType]
            seed=0,
        )
    assert_three_lines(raised.value)
    assert str(raised.value).split("\n")[0] == first_line


@pytest.mark.parametrize(
    ("arguments", "first_line"),
    [
        ({"seed": -1}, "seed=-1 is not allowed."),
        ({"seed": 2**32}, "seed=4294967296 is not allowed."),
        ({"seed": True}, "seed=True is not allowed."),
        ({"seed": 0, "n_samples": 1}, "n_samples=1 is not allowed."),
    ],
)
def test_wrong_seed_or_n_samples_stops_with_three_lines(
    arguments: dict[str, object], first_line: str
) -> None:
    """R-AUG-1, R-ERR-1: a seed outside [0, 2**32 - 1] or n_samples below 2
    stops with a RubberRulerError of three lines."""
    with pytest.raises(rr.RubberRulerError) as raised:
        rr.transform_objective(sphere, [-5.0, -5.0], [5.0, 5.0], **arguments)  # pyright: ignore[reportArgumentType]
    assert_three_lines(raised.value)
    assert str(raised.value).split("\n")[0] == first_line


def test_an_objective_that_raises_stops_with_its_exception_chained() -> None:
    """R-AUG-1, R-ERR-1: an exception of objective on the sample stops with a
    RubberRulerError that says where, chained to the original exception."""

    def failing(x: NDArray[np.float64]) -> float:
        if x[0] > 0:
            raise ZeroDivisionError("no")
        return float(x[0])

    with pytest.raises(rr.RubberRulerError) as raised:
        rr.transform_objective(failing, [-5.0, -5.0], [5.0, 5.0], seed=0)
    assert_three_lines(raised.value)
    first_line = str(raised.value).split("\n")[0]
    assert first_line.startswith("objective raised ZeroDivisionError at sample point ")
    assert first_line.endswith(" of 10000.")
    assert isinstance(raised.value.__cause__, ZeroDivisionError)


@pytest.mark.parametrize(
    ("objective", "first_line"),
    [
        (
            lambda x: 1.0,  # pyright: ignore[reportUnknownLambdaType]
            "objective gave only 1 distinct finite value at the 100 sample points.",
        ),
        (
            lambda x: np.nan,  # pyright: ignore[reportUnknownLambdaType]
            "objective gave only 0 distinct finite values at the 100 sample points.",
        ),
    ],
)
def test_a_constant_objective_stops_with_three_lines(
    objective: object, first_line: str
) -> None:
    """R-AUG-1, R-ERR-1: fewer than 2 distinct finite values on the sample stop
    with a RubberRulerError of three lines."""
    with pytest.raises(rr.RubberRulerError) as raised:
        rr.transform_objective(
            objective,  # pyright: ignore[reportArgumentType]
            [-5.0, -5.0],
            [5.0, 5.0],
            seed=0,
            n_samples=100,
        )
    assert_three_lines(raised.value)
    assert str(raised.value).split("\n")[0] == first_line
