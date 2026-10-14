"""A random strictly increasing transformation of any objective function."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rubberruler._errors import check_int, error
from rubberruler._instances import MAX_SEED
from rubberruler._seeding import generator
from rubberruler._transformations import ObjectiveTransformation, objective_psi


def transform_objective(
    objective: Callable[[NDArray[np.float64]], float],
    lower_bounds: ArrayLike,
    upper_bounds: ArrayLike,
    *,
    seed: int,
    n_samples: int = 10_000,
) -> Callable[[NDArray[np.float64]], float | NDArray[np.float64]]:
    """Return the transformed objective g(f(x)): ``objective`` under a rubber ruler.

    Use it to augment your own training data with transformed objectives of your
    problems. g is strictly increasing, so the order of any two values, and with it
    every rank-based feature, stays the same, while value-based features change.

    g rescales f to [0, 1] by the smallest and the largest finite value of f at
    ``n_samples`` points drawn uniformly from the bounds, and applies psi, a random
    strictly increasing curve from (0, 0) to (1, 1), the same kind that
    ``transform`` applies to BBOB. Outside [0, 1], g only rescales. Values better
    than the sample, near the optimum, are therefore only rescaled, not stretched;
    the uniform samples that features are computed from mostly fall inside.

    The returned function returns a float for a point of shape (d,). If
    ``objective`` also takes n points of shape (n, d) and returns n values, so does
    the returned function. NaN stays NaN, and +-inf stays +-inf. The returned
    function can be pickled if ``objective`` can, for example to use it in worker
    processes.

    Args:
        objective: f, which takes a point of shape (d,) and returns a float.
        lower_bounds: The lower end of the search range, of shape (d,).
        upper_bounds: The upper end of the search range, of shape (d,).
        seed: An int from 0 to 2**32 - 1. psi depends only on ``seed``, and the
            same ``seed`` gives the same psi for every objective. Pass a different
            seed for each transformed objective, such as ``seed=k`` for the k-th.
        n_samples: The number of points that set the smallest and largest value,
            at least 2. objective is called once for each.
    """
    lower, upper = _check_bounds(lower_bounds, upper_bounds)
    seed = check_int(
        "seed",
        seed,
        minimum=0,
        maximum=MAX_SEED,
        example="a different int for each transformed objective, such as seed=0",
    )
    n_samples = check_int(
        "n_samples", n_samples, minimum=2, example="an int such as n_samples=10_000"
    )
    rng = generator("fmax", seed, "objective")
    points = rng.uniform(lower, upper, size=(n_samples, lower.size))
    values = np.empty(n_samples)
    for index, point in enumerate(points):
        try:
            values[index] = float(objective(point))
        except Exception as exception:
            raise error(
                f"objective raised {type(exception).__name__} at sample point "
                f"{index + 1} of {n_samples}.",
                why="g is fitted to the values of objective at points drawn "
                "uniformly from the bounds.",
                fix="make objective take a point of shape (d,) and return a float; "
                "try objective(lower_bounds). The original exception is shown above.",
            ) from exception
    finite = values[np.isfinite(values)]
    distinct = np.unique(finite).size
    if distinct < 2:
        raise error(
            f"objective gave only {distinct} distinct finite "
            f"value{'' if distinct == 1 else 's'} at the {n_samples} sample points.",
            why="g rescales objective by its smallest and largest value, which must "
            "differ.",
            fix="pass an objective that is not constant on the bounds, or a larger "
            "n_samples.",
        )
    transformation = ObjectiveTransformation(
        objective_psi(seed), f_min=float(np.min(finite)), f_max=float(np.max(finite))
    )
    return TransformedObjective(objective, transformation)


@dataclass(frozen=True)
class TransformedObjective:
    """g(objective(x)), returned by rr.transform_objective."""

    objective: Callable[[NDArray[np.float64]], object]
    transformation: ObjectiveTransformation

    def __call__(self, x: NDArray[np.float64], /) -> float | NDArray[np.float64]:
        values = self.transformation(np.asarray(self.objective(x), dtype=np.float64))
        return float(values) if values.ndim == 0 else values


def _check_bounds(
    lower_bounds: object, upper_bounds: object
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lower = _bounds_array("lower_bounds", lower_bounds)
    upper = _bounds_array("upper_bounds", upper_bounds)
    if lower.shape != upper.shape:
        raise error(
            f"lower_bounds has shape {lower.shape} and upper_bounds has shape "
            f"{upper.shape}.",
            why="each dimension needs one lower and one upper bound.",
            fix="pass two arrays of shape (d,).",
        )
    below = np.flatnonzero(lower >= upper)
    if below.size:
        index = int(below[0])
        raise error(
            f"lower_bounds[{index}]={float(lower[index])!r} is not below "
            f"upper_bounds[{index}]={float(upper[index])!r}.",
            why="the sample points are drawn uniformly between the bounds.",
            fix="pass lower_bounds smaller than upper_bounds in every dimension, "
            "such as -5 and 5.",
        )
    return lower, upper


def _bounds_array(name: str, value: object) -> NDArray[np.float64]:
    fix = f"pass {name} as a 1-D array of finite numbers, such as np.full(d, -5.0)."
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise error(
            f"{name} is not an array of numbers.",
            why="the bounds give the search range of each dimension.",
            fix=fix,
        ) from None
    if array.ndim != 1 or array.size == 0:
        raise error(
            f"{name} has shape {array.shape}.",
            why="the bounds give the search range of each dimension, so they are "
            "1-D with one entry per dimension.",
            fix=fix,
        )
    if not np.all(np.isfinite(array)):
        raise error(
            f"{name} contains {float(array[~np.isfinite(array)][0])!r}.",
            why="the sample points are drawn uniformly between the bounds, which "
            "must be finite.",
            fix=fix,
        )
    return array
