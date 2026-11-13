"""A random strictly increasing transformation of any objective function."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rubberruler._errors import check_int, detail, error
from rubberruler._instances import (
    DEFAULT_SPLIT_SEED,
    FMAX_SAMPLES_PER_DIMENSION,
    MAX_SEED,
    OBJECTIVES_SEED,
    TRANSFORMED_SUFFIX,
    InstanceSplit,
    OwnObjective,
    ProblemInstance,
    own_instance,
    own_value,
    transform,
)
from rubberruler._seeding import generator
from rubberruler._transformations import StrictlyIncreasingTransformation, objective_psi


def transform_objective(
    objective: Callable[[NDArray[np.float64]], float],
    lower_bounds: ArrayLike,
    upper_bounds: ArrayLike,
    *,
    seed: int,
    n_samples: int | None = None,
    optimal_value: float | None = None,
) -> Callable[[NDArray[np.float64]], float | NDArray[np.float64]]:
    """Return the transformed objective g(f(x)): ``objective`` under a rubber ruler.

    Use it to augment your own training data with transformed objectives of your
    problems. g is strictly increasing, so the order of any two values, and with it
    every rank-based feature, stays the same, while value-based features change.

    g rescales f by f_min and f_max, the smallest (or ``optimal_value``, if given)
    and the largest finite value of f at ``n_samples`` points drawn uniformly from
    the bounds, so that f_min becomes 0 and f_max becomes 1, and applies psi, a
    random strictly increasing curve from (0, 0) to (1, 1), the same kind that
    ``transform`` applies to BBOB. Outside [0, 1], g only rescales, so a value below
    f_min becomes negative and a value above f_max becomes larger than 1. Values
    better than the sample, near the optimum, are therefore only rescaled, not
    stretched; the uniform samples that features are computed from mostly fall
    inside.

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
            seed for each transformed objective, also across objectives, such as
            ``seed=10 * i + k`` for the k-th of ten transformed objectives of the
            i-th objective.
        n_samples: The number of points that set the smallest and largest value,
            at least 2, or None for 1000 x d, the number that ``transform`` uses
            on BBOB. objective is called once for each, one point at a time; for
            an expensive objective, a smaller n_samples, such as 1_000, costs less,
            and more values above the largest one found are then only rescaled.
        optimal_value: The optimal value of objective, if you know it, as in
            ``optimal_values`` of ``rr.objective_instances``. It replaces the
            smallest sampled value in g, so values near the optimum are stretched
            too. It must not be larger than any sampled value.
    """
    lower, upper = _check_bounds(lower_bounds, upper_bounds)
    seed = check_int(
        "seed",
        seed,
        minimum=0,
        maximum=MAX_SEED,
        example="a different int for each transformed objective, such as seed=0",
    )
    n_samples = (
        FMAX_SAMPLES_PER_DIMENSION * lower.size
        if n_samples is None
        else check_int(
            "n_samples",
            n_samples,
            minimum=2,
            example="an int such as n_samples=1_000, or None for 1000 x d",
        )
    )
    known = _optimal_value(optimal_value)
    rng = generator("fmax", seed, "objective")
    points = rng.uniform(lower, upper, size=(n_samples, lower.size))
    values = np.empty(n_samples)
    for index, point in enumerate(points):
        try:
            values[index] = float(objective(point))
        except Exception as exception:
            raise error(
                f"objective raised {type(exception).__name__} at sample point "
                f"{index + 1} of {n_samples}: {detail(exception)}",
                why="g is fitted to the values of objective at points drawn "
                "uniformly from the bounds.",
                fix="make objective take a point of shape (d,) and return a float; "
                "try objective(lower_bounds). The original exception is chained to "
                "this error, with its traceback.",
            ) from exception
    finite = values[np.isfinite(values)]
    distinct = np.unique(finite).size
    if distinct < 2:
        raise error(
            f"objective gave only {distinct} distinct finite "
            f"value{'' if distinct == 1 else 's'} at the {n_samples} sample points.",
            why="g rescales objective by its smallest and largest value, which must "
            "differ.",
            fix="pass an objective that returns finite values (not NaN or inf) that "
            "are not all equal on the bounds, or a larger n_samples.",
        )
    smallest = float(np.min(finite))
    if known is not None and known > smallest:
        raise error(
            f"optimal_value is {known!r}, but objective is {smallest!r} at a sample "
            "point.",
            why="the optimal value is the smallest value of objective on its bounds.",
            fix="pass the true optimal value of objective, or leave it out.",
        )
    transformation = StrictlyIncreasingTransformation(
        objective_psi(seed),
        f_min=smallest if known is None else known,
        f_max=float(np.max(finite)),
    )
    return TransformedObjective(objective, transformation)


def _optimal_value(optimal_value: object) -> float | None:
    """The given optimal value as a float, or None; stop if it is not a finite
    number."""
    if optimal_value is None:
        return None
    try:
        number = float(optimal_value)  # pyright: ignore[reportArgumentType]
    except (TypeError, ValueError):
        number = float("nan")
    if isinstance(optimal_value, bool) or not np.isfinite(number):
        raise error(
            f"optimal_value is {optimal_value!r}.",
            why="the optimal value is a finite number.",
            fix="pass the optimal value of objective as a float, such as "
            "optimal_value=0.0, or leave it out.",
        )
    return number


@dataclass(frozen=True)
class TransformedObjective:
    """g(objective(x)), returned by rr.transform_objective."""

    objective: Callable[[NDArray[np.float64]], object]
    transformation: StrictlyIncreasingTransformation

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


# One test objective for every 4 objectives, rounded half up: a set of 24 objectives
# has 6 test objectives (with the 1 in 11 of BBOB, it would have 2).
OBJECTIVES_PER_TEST_OBJECTIVE = 4


def objective_instances(
    objectives: Mapping[
        str,
        tuple[
            Callable[[NDArray[np.float64]], float | NDArray[np.float64]],
            ArrayLike,
            ArrayLike,
        ],
    ],
    *,
    test: Sequence[str] | None = None,
    optimal_values: Mapping[str, float] | None = None,
    split_seed: int = 42,
) -> InstanceSplit:
    """Make P_train, P_test, and P~_test from your own objective functions.

    Each objective becomes one instance of the suite "objectives", whose id is its
    name. rubberruler splits them: one in 4, rounded half up and at least one, is a
    test instance, chosen at random by ``split_seed`` from the sorted names, or the
    objectives that ``test`` names; the others are training instances. Each test
    instance also has a transformed test instance, whose objective is g(f(x)) with g
    fitted as by ``rr.transform_objective``: f is rescaled by its smallest and
    largest value at 1000 x d points drawn uniformly from its bounds (or by its
    optimal value, if given, and the largest value), and a random rubber ruler is
    applied. Pass the result to ``rr.run_test(..., instances=split)``.

    ``split_seed`` decides only the split. The runs, the samples of the features, and
    g follow the master seed, as on BBOB: the instances have the master seed 42, and
    ``rr.run_test(..., seed=...)`` replaces it with its own. They do not depend on
    the split, so another ``split_seed`` reuses the cached runs of the same
    objectives, and only the transformed test instances of objectives new to the
    test set run.

    Keep the objectives distinct problems: shifted copies of one function in both
    sets make the test instances easy. With few objectives, the answer can depend on
    the split; repeat the test with a few values of ``split_seed``.

    Args:
        objectives: Names mapped to ``(objective, lower_bounds, upper_bounds)``: a
            function that takes a point of shape (d,) and returns a float, and the
            bounds of its search range, of shape (d,). At least 2, all with the same
            d, with names that are non-empty strings.
        test: The names of the objectives that become test instances, such as
            ``test=["f3", "f7"]``, at least one and not all; None to draw them by
            ``split_seed``.
        optimal_values: Names mapped to the optimal value of the objective, if you
            know it. A known optimal value replaces the smallest sampled value in g.
        split_seed: The seed of the split, from 0 to 2**32 - 1. It is left at its
            default when ``test`` is given.
    """
    split_seed = check_int(
        "split_seed",
        split_seed,
        minimum=0,
        maximum=MAX_SEED,
        example="an int such as split_seed=1",
    )
    if not isinstance(objectives, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise error(
            f"objectives is a {type(objectives).__name__}.",
            why="objectives maps names to (objective, lower_bounds, upper_bounds).",
            fix=FIX_OBJECTIVES,
        )
    if len(objectives) < 2:
        raise error(
            f"objectives has {len(objectives)} objective"
            f"{'' if len(objectives) == 1 else 's'}.",
            why="the objectives are split into training and test instances, so at "
            "least 2 are needed.",
            fix=FIX_OBJECTIVES,
        )
    known = _optimal_values(optimal_values, objectives)
    named = _test_names(test, objectives, split_seed)
    names = sorted(objectives)
    checked = {name: _objective_entry(name, objectives[name]) for name in names}
    dimensions = {name: entry[1].size for name, entry in checked.items()}
    first = names[0]
    for name in names[1:]:
        if dimensions[name] != dimensions[first]:
            raise error(
                f'objectives has d={dimensions[first]} for "{first}" and '
                f'd={dimensions[name]} for "{name}".',
                why="one test uses one dimension, as on BBOB and MA-BBOB.",
                fix="pass objectives of the same dimension, or test each dimension "
                "with its own call.",
            )
    instances: list[ProblemInstance] = []
    for number, name in enumerate(names, start=1):
        function, lower, upper = checked[name]
        own = OwnObjective(function, known.get(name))
        own_value(name, own, (lower + upper) / 2)
        instances.append(own_instance(name, own, lower, upper, number, OBJECTIVES_SEED))
    if named is None:
        test_count = max(
            1,
            (len(names) + OBJECTIVES_PER_TEST_OBJECTIVE // 2)
            // OBJECTIVES_PER_TEST_OBJECTIVE,
        )
        order = generator("split", split_seed, "objectives").permutation(len(names))
        chosen = {int(position) for position in order[:test_count]}
    else:
        chosen = {position for position, name in enumerate(names) if name in named}
    train = tuple(p for position, p in enumerate(instances) if position not in chosen)
    tested = tuple(p for position, p in enumerate(instances) if position in chosen)
    return InstanceSplit(
        train,
        tested,
        tuple(transform(instance) for instance in tested),
        split_seed=split_seed if named is None else None,
    )


FIX_OBJECTIVES = (
    'pass a dict such as {"sphere": (sphere, np.full(5, -5.0), np.full(5, 5.0)), '
    '"rastrigin": (rastrigin, np.full(5, -5.12), np.full(5, 5.12))}.'
)


def _objective_entry(
    name: object, entry: object
) -> tuple[
    Callable[[NDArray[np.float64]], float], NDArray[np.float64], NDArray[np.float64]
]:
    """Check one item of objectives: a name, and (objective, lower, upper)."""
    if not isinstance(name, str) or not name or name.endswith(TRANSFORMED_SUFFIX):
        raise error(
            f"objectives has the name {name!r}.",
            why="the name of an objective becomes its instance_id, a non-empty "
            f'string that does not end with "{TRANSFORMED_SUFFIX}".',
            fix='use names such as "rastrigin".',
        )
    if (
        not isinstance(entry, tuple)
        or len(cast(tuple[object, ...], entry)) != 3
        or not callable(cast(tuple[object, ...], entry)[0])
    ):
        raise error(
            f'objectives["{name}"] is not (objective, lower_bounds, upper_bounds).',
            why="each objective needs its function and the bounds of its search range.",
            fix=f'pass objectives["{name}"] = (my_function, np.full(d, -5.0), '
            "np.full(d, 5.0)).",
        )
    function, lower_bounds, upper_bounds = cast(tuple[object, object, object], entry)
    lower, upper = _check_bounds(lower_bounds, upper_bounds)
    return cast(Callable[[NDArray[np.float64]], float], function), lower, upper


def _test_names(
    test: object, objectives: Mapping[str, object], split_seed: int
) -> frozenset[str] | None:
    """The names of the objectives that become test instances, or None to draw them
    by split_seed; stop on a name that is not in objectives, a repeated name, no
    name, or every name, and with a split_seed other than its default."""
    if test is None:
        return None
    fix = (
        "pass a list of some of the names of objectives, such as "
        'test=["f3", "f7"], or leave out test.'
    )
    if not isinstance(test, Sequence) or isinstance(test, str):
        raise error(
            f"test is a {type(test).__name__}.",
            why="test lists the names of the objectives that become test instances.",
            fix=fix,
        )
    names: list[object] = list(cast(Sequence[object], test))
    for name in names:
        if not isinstance(name, str) or name not in objectives:
            raise error(
                f"test has {name!r}, which is not in objectives.",
                why="test lists the names of the objectives that become test "
                "instances.",
                fix=fix,
            )
    repeated = sorted({str(name) for name in names if names.count(name) > 1})
    if repeated:
        raise error(
            f"test has {repeated[0]!r} more than once.",
            why="each objective is a test instance once.",
            fix=fix,
        )
    if not names or len(names) == len(objectives):
        raise error(
            f"test has {len(names)} of the {len(objectives)} objectives.",
            why="the split needs at least one training and one test instance.",
            fix=fix,
        )
    if split_seed != DEFAULT_SPLIT_SEED:
        raise error(
            f"objective_instances received both test and split_seed={split_seed}.",
            why="split_seed decides only which objectives are test instances, and "
            "test names them.",
            fix="leave out split_seed, or leave out test to draw the test instances "
            "by split_seed.",
        )
    return frozenset(cast(list[str], names))


def _optimal_values(
    optimal_values: object, objectives: Mapping[str, object]
) -> dict[str, float]:
    """The known optimal values as floats; stop on an unknown name or a value that is
    not a finite number."""
    if optimal_values is None:
        return {}
    if not isinstance(optimal_values, Mapping):
        raise error(
            f"optimal_values is a {type(optimal_values).__name__}.",
            why="optimal_values maps the names of objectives to their optimal values.",
            fix='pass a dict such as optimal_values={"rastrigin": 0.0}, or None.',
        )
    known: dict[str, float] = {}
    for name, value in cast(Mapping[object, object], optimal_values).items():
        if not isinstance(name, str) or name not in objectives:
            raise error(
                f"optimal_values has {name!r}, which is not in objectives.",
                why="an optimal value belongs to one of the objectives.",
                fix="use the names of objectives as the keys of optimal_values.",
            )
        try:
            number = float(value)  # pyright: ignore[reportArgumentType]
        except (TypeError, ValueError):
            number = float("nan")
        if not np.isfinite(number):
            raise error(
                f"optimal_values has {value!r} for {name!r}.",
                why="an optimal value is a finite number.",
                fix=f"pass the optimal value of {name!r} as a float, or leave it out.",
            )
        known[name] = number
    return known
