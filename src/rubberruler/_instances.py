"""BBOB and MA-BBOB problem instances and their transformed versions."""

import functools
import itertools
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from rubberruler._errors import check_int, detail, error, three_lines
from rubberruler._seeding import generator, instance_keys
from rubberruler._transformations import (
    StrictlyIncreasingTransformation,
    draw_log_density,
    psi_from,
    sample_psi,
)

try:
    import ioh
except ImportError as exception:
    # ioh's Windows wheels do not bundle MSVCP140.dll, which comes with the
    # Visual C++ Redistributable.
    if "DLL load failed" not in str(exception):
        raise
    raise ImportError(
        three_lines(
            "ioh, the library behind the BBOB functions, could not load its "
            f"compiled part ({exception}).",
            why="ioh needs the Microsoft Visual C++ Redistributable, which is not "
            "installed.",
            fix="install its X64 package from "
            "https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist and "
            "start Python again.",
        )
    ) from exception

Suite = Literal["bbob", "ma-bbob", "objectives"]

FUNCTION_IDS = tuple(range(1, 25))
# The default training and test instance numbers of each suite.
DEFAULT_INSTANCE_NUMBERS: dict[Suite, tuple[range, range]] = {
    "bbob": (range(11, 111), range(1, 11)),
    "ma-bbob": (range(1, 2401), range(2401, 2641)),
}
# g takes the largest value of f (and the smallest, where the optimal value is not
# known) at 1000 x d uniform random points.
FMAX_SAMPLES_PER_DIMENSION = 1000
# The version of the definition of g: psi, its extension by the identity outside
# [0, 1], and the sample of f_min and f_max. The runs of a transformed instance are
# cached under it, so that runs under an earlier definition are not used.
TRANSFORMATION_VERSION = 2
TRANSFORMED_SUFFIX = "~transformed"
OBJECTIVES = "objectives"
# The master seed of the instances of rr.objective_instances; run_test replaces it
# with its own seed.
OBJECTIVES_SEED = 42
# The default split_seed of rr.objective_instances.
DEFAULT_SPLIT_SEED = 42
MAX_SEED = 2**32 - 1
# ioh receives the instance number as a C int.
MAX_INSTANCE_NUMBER = 2**31 - 1


@dataclass(frozen=True)
class ProblemInstance:
    """One problem instance p of BBOB, MA-BBOB, or your objectives, or its
    transformed version.

    Read-only and not callable: only the Problem that an algorithm receives can
    evaluate the objective.

    Attributes:
        instance_id: ``f"f{function_id:02d}_i{instance_number:03d}_d{dimension}"``
            on BBOB, ``f"mabbob_i{instance_number:03d}_d{dimension}"`` on MA-BBOB,
            and the name of the objective on the suite "objectives", followed by
            ``~transformed`` on a transformed instance.
        suite: The problem suite, "bbob", "ma-bbob", or "objectives".
        function_id: The BBOB function, from 1 to 24, or None on MA-BBOB, which
            has no function ID (an MA-BBOB instance combines several BBOB
            functions), and on the suite "objectives".
        instance_number: The instance number in the suite; on the suite
            "objectives", the position of the name among the sorted names, from 1.
        dimension: d.
        lower_bounds: The lower end of the search range, an array of shape (d,).
        upper_bounds: The upper end of the search range, an array of shape (d,).
        seed: The master seed. On the suite "objectives", 42 from
            rr.objective_instances, which run_test replaces with its seed.
        is_transformed: Whether the objective is g(f(x)) rather than f(x).
        original_instance_id: The id before the transformation (the instance's own
            id if it is not transformed).
        own: On the suite "objectives", your objective function and, once
            transformed, its g; None on BBOB and MA-BBOB.
    """

    instance_id: str
    suite: Suite
    function_id: int | None
    instance_number: int
    dimension: int
    lower_bounds: NDArray[np.float64] = field(repr=False, compare=False)
    upper_bounds: NDArray[np.float64] = field(repr=False, compare=False)
    seed: int
    is_transformed: bool
    original_instance_id: str
    own: "OwnObjective | None" = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class OwnObjective:
    """An objective function of the user (the suite "objectives").

    ``transformation`` is g of a transformed instance, fitted when ``transform``
    makes it, and None before.
    """

    function: Callable[[NDArray[np.float64]], object]
    optimal_value: float | None
    transformation: StrictlyIncreasingTransformation | None = None


@dataclass(frozen=True)
class InstanceSplit:
    """The training, test, and transformed test instances (P_train, P_test, P~_test).

    Iterating over it gives every instance: ``train``, then ``test``, then
    ``transformed_test``. So ``measure_performance(portfolio, split)`` measures all.

    ``split_seed`` is the split_seed of rr.objective_instances that drew the test
    instances, and None if ``test`` named them, or on BBOB and MA-BBOB.
    """

    train: tuple[ProblemInstance, ...]
    test: tuple[ProblemInstance, ...]
    transformed_test: tuple[ProblemInstance, ...]
    split_seed: int | None = None

    def __iter__(self) -> Iterator[ProblemInstance]:
        return itertools.chain(self.train, self.test, self.transformed_test)

    def __repr__(self) -> str:
        return (
            f"InstanceSplit(train={len(self.train)} instances, "
            f"test={len(self.test)} instances, "
            f"transformed_test={len(self.transformed_test)} instances)"
        )


def bbob_instances(
    *,
    dimension: int = 10,
    train_instance_numbers: Sequence[int] = DEFAULT_INSTANCE_NUMBERS["bbob"][0],
    test_instance_numbers: Sequence[int] = DEFAULT_INSTANCE_NUMBERS["bbob"][1],
    seed: int = 42,
) -> InstanceSplit:
    """Make P_train, P_test, and P~_test from all 24 BBOB functions.

    Each set has one instance per function and instance number, ordered by function
    and then by instance number. The transformed test instances are
    ``transform(p)`` for each test instance p.

    Args:
        dimension: d, an int of at least 2.
        train_instance_numbers: The BBOB instance numbers of the training instances:
            ints from 1 to 2**31 - 1, without duplicates, and at least one of them.
        test_instance_numbers: The instance numbers of the test instances: the
            same, and none of them among the training ones.
        seed: The master seed, from 0 to 2**32 - 1. It is stored in every instance
            and decides the transformations, the samples, and the run seeds.
    """
    dimension = check_dimension(dimension)
    train, test = check_instance_numbers(
        train_instance_numbers, test_instance_numbers, "bbob"
    )
    seed = check_seed(seed)
    return make_split(dimension, train, test, seed)


def ma_bbob_instances(
    *,
    dimension: int = 10,
    train_instance_numbers: Sequence[int] = DEFAULT_INSTANCE_NUMBERS["ma-bbob"][0],
    test_instance_numbers: Sequence[int] = DEFAULT_INSTANCE_NUMBERS["ma-bbob"][1],
    seed: int = 42,
) -> InstanceSplit:
    """Make P_train, P_test, and P~_test from MA-BBOB.

    An MA-BBOB instance is an affine combination of the BBOB functions, which ioh
    derives from the instance number alone. Each set has one instance per instance
    number, ordered by instance number, and ``function_id`` is None. The
    transformed test instances are ``transform(p)`` for each test instance p.

    Args:
        dimension: d, an int of at least 2.
        train_instance_numbers: The MA-BBOB instance numbers of the training
            instances: ints from 1 to 2**31 - 1, without duplicates, and at least
            one of them.
        test_instance_numbers: The instance numbers of the test instances: the
            same, and none of them among the training ones.
        seed: The master seed, from 0 to 2**32 - 1. It is stored in every instance
            and decides the transformations, the samples, and the run seeds.
    """
    dimension = check_dimension(dimension)
    train, test = check_instance_numbers(
        train_instance_numbers, test_instance_numbers, "ma-bbob"
    )
    seed = check_seed(seed)
    return make_split(dimension, train, test, seed, suite="ma-bbob")


def make_split(
    dimension: int,
    train_instance_numbers: Sequence[int],
    test_instance_numbers: Sequence[int],
    seed: int,
    function_ids: Sequence[int] = FUNCTION_IDS,
    *,
    suite: Suite = "bbob",
) -> InstanceSplit:
    """bbob_instances or ma_bbob_instances without the checks.

    On BBOB, for a chosen set of functions; MA-BBOB ignores ``function_ids``.
    """
    chosen: Sequence[int | None] = function_ids if suite == "bbob" else (None,)
    train = tuple(
        _instance(function_id, number, dimension, seed)
        for function_id in chosen
        for number in train_instance_numbers
    )
    test = tuple(
        _instance(function_id, number, dimension, seed)
        for function_id in chosen
        for number in test_instance_numbers
    )
    return InstanceSplit(train, test, tuple(transform(instance) for instance in test))


def transform(instance: ProblemInstance) -> ProblemInstance:
    """Return the transformed instance, whose objective is g(f(x)).

    g rescales f to [0, 1] by its optimum and the largest of its values at 1000 x d
    uniform random points, and stretches it by a random strictly increasing curve;
    outside [0, 1], g only rescales. It depends only on the master seed, the suite,
    the function (on BBOB), the instance number, and the dimension, so the same
    instance always receives the same g. The search space does not change.
    """
    if not isinstance(instance, ProblemInstance):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise error(
            f"transform received {type(instance).__name__}.",
            why="transform needs a ProblemInstance from rr.bbob_instances() or "
            "rr.ma_bbob_instances().",
            fix="pass an element of split.test, such as split.test[0].",
        )
    if instance.is_transformed:
        raise error(
            f"transform received {instance.instance_id}, which is already transformed.",
            why="each instance has one transformation; applying it twice is not "
            "defined.",
            fix=f"pass the original instance {instance.original_instance_id}.",
        )
    own = instance.own
    if own is not None:
        own = replace(own, transformation=_own_transformation(instance, own))
    return replace(
        instance,
        instance_id=instance.instance_id + TRANSFORMED_SUFFIX,
        is_transformed=True,
        own=own,
    )


def transformation_of(
    instance: ProblemInstance,
) -> StrictlyIncreasingTransformation | None:
    """Return g of a transformed instance, or None for an original instance."""
    if not instance.is_transformed:
        return None
    if instance.own is not None:
        return instance.own.transformation
    return _transformation(
        instance.seed,
        instance.function_id,
        instance.instance_number,
        instance.dimension,
    )


def keys_of(instance: ProblemInstance) -> tuple[str | int, ...]:
    """The instance keys of derive_seed; the same for an instance and its
    transformed instance. On the suite "objectives": "objectives", the name, d."""
    if instance.suite == OBJECTIVES:
        return (OBJECTIVES, instance.original_instance_id, instance.dimension)
    return instance_keys(
        instance.function_id, instance.instance_number, instance.dimension
    )


def reseeded(split: InstanceSplit, seed: int) -> InstanceSplit:
    """The split of objectives of the user with the master seed ``seed``: the same
    objectives in the same sets, and the transformations fitted anew. The split
    itself if its instances already have that seed."""
    if all(instance.seed == seed for instance in split):
        return split
    train = tuple(replace(instance, seed=seed) for instance in split.train)
    test = tuple(replace(instance, seed=seed) for instance in split.test)
    return InstanceSplit(
        train,
        test,
        tuple(transform(instance) for instance in test),
        split_seed=split.split_seed,
    )


def own_instance(
    name: str,
    own: OwnObjective,
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
    instance_number: int,
    seed: int,
) -> ProblemInstance:
    """An instance of the suite "objectives" for one objective of the user."""
    lower_bounds = lower_bounds.copy()
    upper_bounds = upper_bounds.copy()
    lower_bounds.flags.writeable = False
    upper_bounds.flags.writeable = False
    return ProblemInstance(
        instance_id=name,
        suite=OBJECTIVES,
        function_id=None,
        instance_number=instance_number,
        dimension=lower_bounds.size,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        seed=seed,
        is_transformed=False,
        original_instance_id=name,
        own=own,
    )


def own_value(name: str, own: OwnObjective, x: NDArray[np.float64]) -> float:
    """The value of the user's objective at one point, as a float; stop with a
    three-line error that names the objective if it raises or returns no number."""
    try:
        return float(own.function(x))  # pyright: ignore[reportArgumentType]
    except Exception as exception:
        raise error(
            f'objective "{name}" raised {type(exception).__name__} at a point of '
            f"shape {x.shape}: {detail(exception)}",
            why="rubberruler evaluates your objective, one point at a time, in the "
            "runs and on the samples of the features and of g.",
            fix="make the objective take a point of shape (d,) and return a float "
            "for every point of its bounds. The original exception is chained to this "
            "error, with its traceback.",
        ) from exception


def _own_transformation(
    instance: ProblemInstance, own: OwnObjective
) -> StrictlyIncreasingTransformation:
    """g of an objective of the user, as rr.transform_objective fits it,
    with the seeds of the instance keys and the optimal value as f_min if given."""
    name = instance.original_instance_id
    keys = keys_of(instance)
    rng = generator("fmax", instance.seed, *keys)
    n_samples = FMAX_SAMPLES_PER_DIMENSION * instance.dimension
    points = rng.uniform(
        instance.lower_bounds,
        instance.upper_bounds,
        size=(n_samples, instance.dimension),
    )
    values = np.array([own_value(name, own, point) for point in points])
    finite = values[np.isfinite(values)]
    if np.unique(finite).size < 2:
        raise error(
            f'objective "{name}" gave {np.unique(finite).size} distinct finite '
            f"values at the {n_samples} sample points of g.",
            why="g rescales the objective by its smallest and largest value, which "
            "must differ.",
            fix="pass an objective that returns finite values (not NaN or inf) that "
            "are not all equal on its bounds.",
        )
    smallest = float(np.min(finite))
    f_min = smallest if own.optimal_value is None else own.optimal_value
    if f_min > smallest:
        raise error(
            f'optimal_values gives {f_min!r} for "{name}", but the objective is '
            f"{smallest!r} at a sample point.",
            why="the optimal value is the smallest value of the objective on its "
            "bounds.",
            fix=f'pass the true optimal value of "{name}", or leave it out of '
            "optimal_values.",
        )
    psi = psi_from(draw_log_density(generator("transform", instance.seed, *keys)))
    return StrictlyIncreasingTransformation(
        psi, f_min=f_min, f_max=float(np.max(finite))
    )


def original_of(instance: ProblemInstance) -> ProblemInstance:
    """Return the instance before the transformation (itself if not transformed)."""
    if not instance.is_transformed:
        return instance
    return replace(
        instance, instance_id=instance.original_instance_id, is_transformed=False
    )


class InstanceObjective:
    """Evaluates the objective of an instance with ioh: f, or g(f) if transformed."""

    def __init__(self, instance: ProblemInstance) -> None:
        if instance.own is None:
            self._problem = _ioh_problem(
                instance.function_id, instance.instance_number, instance.dimension
            )
        else:
            name, own = instance.original_instance_id, instance.own
            self._problem = _OwnProblem(name, own)
        self._transformation = transformation_of(instance)

    def one(self, x: NDArray[np.float64], /) -> float:
        value = float(self._problem(x))
        if self._transformation is None:
            return value
        return float(self._transformation(np.array([value]))[0])

    def many(self, points: NDArray[np.float64], /) -> NDArray[np.float64]:
        values = np.asarray(self._problem(points), dtype=np.float64)
        if self._transformation is None:
            return values
        return self._transformation(values)


class _OwnProblem:
    """The user's objective as InstanceObjective calls it: one point, or n points
    one after another (the objective need not take a batch)."""

    def __init__(self, name: str, own: OwnObjective) -> None:
        self._name = name
        self._own = own

    def __call__(self, x: NDArray[np.float64]) -> float | NDArray[np.float64]:
        points = np.asarray(x, dtype=np.float64)
        if points.ndim == 1:
            return own_value(self._name, self._own, points)
        return np.array([own_value(self._name, self._own, point) for point in points])


def check_dimension(dimension: object) -> int:
    return check_int(
        "dimension", dimension, minimum=2, example="an int such as dimension=10"
    )


def check_seed(seed: object) -> int:
    return check_int(
        "seed", seed, minimum=0, maximum=MAX_SEED, example="an int such as seed=42"
    )


def check_suite(suite: object) -> Suite:
    if suite == "bbob":
        return "bbob"
    if suite == "ma-bbob":
        return "ma-bbob"
    if suite == OBJECTIVES:
        return OBJECTIVES
    what = (
        f"suite={suite!r} is not allowed."
        if isinstance(suite, str)
        else f"suite received a {type(suite).__name__}."
    )
    raise error(
        what,
        why='suite must be "bbob", "ma-bbob", or "objectives" (with instances from '
        "rr.objective_instances).",
        fix='pass suite="bbob" or suite="ma-bbob"; for objective functions of your '
        "own, pass instances=rr.objective_instances(...).",
    )


def check_instance_numbers(
    train_instance_numbers: object,
    test_instance_numbers: object,
    suite: Suite = "bbob",
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return both sets of numbers sorted; stop if one is invalid or they overlap."""
    default_train, default_test = DEFAULT_INSTANCE_NUMBERS[suite]
    train = _instance_numbers(
        "train_instance_numbers", train_instance_numbers, default_train
    )
    test = _instance_numbers(
        "test_instance_numbers", test_instance_numbers, default_test
    )
    shared = sorted(set(train) & set(test))
    if shared:
        raise error(
            f"test_instance_numbers and train_instance_numbers share "
            f"{_numbers_text(shared)}.",
            why="a test instance must not also be a training instance.",
            fix="pass instance numbers that do not overlap, such as the defaults "
            f"{default_train!r} for training and {default_test!r} for test.",
        )
    return train, test


def instance_tuple(
    instances: Iterable[ProblemInstance], function: str
) -> tuple[ProblemInstance, ...]:
    """Return ``instances`` as a tuple; stop if it is empty or not all instances."""
    fix = "pass split, split.train, split.test, or split.transformed_test."
    try:
        result = tuple(instances)
    except TypeError:
        raise error(
            f"{function} received instances of type {type(instances).__name__}.",
            why="instances must be an iterable of ProblemInstance.",
            fix=fix,
        ) from None
    if not result:
        raise error(
            f"{function} received no instances.",
            why="there is nothing to compute.",
            fix=fix,
        )
    seen: set[str] = set()
    for instance in result:
        if not isinstance(instance, ProblemInstance):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise error(
                f"{function} received {type(instance).__name__} in instances.",
                why="instances must be ProblemInstance objects from "
                "rr.bbob_instances(), rr.ma_bbob_instances(), or "
                "rr.objective_instances().",
                fix=fix,
            )
        if instance.instance_id in seen:
            raise error(
                f"{function} received {instance.instance_id} twice in instances.",
                why="each instance gives one row of the result.",
                fix="pass each instance once.",
            )
        seen.add(instance.instance_id)
    return result


def _instance_numbers(name: str, numbers: object, example: range) -> tuple[int, ...]:
    fix = (
        f"pass a range or list of ints from 1 to {MAX_INSTANCE_NUMBER}, such as "
        f"{name}={example!r}."
    )
    if not isinstance(numbers, Sequence) or isinstance(numbers, str):
        raise error(
            f"{name} is a {type(numbers).__name__}.",
            why=f"{name} must be a sequence of instance numbers.",
            fix=fix,
        )
    values: list[object] = list(numbers)  # pyright: ignore[reportUnknownArgumentType]
    if not values:
        raise error(
            f"{name} is empty.",
            why="each set needs at least one instance number.",
            fix=fix,
        )
    checked: list[int] = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= MAX_INSTANCE_NUMBER
        ):
            raise error(
                f"{name} contains {value!r}.",
                why=f"instance numbers are ints from 1 to {MAX_INSTANCE_NUMBER} "
                "(ioh receives them as C ints).",
                fix=fix,
            )
        checked.append(value)
    repeated = sorted({value for value in checked if checked.count(value) > 1})
    if repeated:
        raise error(
            f"{name} contains {_numbers_text(repeated)} more than once.",
            why="each instance number must appear once.",
            fix=f"remove the repeated numbers from {name}.",
        )
    return tuple(sorted(checked))


def _numbers_text(numbers: Sequence[int]) -> str:
    shown = ", ".join(str(number) for number in numbers[:5])
    return shown if len(numbers) <= 5 else f"{shown}, and {len(numbers) - 5} more"


def _instance(
    function_id: int | None, instance_number: int, dimension: int, seed: int
) -> ProblemInstance:
    """A BBOB instance, or an MA-BBOB instance if ``function_id`` is None."""
    lower_bounds, upper_bounds = _bounds(function_id is None, dimension)
    if function_id is None:
        suite: Suite = "ma-bbob"
        instance_id = f"mabbob_i{instance_number:03d}_d{dimension}"
    else:
        suite = "bbob"
        instance_id = f"f{function_id:02d}_i{instance_number:03d}_d{dimension}"
    return ProblemInstance(
        instance_id=instance_id,
        suite=suite,
        function_id=function_id,
        instance_number=instance_number,
        dimension=dimension,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        seed=seed,
        is_transformed=False,
        original_instance_id=instance_id,
    )


def _ioh_problem(function_id: int | None, instance_number: int, dimension: int) -> Any:
    """Return the ioh problem of BBOB, or of MA-BBOB if ``function_id`` is None.

    It is Any: ioh's stubs do not cover batch calls. ioh registers MA-BBOB under no
    ProblemClass, so ioh.get_problem cannot create it.
    """
    if function_id is None:
        # ioh's stubs do not list ManyAffine.
        problems: Any = ioh.problem
        return problems.ManyAffine(instance=instance_number, n_variables=dimension)
    return ioh.get_problem(
        function_id, instance_number, dimension, ioh.ProblemClass.BBOB
    )


@functools.cache
def _bounds(
    is_ma_bbob: bool, dimension: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The search range of the suite, read from its first instance."""
    bounds = _ioh_problem(None if is_ma_bbob else 1, 1, dimension).bounds
    lower_bounds = np.array(bounds.lb, dtype=np.float64)
    upper_bounds = np.array(bounds.ub, dtype=np.float64)
    lower_bounds.flags.writeable = False
    upper_bounds.flags.writeable = False
    return lower_bounds, upper_bounds


@functools.lru_cache(maxsize=256)
def _transformation(
    seed: int, function_id: int | None, instance_number: int, dimension: int
) -> StrictlyIncreasingTransformation:
    problem = _ioh_problem(function_id, instance_number, dimension)
    lower_bounds, upper_bounds = _bounds(function_id is None, dimension)
    keys = instance_keys(function_id, instance_number, dimension)
    rng = generator("fmax", seed, *keys)
    n_samples = FMAX_SAMPLES_PER_DIMENSION * dimension
    sample = rng.uniform(lower_bounds, upper_bounds, size=(n_samples, dimension))
    f_max = float(np.max(np.asarray(problem(sample), dtype=np.float64)))
    return StrictlyIncreasingTransformation(
        sample_psi(seed, function_id, instance_number, dimension),
        f_min=float(problem.optimum.y),
        f_max=f_max,
    )
