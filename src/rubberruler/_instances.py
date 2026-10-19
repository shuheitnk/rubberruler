"""BBOB and MA-BBOB problem instances and their transformed versions."""

import functools
import itertools
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from rubberruler._errors import check_int, error, three_lines
from rubberruler._seeding import generator, instance_keys
from rubberruler._transformations import StrictlyIncreasingTransformation, sample_psi

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

Suite = Literal["bbob", "ma-bbob"]

FUNCTION_IDS = tuple(range(1, 25))
# The default training and test instance numbers of each suite.
DEFAULT_INSTANCE_NUMBERS: dict[Suite, tuple[range, range]] = {
    "bbob": (range(11, 111), range(1, 11)),
    "ma-bbob": (range(1, 2401), range(2401, 2641)),
}
FMAX_SAMPLES = 10_000
TRANSFORMED_SUFFIX = "~transformed"
MAX_SEED = 2**32 - 1
# ioh receives the instance number as a C int.
MAX_INSTANCE_NUMBER = 2**31 - 1


@dataclass(frozen=True)
class ProblemInstance:
    """One BBOB or MA-BBOB problem instance p, or its transformed version.

    Read-only and not callable: only the Problem that an algorithm receives can
    evaluate the objective.

    Attributes:
        instance_id: ``f"f{function_id:02d}_i{instance_number:03d}_d{dimension}"``
            on BBOB and ``f"mabbob_i{instance_number:03d}_d{dimension}"`` on
            MA-BBOB, followed by ``~transformed`` on a transformed instance.
        suite: The problem suite, "bbob" or "ma-bbob".
        function_id: The BBOB function, from 1 to 24, or None on MA-BBOB, which
            has no function ID (an MA-BBOB instance combines several BBOB
            functions).
        instance_number: The instance number in the suite.
        dimension: d.
        lower_bounds: The lower end of the search range, an array of shape (d,).
        upper_bounds: The upper end of the search range, an array of shape (d,).
        seed: The master seed.
        is_transformed: Whether the objective is g(f(x)) rather than f(x).
        original_instance_id: The id before the transformation (the instance's own
            id if it is not transformed).
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


@dataclass(frozen=True)
class InstanceSplit:
    """The training, test, and transformed test instances (P_train, P_test, P~_test).

    Iterating over it gives every instance: ``train``, then ``test``, then
    ``transformed_test``. So ``measure_performance(portfolio, split)`` measures all.
    """

    train: tuple[ProblemInstance, ...]
    test: tuple[ProblemInstance, ...]
    transformed_test: tuple[ProblemInstance, ...]

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

    g rescales f to [0, 1] by its optimum and the largest of 10,000 random values,
    and stretches it by a random strictly increasing curve. It depends only on the
    master seed, the suite, the function (on BBOB), the instance number, and the
    dimension, so the same instance always receives the same g. The search space
    does not change.
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
    return replace(
        instance,
        instance_id=instance.instance_id + TRANSFORMED_SUFFIX,
        is_transformed=True,
    )


def transformation_of(
    instance: ProblemInstance,
) -> StrictlyIncreasingTransformation | None:
    """Return g of a transformed instance, or None for an original instance."""
    if not instance.is_transformed:
        return None
    return _transformation(
        instance.seed,
        instance.function_id,
        instance.instance_number,
        instance.dimension,
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
        self._problem = _ioh_problem(
            instance.function_id, instance.instance_number, instance.dimension
        )
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
    raise error(
        f"suite={suite!r} is not allowed.",
        why='suite must be "bbob" or "ma-bbob".',
        fix='pass suite="bbob" or suite="ma-bbob".',
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
                "rr.bbob_instances() or rr.ma_bbob_instances().",
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
    sample = rng.uniform(lower_bounds, upper_bounds, size=(FMAX_SAMPLES, dimension))
    f_max = float(np.max(np.asarray(problem(sample), dtype=np.float64)))
    return StrictlyIncreasingTransformation(
        sample_psi(seed, function_id, instance_number, dimension),
        f_min=float(problem.optimum.y),
        f_max=f_max,
    )
