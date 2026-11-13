"""Features of the instances, value-based or rank-based."""

import concurrent.futures
import contextlib
import logging
import math
import os
import pickle
import sys
import traceback
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import cloudpickle  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pandas as pd
import scipy.stats  # pyright: ignore[reportMissingTypeStubs]
from joblib import effective_n_jobs  # pyright: ignore[reportMissingTypeStubs]
from joblib.externals.loky import (  # pyright: ignore[reportMissingTypeStubs]
    get_reusable_executor,  # pyright: ignore[reportUnknownVariableType]
)
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits  # pyright: ignore[reportMissingTypeStubs]

from rubberruler._console import progress
from rubberruler._errors import RubberRulerError, check_int, detail, error
from rubberruler._instances import (
    InstanceObjective,
    ProblemInstance,
    instance_tuple,
    keys_of,
    original_of,
    transformation_of,
)
from rubberruler._performance import check_n_jobs, check_sendable, watch_parent
from rubberruler._seeding import generator

logger = logging.getLogger("rubberruler")

FeatureType = Literal["value", "rank"]
FeatureExtractor = Callable[
    [NDArray[np.float64], NDArray[np.float64]], Mapping[str, float]
]
FEATURE_TYPES: tuple[FeatureType, ...] = ("value", "rank")
MA_BBOB_PRECHECK_INSTANCES = 24
FIX_KEYS = (
    'return a dict with fixed keys; use float("nan") for values you cannot compute.'
)


@dataclass(frozen=True)
class FeatureTable:
    """The features of one feature type, with the instances to warn about."""

    features: pd.DataFrame
    nan_instances: tuple[str, ...]
    constant_instances: tuple[str, ...]


@dataclass(frozen=True)
class Row:
    """The features of one instance, and where they came from."""

    features: dict[str, float]
    instance_id: str
    feature_type: FeatureType


def compute_features(
    feature_extractor: FeatureExtractor,
    instances: Iterable[ProblemInstance],
    feature_type: Literal["value", "rank"],
    *,
    n_samples_per_dimension: int = 50,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Compute the features phi_p of every instance with one feature type.

    For each instance, X is n = n_samples_per_dimension x d points drawn uniformly
    from the search range (the same X for an instance and its transformed version).
    y is the objective at X, preprocessed by ``feature_type``: "value" rescales it
    to [0, 1] by its minimum and maximum, and "rank" rescales its average ranks in
    the same way.
    ``feature_extractor(X, y)`` returns one row of features.

    The instances are divided among ``n_jobs`` worker processes, each of which
    receives the extractor once and uses one thread for NumPy and similar libraries.
    The features are the same, to the last bit, for every n_jobs.

    Args:
        feature_extractor: A function ``(X, y) -> dict[str, float]`` that returns the
            same keys for every instance. Use ``float("nan")`` for values it cannot
            compute.
        instances: The instances, such as ``split`` from ``rr.bbob_instances()`` or
            ``rr.ma_bbob_instances()``.
        feature_type: "value" or "rank".
        n_samples_per_dimension: n divided by d.
        n_jobs: The number of worker processes; -1 uses every CPU, and 1 computes
            the features in this process. Each worker holds its own copy of the
            extractor, so a large model, such as DeepELA's (about 0.5 GB per
            worker), calls for a smaller n_jobs when memory is short.

    Returns:
        A DataFrame with one row per ``instance_id`` and one column per feature, in
        the key order of the first instance. ``attrs["feature_type"]`` holds the
        feature type.
    """
    checked_type = check_feature_type(feature_type)
    selected = instance_tuple(instances, "compute_features")
    n_samples_per_dimension = check_n_samples_per_dimension(n_samples_per_dimension)
    n_jobs = check_n_jobs(n_jobs)
    precheck_extractor(
        feature_extractor, selected, n_samples_per_dimension, n_jobs=n_jobs
    )
    check_sendable(selected, n_jobs)
    with progress("Computing features", len(selected)) as advance:
        table = feature_tables(
            feature_extractor,
            selected,
            (checked_type,),
            n_samples_per_dimension,
            n_jobs=n_jobs,
            advance=advance,
        )[checked_type]
    for line in feature_warnings(table.nan_instances, table.constant_instances):
        logger.warning(line)
    return table.features


@dataclass(frozen=True)
class Computed:
    """One row of features, and whether y was constant on the sample."""

    row: Row
    is_constant: bool


@dataclass(frozen=True)
class ExtractorFailure:
    """A RubberRulerError of the extractor, with its cause, to raise in order."""

    error: RubberRulerError
    cause: BaseException | None


Group = tuple[ProblemInstance, ...]


def feature_tables(
    feature_extractor: FeatureExtractor,
    instances: Sequence[ProblemInstance],
    feature_types: Sequence[FeatureType],
    n_samples_per_dimension: int,
    *,
    n_jobs: int,
    advance: Callable[[int], object],
) -> dict[FeatureType, FeatureTable]:
    """Compute the features of each feature type, without the checks.

    An instance and its transformed version form one group, so that f(X) is
    evaluated once. The groups are computed in chunks, in this process for n_jobs=1
    and in worker processes otherwise, and the chunks are read in their order: the
    first error and the order of the feature names do not depend on n_jobs.
    ``advance`` receives the number of rows of each finished chunk.
    """
    groups = feature_groups(instances)
    rows: dict[FeatureType, list[Row]] = {kind: [] for kind in feature_types}
    constant: dict[FeatureType, list[str]] = {kind: [] for kind in feature_types}
    arguments = (tuple(feature_types), n_samples_per_dimension)
    chunks = computed_chunks(feature_extractor, groups, arguments, n_jobs, advance)
    with contextlib.closing(chunks):
        for computed, failure in chunks:
            for item in computed:
                kind = item.row.feature_type
                if rows[kind]:
                    check_same_keys(item.row, rows[kind][0])
                rows[kind].append(item.row)
                if item.is_constant:
                    constant[kind].append(item.row.instance_id)
            if failure is not None:
                raise failure.error from failure.cause
    order = {instance.instance_id: index for index, instance in enumerate(instances)}
    return {
        kind: table_of(
            kind,
            sorted(rows[kind], key=lambda row: order[row.instance_id]),
            list(rows[kind][0].features),
            sorted(constant[kind], key=order.__getitem__),
        )
        for kind in feature_types
    }


def feature_groups(instances: Sequence[ProblemInstance]) -> list[Group]:
    """The instances grouped by their original instance, in order of appearance."""
    groups: dict[tuple[str, int], list[ProblemInstance]] = {}
    for instance in instances:
        original = original_of(instance)
        groups.setdefault((original.instance_id, original.seed), []).append(instance)
    return [tuple(group) for group in groups.values()]


def table_of(
    feature_type: FeatureType,
    rows: Sequence[Row],
    columns: Sequence[str],
    constant: Sequence[str],
) -> FeatureTable:
    """The DataFrame of the rows, with the instances to warn about."""
    values = np.array(
        [[row.features[column] for column in columns] for row in rows], dtype=np.float64
    ).reshape(len(rows), len(columns))
    index = pd.Index([row.instance_id for row in rows], name="instance_id")
    features = pd.DataFrame(values, index=index, columns=list(columns))
    features.attrs["feature_type"] = feature_type
    nan_instances = tuple(
        row.instance_id
        for row, row_values in zip(rows, values, strict=True)
        if np.isnan(row_values).any()
    )
    return FeatureTable(features, nan_instances, tuple(constant))


def computed_chunks(
    feature_extractor: FeatureExtractor,
    groups: Sequence[Group],
    arguments: tuple[tuple[FeatureType, ...], int],
    n_jobs: int,
    advance: Callable[[int], object],
) -> Generator[tuple[list[Computed], ExtractorFailure | None], None, None]:
    """The results of the chunks of groups, in the order of the groups."""
    if n_jobs == 1:
        with threadpool_limits(limits=1):
            for group in groups:
                computed, failure = compute_chunk(
                    feature_extractor, (group,), *arguments
                )
                advance(len(computed))
                yield computed, failure
        return
    workers = min(cast(int, effective_n_jobs(n_jobs)), len(groups))
    size = max(1, min(CHUNK_GROUPS, -(-len(groups) // workers)))
    chunks = [
        tuple(groups[start : start + size]) for start in range(0, len(groups), size)
    ]
    executor = feature_executor(feature_extractor, workers)
    futures = [
        executor.submit(compute_worker_chunk, chunk, *arguments) for chunk in chunks
    ]
    try:
        finished: dict[int, tuple[list[Computed], ExtractorFailure | None]] = {}
        index = {future: position for position, future in enumerate(futures)}
        next_position = 0
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            advance(len(result[0]))
            finished[index[future]] = result
            while next_position in finished:
                yield finished.pop(next_position)
                next_position += 1
    finally:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, kill_workers=True)


# Groups per chunk sent to a worker: small enough for the progress to move, large
# enough that sending a chunk costs little.
CHUNK_GROUPS = 8

# What a worker received: the extractor, or the exception of unpickling it.
_worker_extractor: list[FeatureExtractor | Exception] = []
# The worker processes start with one thread for the numerical libraries that read
# these variables when they load.
ONE_THREAD = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def feature_executor(feature_extractor: FeatureExtractor, workers: int) -> Any:
    """Worker processes that each receive the extractor once, when they start."""
    reusable = cast(Callable[..., Any], get_reusable_executor)
    return reusable(
        max_workers=workers,
        reuse=False,
        initializer=install_extractor,
        initargs=(cloudpickle.dumps(feature_extractor), os.getpid()),  # pyright: ignore[reportUnknownMemberType]
        env=ONE_THREAD,
    )


def install_extractor(pickled: bytes, parent: int) -> None:
    """Set up a worker: the extractor, one thread for numerical libraries, and the
    watch of the parent process."""
    watch_parent(parent)
    try:
        received = cloudpickle.loads(pickled)  # pyright: ignore[reportUnknownMemberType]
    except Exception as exception:
        received = exception
    _worker_extractor[:] = [received]
    # After loading the extractor, whose modules may load more numerical libraries.
    threadpool_limits(limits=1)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(1)


def compute_worker_chunk(
    groups: Sequence[Group], feature_types: tuple[FeatureType, ...], n: int
) -> tuple[list[Computed], ExtractorFailure | None]:
    """compute_chunk in a worker, with the extractor that it received."""
    received = _worker_extractor[0]
    if isinstance(received, Exception):
        unloaded = error(
            "feature_extractor could not be loaded in a worker process "
            f"({type(received).__name__}: {received}).",
            why="the worker processes import the module of the extractor, unless "
            "it is defined in your script.",
            fix="define the extractor in your script or in a module that Python can "
            "import, or pass n_jobs=1.",
        )
        return [], ExtractorFailure(unloaded, picklable(received))
    computed, failure = compute_chunk(received, groups, feature_types, n)
    if failure is not None:
        failure = ExtractorFailure(failure.error, picklable(failure.cause))
    return computed, failure


def compute_chunk(
    feature_extractor: FeatureExtractor,
    groups: Sequence[Group],
    feature_types: tuple[FeatureType, ...],
    n_samples_per_dimension: int,
) -> tuple[list[Computed], ExtractorFailure | None]:
    """The rows of the groups; stop at the first error of the extractor and return
    it with the rows before it. The caller limits the threads to one.
    """
    computed: list[Computed] = []
    for group in groups:
        raw: dict[tuple[str, int], NDArray[np.float64]] = {}
        for instance in group:
            for feature_type in feature_types:
                try:
                    sample, y, is_constant = feature_sample(
                        instance, feature_type, n_samples_per_dimension, raw
                    )
                    row = extract(feature_extractor, sample, y, instance, feature_type)
                except RubberRulerError as exception:
                    return computed, ExtractorFailure(exception, exception.__cause__)
                computed.append(Computed(row, is_constant))
    return computed, None


def picklable(cause: BaseException | None) -> BaseException | None:
    """The cause with its traceback as a note, or a RuntimeError with its text if
    it cannot be pickled, so that it reaches the parent process."""
    if cause is None:
        return None
    text = "".join(traceback.format_exception(cause)).rstrip()
    try:
        pickle.dumps(cause)
    except Exception:
        cause = RuntimeError(f"{type(cause).__name__}: {cause}")
    cause.add_note(f"Raised in a worker process:\n{text}")
    return cause


def feature_sample(
    instance: ProblemInstance,
    feature_type: FeatureType,
    n_samples_per_dimension: int,
    raw_values: dict[tuple[str, int], NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
    """Return X, the preprocessed y, and whether y was constant.

    X depends only on the master seed and the instance keys (the suite, the function
    on BBOB, the instance number, and the dimension; on the suite "objectives", the
    name and the dimension), so an instance and its transformed version share it, and
    so does every split of the same objectives. f(X) is evaluated once per original
    instance and kept in ``raw_values``; a transformed instance uses g(f(X)).
    """
    dimension = instance.dimension
    rng = generator("features", instance.seed, *keys_of(instance))
    sample = rng.uniform(
        instance.lower_bounds,
        instance.upper_bounds,
        size=(n_samples_per_dimension * dimension, dimension),
    )
    original = original_of(instance)
    key = (original.instance_id, original.seed)
    if key not in raw_values:
        raw_values[key] = InstanceObjective(original).many(sample)
    y_raw = raw_values[key]
    transformation = transformation_of(instance)
    if transformation is not None:
        y_raw = transformation(y_raw)
    y, is_constant = preprocess(y_raw, feature_type)
    return sample, y, is_constant


def preprocess(
    y_raw: NDArray[np.float64], feature_type: FeatureType
) -> tuple[NDArray[np.float64], bool]:
    """Rescale y ("value") or the average ranks of y ("rank") by min and max.

    The two feature types differ only in whether y or its ranks are rescaled.
    Returns the values and whether y was constant, in which case both give zeros.
    """
    if feature_type == "rank":
        y_raw = scipy.stats.rankdata(  # pyright: ignore[reportUnknownMemberType]
            y_raw, method="average"
        )
    return min_max(y_raw)


def min_max(values: NDArray[np.float64]) -> tuple[NDArray[np.float64], bool]:
    """Rescale the values to [0, 1] by their minimum and maximum; zeros if they are
    all equal. Returns the values and whether they were all equal."""
    low = float(np.min(values))
    high = float(np.max(values))
    if high == low:
        return np.zeros_like(values), True
    return (values - low) / (high - low), False


def extract(
    feature_extractor: FeatureExtractor,
    sample: NDArray[np.float64],
    y: NDArray[np.float64],
    instance: ProblemInstance,
    feature_type: FeatureType,
) -> Row:
    """Call the extractor and check its dict; stop with a RubberRulerError if wrong."""
    where = f"{instance.instance_id} ({feature_type})"
    try:
        result: object = feature_extractor(sample, y)
    except Exception as exception:
        raise error(
            f"feature_extractor raised {type(exception).__name__} on {where}: "
            f"{detail(exception)}",
            why="every instance needs its features.",
            fix='make the extractor work on this sample; use float("nan") for '
            "values it cannot compute. The original exception is chained to this "
            "error, with its traceback.",
        ) from exception
    if not isinstance(result, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise error(
            f"feature_extractor returned a {type(result).__name__} on {where}.",
            why="feature_extractor must return a dict of feature names and values.",
            fix='return a dict such as {"ela_meta.lin_simple.adj_r2": 0.93}; wrap '
            "another form in one line, such as lambda X, y: dict(my_features(X, y)) "
            "for a pandas Series or dict(zip(names, values)) for an array.",
        )
    values: dict[str, float] = {}
    for key, value in cast(Mapping[object, object], result).items():
        if not isinstance(key, str):
            raise error(
                f"feature_extractor returned the key {key!r} on {where}.",
                why="feature names must be strings.",
                fix="use str keys, such as str(key).",
            )
        try:
            number = float(value)  # pyright: ignore[reportArgumentType]
        except (TypeError, ValueError) as exception:
            raise error(
                f'feature_extractor returned {value!r} for "{key}" on {where}.',
                why="every feature value must convert with float().",
                fix='return a number, or float("nan") for a value you cannot compute.',
            ) from exception
        # Infinity counts as a value that could not be computed.
        values[key] = math.nan if math.isinf(number) else number
    return Row(values, instance.instance_id, feature_type)


def check_same_keys(row: Row, first: Row) -> None:
    """Stop if ``row`` has other feature names than ``first``."""
    if set(row.features) == set(first.features):
        return
    where = f"{row.instance_id} ({row.feature_type})"
    first_where = f"{first.instance_id} ({first.feature_type})"
    if len(row.features) != len(first.features):
        what = (
            f"feature_extractor returned {len(row.features)} features on {where} "
            f"but {len(first.features)} on {first_where}."
        )
    else:
        missing = sorted(set(first.features) - set(row.features))
        new = sorted(set(row.features) - set(first.features))
        what = (
            f"feature_extractor returned other features on {where} than on "
            f"{first_where}: {_names(missing)} missing and {_names(new)} new."
        )
    raise error(what, why="every instance must give the same features.", fix=FIX_KEYS)


def precheck_extractor(
    feature_extractor: FeatureExtractor,
    instances: Sequence[ProblemInstance],
    n_samples_per_dimension: int,
    *,
    n_jobs: int,
) -> tuple[str, ...]:
    """Stop if the extractor fails on the first instance of a function; return the
    feature names.

    On the first of the instances of each BBOB function and on the first 24 MA-BBOB
    instances, it must work for both feature types, return a Mapping with str keys
    and values that convert with float(), and give the same keys everywhere. With
    n_jobs other than 1 it must also go through cloudpickle. It runs with one thread
    for numerical libraries, as the features do.
    """
    if not callable(feature_extractor):
        raise error(
            f"feature_extractor is a {type(feature_extractor).__name__}, "
            "not a function.",
            why="rubberruler calls feature_extractor(X, y) for every instance.",
            fix="pass a function that returns a dict of features, such as "
            "lambda X, y: {'mean': float(y.mean())}.",
        )
    raw: dict[tuple[str, int], NDArray[np.float64]] = {}
    rows: list[Row] = []
    with threadpool_limits(limits=1):
        for instance in precheck_instances(instances):
            for feature_type in FEATURE_TYPES:
                sample, y, _ = feature_sample(
                    instance, feature_type, n_samples_per_dimension, raw
                )
                row = extract(feature_extractor, sample, y, instance, feature_type)
                if rows:
                    check_same_keys(row, rows[0])
                rows.append(row)
    if n_jobs != 1:
        try:
            cloudpickle.dumps(feature_extractor)  # pyright: ignore[reportUnknownMemberType]
        except Exception as exception:
            raise error(
                "feature_extractor cannot be sent to worker processes "
                f"({type(exception).__name__}: {exception}).",
                why=f"n_jobs={n_jobs} computes the features in other processes, "
                "which needs cloudpickle.",
                fix="keep open files, locks, and connections out of the extractor, "
                "or pass n_jobs=1.",
            ) from exception
    return tuple(rows[0].features)


def precheck_instances(
    instances: Sequence[ProblemInstance],
) -> list[ProblemInstance]:
    """The first instance of each BBOB function and the first 24 MA-BBOB instances,
    in the order of instances.

    MA-BBOB has no function to take the first instance of, so its first 24
    instances give as many calls as the 24 functions of BBOB.
    """
    chosen: list[ProblemInstance] = []
    functions: set[int] = set()
    ma_bbob = 0
    for instance in instances:
        if instance.function_id is None:
            if ma_bbob < MA_BBOB_PRECHECK_INSTANCES:
                ma_bbob += 1
                chosen.append(instance)
        elif instance.function_id not in functions:
            functions.add(instance.function_id)
            chosen.append(instance)
    return chosen


def feature_warnings(
    nan_instances: Sequence[str], constant_instances: Sequence[str]
) -> list[str]:
    """The warning lines about features (singular for one instance)."""
    lines: list[str] = []
    if nan_instances:
        lines.append(f"{_instances_have(len(nan_instances))} NaN features")
    if constant_instances:
        lines.append(
            f"{_instances_have(len(constant_instances))} a constant objective on the "
            "feature sample"
        )
    return lines


def _instances_have(count: int) -> str:
    return "1 instance has" if count == 1 else f"{count} instances have"


def check_feature_type(feature_type: object) -> FeatureType:
    if feature_type == "value":
        return "value"
    if feature_type == "rank":
        return "rank"
    raise error(
        f"feature_type={feature_type!r} is not allowed.",
        why='feature_type must be "value" or "rank".',
        fix='pass feature_type="value" or feature_type="rank".',
    )


def check_n_samples_per_dimension(value: object) -> int:
    return check_int(
        "n_samples_per_dimension",
        value,
        minimum=1,
        example="an int such as n_samples_per_dimension=50",
    )


def _names(names: Sequence[str]) -> str:
    if not names:
        return "none"
    shown = ", ".join(f'"{name}"' for name in names[:3])
    return shown if len(names) <= 3 else f"{shown}, and {len(names) - 3} more"
