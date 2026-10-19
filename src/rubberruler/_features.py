"""Features of the instances, value-based or rank-based."""

import logging
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd
import scipy.stats  # pyright: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray

from rubberruler._console import progress
from rubberruler._errors import check_int, error
from rubberruler._instances import (
    InstanceObjective,
    ProblemInstance,
    instance_tuple,
    original_of,
    transformation_of,
)
from rubberruler._seeding import generator, instance_keys

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
) -> pd.DataFrame:
    """Compute the features phi_p of every instance with one feature type.

    For each instance, X is n = n_samples_per_dimension x d points drawn uniformly
    from the search range (the same X for an instance and its transformed version).
    y is the objective at X, preprocessed by ``feature_type``: "value" rescales it
    to [0, 1] by its minimum and maximum, and "rank" divides its average ranks by n.
    ``feature_extractor(X, y)`` returns one row of features.

    Args:
        feature_extractor: A function ``(X, y) -> dict[str, float]`` that returns the
            same keys for every instance. Use ``float("nan")`` for values it cannot
            compute.
        instances: The instances, such as ``split`` from ``rr.bbob_instances()`` or
            ``rr.ma_bbob_instances()``.
        feature_type: "value" or "rank".
        n_samples_per_dimension: n divided by d.

    Returns:
        A DataFrame with one row per ``instance_id`` and one column per feature, in
        the key order of the first instance. ``attrs["feature_type"]`` holds the
        feature type.
    """
    checked_type = check_feature_type(feature_type)
    selected = instance_tuple(instances, "compute_features")
    n_samples_per_dimension = check_n_samples_per_dimension(n_samples_per_dimension)
    precheck_extractor(feature_extractor, selected, n_samples_per_dimension)
    with progress("Computing features", len(selected)) as advance:
        table = feature_table(
            feature_extractor,
            selected,
            checked_type,
            n_samples_per_dimension,
            advance=advance,
        )
    for line in feature_warnings(table.nan_instances, table.constant_instances):
        logger.warning(line)
    return table.features


def feature_table(
    feature_extractor: FeatureExtractor,
    instances: Sequence[ProblemInstance],
    feature_type: FeatureType,
    n_samples_per_dimension: int,
    *,
    advance: Callable[[int], object],
    raw_values: dict[tuple[str, int], NDArray[np.float64]] | None = None,
) -> FeatureTable:
    """Compute the features without the checks of compute_features.

    ``advance(1)`` is called after each instance. ``raw_values`` keeps f(X) of each
    original instance, so that f(X) is evaluated once even across feature types.
    """
    raw = {} if raw_values is None else raw_values
    rows: list[Row] = []
    constant: list[str] = []
    for instance in instances:
        sample, y, is_constant = feature_sample(
            instance, feature_type, n_samples_per_dimension, raw
        )
        if is_constant:
            constant.append(instance.instance_id)
        row = extract(feature_extractor, sample, y, instance, feature_type)
        if rows:
            check_same_keys(row, rows[0])
        rows.append(row)
        advance(1)
    columns = list(rows[0].features)
    values = np.array(
        [[row.features[column] for column in columns] for row in rows], dtype=np.float64
    ).reshape(len(rows), len(columns))
    index = pd.Index([row.instance_id for row in rows], name="instance_id")
    features = pd.DataFrame(values, index=index, columns=columns)
    features.attrs["feature_type"] = feature_type
    nan_instances = tuple(
        row.instance_id
        for row, row_values in zip(rows, values, strict=True)
        if np.isnan(row_values).any()
    )
    return FeatureTable(features, nan_instances, tuple(constant))


def feature_sample(
    instance: ProblemInstance,
    feature_type: FeatureType,
    n_samples_per_dimension: int,
    raw_values: dict[tuple[str, int], NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
    """Return X, the preprocessed y, and whether y was constant.

    X depends only on the master seed and the instance keys (the suite, the function
    on BBOB, the instance number, and the dimension), so an instance and its
    transformed version share it. f(X) is evaluated once per original instance and
    kept in ``raw_values``; a transformed instance uses g(f(X)).
    """
    dimension = instance.dimension
    keys = instance_keys(instance.function_id, instance.instance_number, dimension)
    rng = generator("features", instance.seed, *keys)
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
    """Rescale y by min and max ("value") or by average rank / n ("rank").

    Returns the values and whether y was constant, in which case "value" gives zeros.
    """
    if feature_type == "rank":
        ranks: NDArray[np.float64] = scipy.stats.rankdata(  # pyright: ignore[reportUnknownMemberType]
            y_raw, method="average"
        )
        return ranks / y_raw.size, False
    low = float(np.min(y_raw))
    high = float(np.max(y_raw))
    if high == low:
        return np.zeros_like(y_raw), True
    return (y_raw - low) / (high - low), False


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
            f"feature_extractor raised {type(exception).__name__} on {where}.",
            why="every instance needs its features.",
            fix='make the extractor work on this sample; use float("nan") for '
            "values it cannot compute. The original exception is shown above.",
        ) from exception
    if not isinstance(result, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise error(
            f"feature_extractor returned a {type(result).__name__} on {where}.",
            why="feature_extractor must return a dict of feature names and values.",
            fix='return a dict such as {"ela_meta.lin_simple.adj_r2": 0.93}.',
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
) -> None:
    """Stop if the extractor fails on the first instance of a function.

    On the first of the instances of each BBOB function and on the first 24 MA-BBOB
    instances, it must work for both feature types, return a Mapping with str keys
    and values that convert with float(), and give the same keys everywhere.
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
    for instance in precheck_instances(instances):
        for feature_type in FEATURE_TYPES:
            sample, y, _ = feature_sample(
                instance, feature_type, n_samples_per_dimension, raw
            )
            row = extract(feature_extractor, sample, y, instance, feature_type)
            if rows:
                check_same_keys(row, rows[0])
            rows.append(row)


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
