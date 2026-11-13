"""Tables of performance and features that the user gives to run_test."""

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import pandas as pd

from rubberruler._errors import error
from rubberruler._features import FEATURE_TYPES, FeatureTable, FeatureType
from rubberruler._instances import ProblemInstance
from rubberruler._prediction import performance_matrix

FIX_FORM = (
    "pass a table of the form of result.{name}: one row per instance_id of the "
    "split, {columns}."
)


def given_performance(
    performance: object,
    instances: Sequence[ProblemInstance],
    algorithms: Sequence[str] | None,
) -> pd.DataFrame | None:
    """The given J of instances of the split, in the order of the algorithms.

    Stop if it is not a DataFrame, has rows of other instances, other algorithms
    than the portfolio, or values that are not J (as fit_predictor does).
    """
    if performance is None:
        return None
    fix = FIX_FORM.format(name="performance", columns="one column per algorithm")
    if not isinstance(performance, pd.DataFrame):
        raise error(
            f"run_test received performance of type {type(performance).__name__}.",
            why="performance is a table of J, as result.performance.",
            fix=fix,
        )
    present = _rows_of_the_split(performance, instances, "performance", fix)
    names = [str(name) for name in performance.columns]
    if algorithms is None:
        if len(names) < 2:
            raise error(
                f"run_test received performance with {len(names)} algorithm"
                f"{'' if len(names) == 1 else 's'} and no portfolio.",
                why="J compares each algorithm with the others, so it needs at least "
                "2.",
                fix=fix,
            )
        algorithms = names
    elif set(names) != set(algorithms) or len(names) != len(algorithms):
        raise error(
            f"run_test received performance for {_shown(names)}, but the portfolio "
            f"has {_shown(algorithms)}.",
            why="J of an algorithm depends on the other algorithms, so the table "
            "must have the algorithms of the portfolio.",
            fix="pass the performance of the algorithms of the portfolio, or a "
            "portfolio with the algorithms of the table.",
        )
    renamed = performance.rename(columns=str)[list(algorithms)]
    _, values = performance_matrix(renamed, present, "run_test")
    index = pd.Index([p.instance_id for p in present], name="instance_id")
    return pd.DataFrame(values, index=index, columns=list(algorithms))


def given_features(
    features: object, instances: Sequence[ProblemInstance]
) -> dict[FeatureType, pd.DataFrame]:
    """The given features of instances of the split, per feature type, as float64
    with infinity as NaN; stop if their form differs from result.features."""
    if features is None:
        return {}
    fix = FIX_FORM.format(
        name="features",
        columns='as {"value": table, "rank": table}, with one column per feature',
    )
    if not isinstance(features, Mapping):
        raise error(
            f"run_test received features of type {type(features).__name__}.",
            why="features maps each feature type to a table, as result.features.",
            fix=fix,
        )
    given: dict[FeatureType, pd.DataFrame] = {}
    for key, table in cast(Mapping[object, object], features).items():
        if key not in FEATURE_TYPES:
            raise error(
                f"run_test received features with the key {key!r}.",
                why='the keys of features are the feature types, "value" and "rank".',
                fix=fix,
            )
        feature_type = cast(FeatureType, key)
        if not isinstance(table, pd.DataFrame):
            raise error(
                f'run_test received features["{feature_type}"] of type '
                f"{type(table).__name__}.",
                why="each feature type has a table, as result.features.",
                fix=fix,
            )
        stated = table.attrs.get("feature_type", feature_type)
        if stated != feature_type:
            raise error(
                f'run_test received features["{feature_type}"] with '
                f'attrs["feature_type"] = {stated!r}.',
                why="value-based and rank-based features must not be mixed up.",
                fix=fix,
            )
        name = f'features["{feature_type}"]'
        present = _rows_of_the_split(table, instances, name, fix)
        columns = [str(column) for column in table.columns]
        if not columns or len(set(columns)) != len(columns):
            raise error(
                f"run_test received {name} with the columns {_shown(columns)}.",
                why="each feature has one column with its own name.",
                fix=fix,
            )
        rows = table.loc[[p.instance_id for p in present]]
        try:
            values = rows.to_numpy(dtype=np.float64, copy=True)
        except (TypeError, ValueError):
            raise error(
                f"run_test received {name} with values that are not numbers.",
                why="the predictor receives the features as numbers.",
                fix=fix,
            ) from None
        values[np.isinf(values)] = math.nan
        index = pd.Index([p.instance_id for p in present], name="instance_id")
        converted = pd.DataFrame(values, index=index, columns=columns)
        converted.attrs["feature_type"] = feature_type
        given[feature_type] = converted
    return given


def _rows_of_the_split(
    table: pd.DataFrame,
    instances: Sequence[ProblemInstance],
    name: str,
    fix: str,
) -> list[ProblemInstance]:
    """The instances of the split that the table has rows for, in their order.

    Rows of other instances are not used, so that the tables of one split of your
    objectives serve another. Stop if no row is an instance of the split, and
    on repeated rows.
    """
    known = {instance.instance_id for instance in instances}
    labels = [str(label) for label in table.index]
    if labels and not known.intersection(labels):
        raise error(
            f"run_test received {name} with rows for {_shown(labels)}, none of which "
            "is an instance of the split.",
            why="the rows of a given table are used for the instances with the same "
            "instance_id.",
            fix=fix,
        )
    repeated = sorted(label for label, count in Counter(labels).items() if count > 1)
    if repeated:
        raise error(
            f"run_test received {name} with more than one row for {_shown(repeated)}.",
            why="each instance has one row.",
            fix=fix,
        )
    present = set(labels)
    return [instance for instance in instances if instance.instance_id in present]


def _shown(names: Sequence[str]) -> str:
    shown = ", ".join(f'"{name}"' for name in names[:3])
    return shown if len(names) <= 3 else f"{shown}, and {len(names) - 3} more"


def check_missing_inputs(
    algorithms: Mapping[str, object] | None,
    feature_extractor: object,
    to_measure: Sequence[ProblemInstance],
    to_compute: Mapping[FeatureType, Sequence[ProblemInstance]],
    total: int,
) -> None:
    """Stop if rows are missing that only a portfolio or an extractor could give."""
    if algorithms is None and to_measure:
        raise error(
            f"portfolio is None, but performance has no rows for {len(to_measure):,} "
            f"of the {total:,} instances.",
            why="rubberruler measures the missing performance with the portfolio.",
            fix="pass the portfolio, or performance for every instance of the split.",
        )
    missing = {kind: len(selected) for kind, selected in to_compute.items() if selected}
    if feature_extractor is None and missing:
        kind, count = next(iter(missing.items()))
        raise error(
            f'feature_extractor is None, but features["{kind}"] has no rows for '
            f"{count:,} of the {total:,} instances.",
            why="rubberruler computes the missing features with the extractor.",
            fix="pass the extractor, or features of both feature types for every "
            "instance of the split.",
        )


def check_given_columns(
    names: Sequence[str],
    given: Mapping[FeatureType, pd.DataFrame],
    to_compute: Mapping[FeatureType, Sequence[ProblemInstance]],
) -> None:
    """Stop if the extractor gives other features than a table it completes."""
    for kind, table in given.items():
        if to_compute[kind] and set(names) != set(table.columns):
            raise error(
                f"feature_extractor returns {_shown(list(names))}, but "
                f'features["{kind}"] has the columns {_shown(list(table.columns))}.',
                why="the rows that rubberruler computes join the given rows, so they "
                "need the same features.",
                fix="pass the tables of the same extractor, or leave out features.",
            )


def joined_performance(
    given: pd.DataFrame | None,
    measured: pd.DataFrame | None,
    instances: Sequence[ProblemInstance],
) -> pd.DataFrame:
    """J of every instance: the given rows and the measured rows, in instance order."""
    parts = [part for part in (given, measured) if part is not None and len(part)]
    columns = list(parts[0].columns)
    joined = pd.concat([part[columns] for part in parts])
    ids = [instance.instance_id for instance in instances]
    return joined.loc[ids].rename_axis("instance_id")


def joined_features(
    feature_type: FeatureType,
    given: pd.DataFrame | None,
    computed: FeatureTable | None,
    instances: Sequence[ProblemInstance],
) -> FeatureTable:
    """The features of every instance: the given rows and the computed rows, in
    instance order, in the column order of the given table if there is one."""
    parts = [part for part in (given, None if computed is None else computed.features)]
    present = [part for part in parts if part is not None]
    columns = list(present[0].columns)
    ids = [instance.instance_id for instance in instances]
    joined = pd.concat([part[columns] for part in present]).loc[ids]
    joined = joined.rename_axis("instance_id")
    joined.attrs["feature_type"] = feature_type
    values = joined.to_numpy(dtype=np.float64)
    nan_instances = tuple(
        str(label)
        for label, row in zip(joined.index, values, strict=True)
        if np.isnan(row).any()
    )
    constant = () if computed is None else computed.constant_instances
    return FeatureTable(joined, nan_instances, constant)


def given_items(
    performance: pd.DataFrame | None,
    features: Mapping[FeatureType, pd.DataFrame],
    total: int,
) -> list[str]:
    """The items of the card that state which rows came from given tables."""
    items: list[str] = []
    if performance is not None:
        items.append(f"performance given for {len(performance)} of {total} instances")
    for kind in FEATURE_TYPES:
        if kind in features:
            items.append(
                f"{kind} features given for {len(features[kind])} of {total} instances"
            )
    return items
