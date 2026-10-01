"""Performance predictors and their R^2."""

import copy
import logging
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from rubberruler._errors import error
from rubberruler._features import FeatureType
from rubberruler._instances import ProblemInstance, instance_tuple

logger = logging.getLogger("rubberruler")

RANDOM_STATE_WARNING = (
    "predictor has random_state=None, so R^2 may change between calls"
)
UNDEFINED_R2 = "R^2 is undefined because all performance values are equal; returned nan"
# Two feature values a and b differ only by rounding when
# math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL).
REL_TOL = 1e-9
ABS_TOL = 1e-12


@dataclass(frozen=True, repr=False)
class FittedPredictor:
    """predictor_tau: one fitted copy of the predictor per algorithm.

    It remembers the feature type and the feature names it was fitted on, so that
    ``r2_score`` can stop when it gets other features.
    """

    feature_type: FeatureType
    feature_names: tuple[str, ...]
    algorithms: tuple[str, ...]
    predictors: tuple[Any, ...]
    # Per column, the training values kept by merge_rounding.
    training_values: tuple[NDArray[np.float64], ...]

    def __repr__(self) -> str:
        return (
            f"FittedPredictor(feature_type={self.feature_type!r}, "
            f"{len(self.algorithms)} algorithms, {len(self.feature_names)} features)"
        )


def fit_predictor(
    predictor: object,
    features: pd.DataFrame,
    performance: pd.DataFrame,
    *,
    instances: Iterable[ProblemInstance],
) -> FittedPredictor:
    """Fit predictor_tau: a copy of ``predictor`` for each algorithm a.

    Each copy learns J(., a) from the features of ``instances`` (usually
    ``split.train``). The copies are made with ``sklearn.base.clone`` if the
    predictor has ``get_params``, and with ``copy.deepcopy`` otherwise, so
    ``predictor`` itself is not changed. Feature values that differ only by
    rounding (``math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)``) are made equal
    first, here and in ``rr.r2_score``, so that the predictor cannot split them
    apart.

    Args:
        predictor: An unfitted object with ``fit(X, y)`` and ``predict(X)``, such as
            a scikit-learn regressor. X is a float64 array (m, k), y is (m,).
        features: Features from ``rr.compute_features`` (one feature type).
        performance: J from ``rr.measure_performance``.
        instances: The instances whose rows are used.
    """
    precheck_predictor(predictor)
    feature_type = feature_type_of(features, "fit_predictor")
    selected = instance_tuple(instances, "fit_predictor")
    matrix = feature_matrix(features, selected, "fit_predictor")
    algorithms, targets = performance_matrix(performance, selected, "fit_predictor")
    if has_unset_random_state(predictor):
        logger.warning(RANDOM_STATE_WARNING)
    return fit(
        predictor, feature_type, list(features.columns), algorithms, matrix, targets
    )


def fit(
    predictor: object,
    feature_type: FeatureType,
    feature_names: Sequence[str],
    algorithms: Sequence[str],
    matrix: NDArray[np.float64],
    targets: NDArray[np.float64],
) -> FittedPredictor:
    """Fit one copy per algorithm, without the checks of fit_predictor."""
    training_values = merge_rounding(matrix)
    matrix = snap(matrix, training_values)
    fitted: list[Any] = []
    for column, algorithm in enumerate(algorithms):
        model = copy_of(predictor)
        try:
            model.fit(matrix.copy(), targets[:, column].copy())
        except Exception as exception:
            where = f'algorithm "{algorithm}" ({feature_type} features)'
            raise predictor_error("fit", exception, where, matrix) from exception
        fitted.append(model)
    return FittedPredictor(
        feature_type,
        tuple(feature_names),
        tuple(algorithms),
        tuple(fitted),
        training_values,
    )


def merge_rounding(
    matrix: NDArray[np.float64],
) -> tuple[NDArray[np.float64], ...]:
    """Per column, the training values that are kept when rounding is merged.

    Going up from the smallest value, a value is kept unless it differs only by
    rounding from the last kept value. Whether two values do depends on those two
    alone, so one huge value does not merge the others. NaN and infinity are left out.
    """
    columns: list[NDArray[np.float64]] = []
    for column in matrix.T:
        kept: list[float] = []
        for value in np.sort(column[np.isfinite(column)]).tolist():
            if not kept or not math.isclose(
                value, kept[-1], rel_tol=REL_TOL, abs_tol=ABS_TOL
            ):
                kept.append(value)
        columns.append(np.array(kept, dtype=np.float64))
    return tuple(columns)


def snap(
    matrix: NDArray[np.float64], training_values: Sequence[NDArray[np.float64]]
) -> NDArray[np.float64]:
    """A copy in which each value becomes the nearest kept training value, if close.

    Close is as in merge_rounding. NaN and infinity stay as they are.
    """
    snapped = matrix.copy()
    for index, kept in enumerate(training_values):
        if kept.size == 0:
            continue
        values = snapped[:, index]
        present = np.isfinite(values)
        given = values[present]
        position = np.searchsorted(kept, given)
        below = kept[np.maximum(position - 1, 0)]
        above = kept[np.minimum(position, kept.size - 1)]
        nearest = np.where(given - below <= above - given, below, above)
        # math.isclose on arrays: |a - b| <= max(REL_TOL * max(|a|, |b|), ABS_TOL).
        tolerance = np.maximum(
            REL_TOL * np.maximum(np.abs(given), np.abs(nearest)), ABS_TOL
        )
        values[present] = np.where(np.abs(given - nearest) <= tolerance, nearest, given)
    return snapped


def r2_score(
    fitted: FittedPredictor,
    features: pd.DataFrame,
    performance: pd.DataFrame,
    *,
    instances: Iterable[ProblemInstance],
) -> float:
    """R^2_tau(Q): how well the fitted predictor predicts J on the instances Q.

    R^2 = 1 - sum (J - J_hat)^2 / sum (J - mean J)^2, with the sums and the mean over
    every pair of an instance in Q and an algorithm. The predictions J_hat are clipped
    to [0, 1]. If every J is equal, R^2 is undefined and nan is returned.

    Args:
        fitted: The result of ``rr.fit_predictor``.
        features: Features of the same feature type and columns as in the fit.
        performance: J from ``rr.measure_performance``.
        instances: Q, such as ``split.test`` or ``split.transformed_test``.
    """
    if not isinstance(fitted, FittedPredictor):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise error(
            f"r2_score got a {type(fitted).__name__} as fitted.",
            why="fitted must be a predictor fitted by rr.fit_predictor.",
            fix="pass the result of rr.fit_predictor(...).",
        )
    feature_type = feature_type_of(features, "r2_score")
    check_same_features(fitted, feature_type, list(features.columns))
    selected = instance_tuple(instances, "r2_score")
    matrix = feature_matrix(features, selected, "r2_score")
    algorithms, targets = performance_matrix(performance, selected, "r2_score")
    if algorithms != list(fitted.algorithms):
        raise error(
            f"r2_score got performance for {algorithms}, but the predictor was "
            f"fitted for {list(fitted.algorithms)}.",
            why="each algorithm has its own fitted predictor.",
            fix="pass the performance DataFrame that was given to fit_predictor.",
        )
    return score(fitted, matrix, targets)


def score(
    fitted: FittedPredictor, matrix: NDArray[np.float64], targets: NDArray[np.float64]
) -> float:
    """R^2 of the clipped predictions, without the checks of r2_score."""
    matrix = snap(matrix, fitted.training_values)
    predicted = np.empty_like(targets)
    for column, (algorithm, model) in enumerate(
        zip(fitted.algorithms, fitted.predictors, strict=True)
    ):
        where = f'algorithm "{algorithm}" ({fitted.feature_type} features)'
        try:
            output: object = model.predict(matrix.copy())
        except Exception as exception:
            raise predictor_error("predict", exception, where, matrix) from exception
        predicted[:, column] = predictions_of(output, matrix.shape[0], where)
    return coefficient_of_determination(targets, np.clip(predicted, 0.0, 1.0))


def coefficient_of_determination(
    actual: NDArray[np.float64], predicted: NDArray[np.float64]
) -> float:
    """1 - sum (J - J_hat)^2 / sum (J - mean J)^2 over all entries; nan if J is equal.

    Equal values are checked directly: their computed mean can differ from them by
    rounding, which would make the denominator tiny instead of 0.
    """
    if np.all(actual == actual.flat[0]):
        logger.warning(UNDEFINED_R2)
        return float("nan")
    residual = float(np.sum((actual - predicted) ** 2))
    total = float(np.sum((actual - np.mean(actual)) ** 2))
    return 1.0 - residual / total


def precheck_predictor(predictor: object) -> None:
    """Stop if the predictor cannot fit and predict dummy data."""
    if isinstance(predictor, type):
        raise error(
            f"predictor is the class {predictor.__name__}, not an object.",
            why="rubberruler fits copies of an unfitted predictor object.",
            fix=f"add (), such as predictor={predictor.__name__}().",
        )
    missing = [
        name
        for name in ("fit", "predict")
        if not callable(getattr(predictor, name, None))
    ]
    if missing:
        raise error(
            f"predictor ({type(predictor).__name__}) has no "
            f"{' or '.join(missing)} method.",
            why="rubberruler calls predictor.fit(X, y) and predictor.predict(X), "
            "as in scikit-learn.",
            fix="pass a regressor such as RandomForestRegressor(random_state=0), or "
            "add the missing method.",
        )
    rng = np.random.default_rng(0)
    matrix = rng.uniform(size=(10, 3))
    targets = rng.uniform(size=10)
    where = "dummy data of 10 rows and 3 features in the pre-check"
    model = copy_of(predictor)
    try:
        model.fit(matrix, targets)
    except Exception as exception:
        raise predictor_error("fit", exception, where, matrix) from exception
    try:
        output: object = model.predict(matrix)
    except Exception as exception:
        raise predictor_error("predict", exception, where, matrix) from exception
    predictions_of(output, 10, where)


def copy_of(predictor: object) -> Any:
    """An unfitted copy: sklearn.base.clone if it has get_params, else deepcopy."""
    if callable(getattr(predictor, "get_params", None)):
        try:
            import sklearn.base  # pyright: ignore[reportMissingTypeStubs]
        except ImportError:
            return copy.deepcopy(predictor)
        clone = cast(Callable[[object], Any], sklearn.base.clone)  # pyright: ignore[reportUnknownMemberType]
        try:
            return clone(predictor)
        except Exception as exception:
            raise error(
                f"sklearn.base.clone could not copy the predictor "
                f"({type(exception).__name__}: {exception}).",
                why="rubberruler fits a fresh copy of the predictor for each "
                "algorithm and feature type.",
                fix="make get_params() return the constructor arguments, as "
                "scikit-learn estimators do.",
            ) from exception
    return copy.deepcopy(predictor)


def predictions_of(output: object, rows: int, where: str) -> NDArray[np.float64]:
    """Convert the output of predict to shape (rows,); (rows, 1) is flattened."""
    try:
        values = np.asarray(output, dtype=np.float64)
    except (TypeError, ValueError) as exception:
        raise error(
            f"predictor.predict returned a {type(output).__name__} that is not "
            f"numbers on {where}.",
            why="predict must return one number per row of X.",
            fix="return an array of shape (m,), such as model(X).ravel().",
        ) from exception
    if values.shape == (rows, 1):
        return values[:, 0]
    if values.shape != (rows,):
        raise error(
            f"predictor.predict returned shape {values.shape} for {rows} rows on "
            f"{where}.",
            why="predict must return one number per row of X.",
            fix=f"return an array of shape ({rows},), such as model(X).ravel().",
        )
    return values


def predictor_error(
    method: str, exception: Exception, where: str, matrix: NDArray[np.float64]
) -> Exception:
    """The error for a predictor that raised; it explains NaN features if any."""
    nan_count = int(np.count_nonzero(np.isnan(matrix)))
    if nan_count:
        why = (
            f"the features have {nan_count} NaN values, which many predictors "
            "cannot take."
        )
        fix = (
            "fill them inside the predictor, such as make_pipeline(SimpleImputer(), "
            "RandomForestRegressor(random_state=0)) from scikit-learn."
        )
    else:
        why = f"rubberruler needs predictor.{method} to work on these features."
        fix = "fix the predictor; the original exception is shown above."
    return error(
        f"predictor.{method} raised {type(exception).__name__} for {where}.",
        why=why,
        fix=fix,
    )


def has_unset_random_state(predictor: object) -> bool:
    """Whether get_params() has a random_state (or a nested one) that is None."""
    get_params = getattr(predictor, "get_params", None)
    if not callable(get_params):
        return False
    try:
        params: object = get_params()
    except Exception:
        return False
    if not isinstance(params, dict):
        return False
    return any(
        (key == "random_state" or str(key).endswith("__random_state")) and value is None
        for key, value in cast(dict[object, object], params).items()
    )


def feature_type_of(features: object, function: str) -> FeatureType:
    """The feature type in features.attrs; stop if it is not a DataFrame from it."""
    if not isinstance(features, pd.DataFrame):
        raise error(
            f"{function} got features of type {type(features).__name__}.",
            why="features must be a DataFrame from rr.compute_features.",
            fix='pass result.features["rank"] or the result of rr.compute_features.',
        )
    feature_type = features.attrs.get("feature_type")
    if feature_type == "value":
        return "value"
    if feature_type == "rank":
        return "rank"
    raise error(
        f"{function} got features without a feature type in "
        'features.attrs["feature_type"].',
        why="the feature type tells value-based and rank-based features apart.",
        fix='pass result.features["rank"] or the result of rr.compute_features.',
    )


def check_same_features(
    fitted: FittedPredictor, feature_type: FeatureType, columns: Sequence[str]
) -> None:
    """Stop if the features are not the kind the predictor was fitted on."""
    if feature_type != fitted.feature_type:
        raise error(
            f"r2_score got {feature_type} features, but the predictor was fitted on "
            f"{fitted.feature_type} features.",
            why="a predictor fitted on one feature type cannot score the other.",
            fix=f'pass result.features["{fitted.feature_type}"], or fit a predictor '
            f"on the {feature_type} features.",
        )
    if tuple(columns) != fitted.feature_names:
        raise error(
            f"r2_score got features with the columns {_names(columns)}, but the "
            f"predictor was fitted on {_names(fitted.feature_names)}.",
            why="the predictor reads each feature by its column position.",
            fix="pass features from the same feature_extractor as in fit_predictor.",
        )


def feature_matrix(
    features: pd.DataFrame, instances: Sequence[ProblemInstance], function: str
) -> NDArray[np.float64]:
    """The rows of ``instances`` as a float64 array (m, k)."""
    rows = rows_of(features, instances, "features", function)
    return features.loc[rows].to_numpy(dtype=np.float64, copy=True)


def performance_matrix(
    performance: object, instances: Sequence[ProblemInstance], function: str
) -> tuple[list[str], NDArray[np.float64]]:
    """The algorithm names and the J of ``instances`` as a float64 array (m, |A|)."""
    if not isinstance(performance, pd.DataFrame):
        raise error(
            f"{function} got performance of type {type(performance).__name__}.",
            why="performance must be a DataFrame from rr.measure_performance.",
            fix="pass result.performance or the result of rr.measure_performance.",
        )
    rows = rows_of(performance, instances, "performance", function)
    algorithms = [str(name) for name in performance.columns]
    return algorithms, performance.loc[rows].to_numpy(dtype=np.float64, copy=True)


def rows_of(
    table: pd.DataFrame,
    instances: Sequence[ProblemInstance],
    name: str,
    function: str,
) -> list[str]:
    """The instance ids of ``instances``; stop if ``table`` lacks one."""
    rows = [instance.instance_id for instance in instances]
    missing = [row for row in rows if row not in table.index]
    if missing:
        raise error(
            f"{function} got {name} without a row for {_names(missing)}.",
            why=f"{function} uses the {name} of every instance in instances.",
            fix=f"pass {name} computed for these instances, such as the {name} of "
            "the same split.",
        )
    return rows


def _names(names: Sequence[str]) -> str:
    shown = ", ".join(f'"{name}"' for name in names[:3])
    return shown if len(names) <= 3 else f"{shown}, and {len(names) - 3} more"
