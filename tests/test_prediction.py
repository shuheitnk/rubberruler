import logging
import math
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from helpers import assert_three_lines
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score as sklearn_r2_score
from sklearn.pipeline import make_pipeline

import rubberruler as rr
from rubberruler import _prediction
from rubberruler._instances import InstanceSplit, make_split
from rubberruler._prediction import coefficient_of_determination

ALGORITHMS = ["a", "b", "c"]


@pytest.fixture(scope="module")
def split() -> InstanceSplit:
    return make_split(2, (1, 2, 3, 4, 5, 6), (7, 8), seed=3, function_ids=(1, 2))


def make_features(
    split: InstanceSplit, feature_type: str, columns: int = 3
) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    ids = [instance.instance_id for instance in split]
    features = pd.DataFrame(
        rng.uniform(size=(len(ids), columns)),
        index=pd.Index(ids, name="instance_id"),
        columns=[f"x{index}" for index in range(columns)],
    )
    features.attrs["feature_type"] = feature_type
    return features


def make_performance(features: pd.DataFrame) -> pd.DataFrame:
    x = features.to_numpy()
    values = np.column_stack([x[:, 0], 1 - x[:, 0], 0.5 * x[:, 1] + 0.25])
    return pd.DataFrame(values, index=features.index, columns=ALGORITHMS)


class MeanPredictor:
    """Not scikit-learn: no get_params, so it is copied with deepcopy."""

    def __init__(self) -> None:
        self.mean: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.mean = float(np.mean(y))

    def predict(self, X: np.ndarray) -> list[list[float]]:
        assert self.mean is not None
        return [[self.mean] for _ in range(X.shape[0])]


@settings(deadline=None)
@given(
    shape=st.tuples(st.integers(2, 12), st.integers(2, 4)),
    data=st.data(),
)
def test_r2_matches_scikit_learn_on_the_flattened_arrays(
    shape: tuple[int, int], data: st.DataObject
) -> None:
    """R-R2-1: R^2 equals sklearn r2_score(J.ravel(), clip(J_hat).ravel())."""
    actual = data.draw(arrays(np.float64, shape, elements=st.floats(0, 1)))
    predicted = data.draw(arrays(np.float64, shape, elements=st.floats(-1, 2)))
    clipped = np.clip(predicted, 0, 1)
    if np.all(actual == actual.flat[0]):
        assert math.isnan(coefficient_of_determination(actual, clipped))
        return
    # With almost equal J the denominator is very small, and summation order dominates.
    assume(np.ptp(actual) > 1e-3)
    assert coefficient_of_determination(actual, clipped) == pytest.approx(
        sklearn_r2_score(actual.ravel(), clipped.ravel())
    )


def test_r2_is_nan_when_every_performance_is_equal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R-R2-1: a zero denominator gives nan and a log message."""
    with caplog.at_level(logging.WARNING, logger="rubberruler"):
        value = coefficient_of_determination(np.full((3, 2), 0.5), np.zeros((3, 2)))
    assert math.isnan(value)
    assert caplog.messages == [
        "R^2 is undefined because all performance values are equal; returned nan"
    ]


def test_a_copy_is_fitted_for_each_algorithm(split: InstanceSplit) -> None:
    """One fitted copy per algorithm; the given predictor is not fitted."""
    features = make_features(split, "rank")
    performance = make_performance(features)
    predictor = LinearRegression()
    fitted = rr.fit_predictor(predictor, features, performance, instances=split.train)
    assert fitted.feature_type == "rank"
    assert fitted.algorithms == tuple(ALGORITHMS)
    assert len({id(model) for model in fitted.predictors}) == 3
    assert not hasattr(predictor, "coef_")
    r2 = rr.r2_score(fitted, features, performance, instances=split.test)
    assert r2 == pytest.approx(1.0)


def test_r2_uses_clipped_predictions_of_the_selected_rows(split: InstanceSplit) -> None:
    """R-R2-1: R^2 over the pairs (q, a) of Q, with predictions clipped to [0, 1]."""
    features = make_features(split, "value")
    performance = make_performance(features)
    fitted = rr.fit_predictor(
        LinearRegression(), features, performance, instances=split.train
    )
    ids = [instance.instance_id for instance in split.transformed_test]
    x = features.loc[ids].to_numpy()
    predicted = np.column_stack([model.predict(x) for model in fitted.predictors])
    expected = sklearn_r2_score(
        performance.loc[ids].to_numpy().ravel(), np.clip(predicted, 0, 1).ravel()
    )
    r2 = rr.r2_score(fitted, features, performance, instances=split.transformed_test)
    assert r2 == pytest.approx(expected)


def test_pipelines_and_other_predictors_work_without_glue(split: InstanceSplit) -> None:
    """Pipeline, and objects without get_params (copied with deepcopy)."""
    features = make_features(split, "rank")
    performance = make_performance(features)
    for predictor in (
        make_pipeline(
            SimpleImputer(), RandomForestRegressor(n_estimators=5, random_state=0)
        ),
        MeanPredictor(),
    ):
        fitted = rr.fit_predictor(
            predictor, features, performance, instances=split.train
        )
        r2 = rr.r2_score(fitted, features, performance, instances=split.test)
        assert math.isfinite(r2)


def test_other_feature_types_or_columns_stop(split: InstanceSplit) -> None:
    """r2_score stops if the features are not those of the fit."""
    rank = make_features(split, "rank")
    performance = make_performance(rank)
    fitted = rr.fit_predictor(
        LinearRegression(), rank, performance, instances=split.train
    )
    value = make_features(split, "value")
    with pytest.raises(rr.RubberRulerError, match="received value features") as raised:
        rr.r2_score(fitted, value, performance, instances=split.test)
    assert_three_lines(raised.value)
    wider = make_features(split, "rank", columns=4)
    with pytest.raises(rr.RubberRulerError, match="columns") as raised:
        rr.r2_score(fitted, wider, performance, instances=split.test)
    assert_three_lines(raised.value)
    unlabeled = rank.copy()
    unlabeled.attrs = {}
    with pytest.raises(rr.RubberRulerError, match="without a feature type") as raised:
        rr.r2_score(fitted, unlabeled, performance, instances=split.test)
    assert_three_lines(raised.value)
    with pytest.raises(rr.RubberRulerError, match="as fitted") as raised:
        rr.r2_score(LinearRegression(), rank, performance, instances=split.test)  # pyright: ignore[reportArgumentType]
    assert_three_lines(raised.value)


def test_missing_rows_stop(split: InstanceSplit) -> None:
    """Every instance needs its row."""
    features = make_features(split, "rank")
    performance = make_performance(features).drop(index=split.test[0].instance_id)
    fitted = rr.fit_predictor(
        LinearRegression(), features, performance, instances=split.train
    )
    with pytest.raises(rr.RubberRulerError, match=split.test[0].instance_id) as raised:
        rr.r2_score(fitted, features, performance, instances=split.test)
    assert_three_lines(raised.value)


class ChangesItsInput:
    """Overwrites X and y in place, as some hand-written predictors do."""

    seen: ClassVar[list[np.ndarray]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ChangesItsInput":
        assert X.flags.writeable and y.flags.writeable
        ChangesItsInput.seen.append(X.copy())
        X[:] = 0.0
        y[:] = 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert X.flags.writeable
        X[:] = 0.0
        return np.full(X.shape[0], 0.5)


def test_predictors_get_their_own_writable_arrays(split: InstanceSplit) -> None:
    """Each fit and predict receives writable copies, safe to change in place."""
    features = make_features(split, "rank")
    performance = make_performance(features)
    before = features.copy()
    ChangesItsInput.seen.clear()
    fitted = rr.fit_predictor(
        ChangesItsInput(), features, performance, instances=split.train
    )
    rr.r2_score(fitted, features, performance, instances=split.test)
    train = features.loc[[instance.instance_id for instance in split.train]]
    seen = ChangesItsInput.seen[-len(ALGORITHMS) :]
    assert len(seen) == len(ALGORITHMS)
    for matrix in seen:
        np.testing.assert_array_equal(matrix, train.to_numpy())
    pd.testing.assert_frame_equal(features, before)


def with_rounding_noise(
    features: pd.DataFrame, seed: int, distinct: dict[int, float]
) -> pd.DataFrame:
    """Add the column "skewness": 0 up to rounding (+-1e-15), except the given rows.

    Like the skewness of rank-based y: 0 for every instance without ties in y.
    """
    rng = np.random.default_rng(seed)
    values = rng.integers(-10, 11, len(features)) * 1e-16
    for row, value in distinct.items():
        values[row] = value
    noisy = features.copy()
    noisy["skewness"] = values
    return noisy


class RecordsItsInput:
    """Keeps the X of every fit and predict."""

    seen: ClassVar[list[np.ndarray]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RecordsItsInput":
        RecordsItsInput.seen.append(X.copy())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        RecordsItsInput.seen.append(X.copy())
        return np.full(X.shape[0], 0.5)


def test_values_that_differ_only_by_rounding_become_equal(split: InstanceSplit) -> None:
    """R-R2-1: values within 1e-12 of 0 (+-1e-15) or 1e-9 of each other (0.3 + ulp)
    are made equal for fit and predict; the others, and NaN, stay as they are."""
    features = with_rounding_noise(
        make_features(split, "rank"), seed=0, distinct={0: 3e-4, 1: np.nan}
    )
    features["constant"] = 0.3 + np.arange(len(features)) % 3 * np.spacing(0.3)
    performance = make_performance(features)
    RecordsItsInput.seen.clear()
    fitted = rr.fit_predictor(
        RecordsItsInput(), features, performance, instances=split.train
    )
    rr.r2_score(fitted, features, performance, instances=split.test)
    train = features.loc[[instance.instance_id for instance in split.train]]
    test = features.loc[[instance.instance_id for instance in split.test]]
    # seen[0] and seen[1] are the pre-check's fit and predict on dummy data.
    fit_x, predict_x = RecordsItsInput.seen[2], RecordsItsInput.seen[-1]
    np.testing.assert_array_equal(fit_x[:, :3], train.iloc[:, :3].to_numpy())
    np.testing.assert_array_equal(predict_x[:, :3], test.iloc[:, :3].to_numpy())
    assert fit_x[0, 3] == 3e-4
    assert np.isnan(fit_x[1, 3])
    assert len(set(fit_x[2:, 3]) | set(predict_x[:, 3])) == 1
    assert abs(fit_x[2, 3]) <= 1e-15
    assert len(set(fit_x[:, 4]) | set(predict_x[:, 4])) == 1


def test_one_huge_value_does_not_merge_the_other_values(split: InstanceSplit) -> None:
    """R-R2-1: whether two values are made equal depends on those two alone
    (math.isclose with rel_tol=1e-9, abs_tol=1e-12), so a ratio of 1e12 on one
    instance leaves the other values of its column as they are."""
    features = make_features(split, "rank")
    ratio = np.linspace(1.0, 3.9e3, len(features))  # like coef.max_by_min of pflacco
    ratio[0] = 1e12
    ratio[1] = 1e12 * (1 + 5e-10)  # within 1e-9 of 1e12 itself
    features["ratio"] = ratio
    performance = make_performance(features)
    RecordsItsInput.seen.clear()
    fitted = rr.fit_predictor(
        RecordsItsInput(), features, performance, instances=split.train
    )
    rr.r2_score(fitted, features, performance, instances=split.test)
    train = features.loc[[instance.instance_id for instance in split.train]]
    test = features.loc[[instance.instance_id for instance in split.test]]
    fit_x, predict_x = RecordsItsInput.seen[2], RecordsItsInput.seen[-1]
    assert fit_x[0, 3] == fit_x[1, 3] == 1e12
    np.testing.assert_array_equal(fit_x[2:, 3], train["ratio"].to_numpy()[2:])
    np.testing.assert_array_equal(predict_x[:, 3], test["ratio"].to_numpy())


class NearestOnLastFeature:
    """Predicts the J of the first training row whose last feature is the closest.

    Like a tree, it depends on how the values of a feature are ordered, so rounding
    noise in that feature changes its predictions.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NearestOnLastFeature":
        self.x, self.y = X[:, -1].copy(), y.copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array(
            [self.y[np.argmin(np.abs(self.x - value))] for value in X[:, -1]]
        )


def test_rounding_noise_does_not_change_r2(
    split: InstanceSplit, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two patterns of rounding noise give the same R^2, to the last digit;
    without making the values equal, the noise decides the predictions."""
    base = make_features(split, "value", columns=2)
    performance = make_performance(base)

    def r2(seed: int) -> float:
        features = with_rounding_noise(base, seed, {0: 3e-4, 5: -2e-4})
        fitted = rr.fit_predictor(
            NearestOnLastFeature(), features, performance, instances=split.train
        )
        return rr.r2_score(fitted, features, performance, instances=split.test)

    assert r2(1) == r2(2)
    monkeypatch.setattr(_prediction, "REL_TOL", 0.0)
    monkeypatch.setattr(_prediction, "ABS_TOL", 0.0)
    assert r2(1) != r2(2)


class NoPredict:
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass


class FailsToFit:
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise RuntimeError("cannot fit")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])


class WrongShape:
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros((X.shape[0], 2))


@pytest.mark.parametrize(
    ("predictor", "message"),
    [
        (
            RandomForestRegressor,
            r"add \(\), such as predictor=RandomForestRegressor\(\)",
        ),
        (NoPredict(), "has no predict method"),
        (FailsToFit(), "predictor.fit raised RuntimeError"),
        (WrongShape(), r"returned shape \(10, 2\) for 10 rows"),
    ],
)
def test_broken_predictors_stop_in_the_pre_check(
    split: InstanceSplit, predictor: object, message: str
) -> None:
    """The pre-check stops them on dummy data."""
    features = make_features(split, "rank")
    with pytest.raises(rr.RubberRulerError, match=message) as raised:
        rr.fit_predictor(
            predictor, features, make_performance(features), instances=split.train
        )
    assert_three_lines(raised.value)


def test_nan_features_suggest_an_imputer(split: InstanceSplit) -> None:
    """A failing fit on NaN features gives the count and SimpleImputer."""
    features = make_features(split, "rank")
    performance = make_performance(features)
    features.iloc[0, 0] = np.nan
    features.iloc[3, 1] = np.nan
    with pytest.raises(rr.RubberRulerError) as raised:
        rr.fit_predictor(
            LinearRegression(), features, performance, instances=split.train
        )
    assert_three_lines(raised.value)
    message = str(raised.value)
    assert (
        'predictor.fit raised ValueError for algorithm "a" (rank features)' in message
    )
    assert "2 NaN values" in message
    assert "SimpleImputer" in message
    assert isinstance(raised.value.__cause__, ValueError)


def test_an_unset_random_state_is_warned(
    split: InstanceSplit, caplog: pytest.LogCaptureFixture
) -> None:
    """random_state=None, also inside a Pipeline, gives the warning."""
    features = make_features(split, "rank")
    performance = make_performance(features)
    for predictor in (
        RandomForestRegressor(n_estimators=3),
        make_pipeline(SimpleImputer(), RandomForestRegressor(n_estimators=3)),
    ):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="rubberruler"):
            rr.fit_predictor(predictor, features, performance, instances=split.train)
        assert caplog.messages == [
            "predictor has random_state=None, so R^2 may change between calls"
        ]
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="rubberruler"):
        rr.fit_predictor(
            RandomForestRegressor(n_estimators=3, random_state=0),
            features,
            performance,
            instances=split.train,
        )
    assert caplog.messages == []


class ReturnsWords:
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def predict(self, X: np.ndarray) -> list[str]:
        return ["high"] * X.shape[0]


class CannotBeCloned:
    """Has get_params, but its constructor does not take what it returns."""

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {"depth": 3}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])


@pytest.mark.parametrize(
    ("predictor", "message"),
    [
        (ReturnsWords(), "returned a list that is not numbers"),
        (CannotBeCloned(), "sklearn.base.clone could not copy the predictor"),
    ],
)
def test_predictors_that_cannot_be_used_stop(
    split: InstanceSplit, predictor: object, message: str
) -> None:
    """Predictions that are not numbers, and copies that fail."""
    features = make_features(split, "rank")
    with pytest.raises(rr.RubberRulerError, match=message) as raised:
        rr.fit_predictor(
            predictor, features, make_performance(features), instances=split.train
        )
    assert_three_lines(raised.value)


def test_performance_of_other_algorithms_stops(split: InstanceSplit) -> None:
    """r2_score needs the algorithms that the predictor learned."""
    features = make_features(split, "rank")
    performance = make_performance(features)
    fitted = rr.fit_predictor(
        LinearRegression(), features, performance, instances=split.train
    )
    other = performance.rename(columns={"a": "z"})
    with pytest.raises(rr.RubberRulerError, match="was fitted for") as raised:
        rr.r2_score(fitted, features, other, instances=split.test)
    assert_three_lines(raised.value)


@pytest.mark.parametrize("wrong", ["features", "performance"])
def test_features_and_performance_must_be_dataframes(
    split: InstanceSplit, wrong: str
) -> None:
    """fit_predictor takes the DataFrames of the parts."""
    features = make_features(split, "rank")
    performance = make_performance(features)
    arguments: dict[str, object] = {"features": features, "performance": performance}
    arguments[wrong] = arguments[wrong].to_numpy()  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(rr.RubberRulerError, match=f"received {wrong} of type ndarray"):
        rr.fit_predictor(LinearRegression(), instances=split.train, **arguments)  # pyright: ignore[reportArgumentType]
