import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest
from helpers import assert_three_lines, run_readme_block
from scipy.stats import rankdata

import rubberruler as rr
from rubberruler._features import (
    feature_sample,
    feature_table,
    feature_warnings,
    preprocess,
)
from rubberruler._instances import InstanceObjective, InstanceSplit, make_split
from rubberruler._seeding import generator


def summary(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """A small extractor: a few statistics of the sample."""
    return {
        "y_mean": float(np.mean(y)),
        "y_top_corner": float(y[np.argmin(X[:, 0])]),
        "size": X.shape[0],
    }


@pytest.fixture(scope="module")
def split() -> InstanceSplit:
    return make_split(2, (1,), (2,), seed=3, function_ids=(1, 10))


def test_value_rescales_y_to_0_and_1() -> None:
    """value is (y - min) / (max - min)."""
    y, constant = preprocess(np.array([3.0, 1.0, 5.0]), "value")
    np.testing.assert_array_equal(y, [0.5, 0.0, 1.0])
    assert not constant


def test_value_of_a_constant_objective_is_zero() -> None:
    """If max = min, value is a zero vector and the instance is counted."""
    y, constant = preprocess(np.array([2.0, 2.0, 2.0]), "value")
    np.testing.assert_array_equal(y, [0.0, 0.0, 0.0])
    assert constant


def test_rank_is_the_average_rank_divided_by_n() -> None:
    """rank is scipy.stats.rankdata(y, method="average") / n."""
    y_raw = np.array([3.0, 1.0, 3.0, 7.0])
    y, constant = preprocess(y_raw, "rank")
    np.testing.assert_array_equal(y, [2.5 / 4, 1 / 4, 2.5 / 4, 4 / 4])
    np.testing.assert_array_equal(y, rankdata(y_raw, method="average") / 4)
    assert not constant


def test_the_sample_is_uniform_and_shared_by_the_transformed_instance(
    split: InstanceSplit,
) -> None:
    """X comes from the "features" seed, the same after transform."""
    original, transformed = split.test[0], split.transformed_test[0]
    raw: dict[tuple[str, int], np.ndarray] = {}
    sample, _, _ = feature_sample(original, "value", 7, raw)
    sample_after, _, _ = feature_sample(transformed, "value", 7, raw)
    expected = generator("features", 3, "bbob", 1, 2, 2).uniform(
        original.lower_bounds, original.upper_bounds, size=(14, 2)
    )
    np.testing.assert_array_equal(sample, expected)
    np.testing.assert_array_equal(sample_after, expected)


def test_rank_features_do_not_change_under_the_transformation(
    split: InstanceSplit,
) -> None:
    """g keeps the order of y, so rank features stay and value ones move."""
    instances = [split.test[0], split.transformed_test[0]]
    rank = rr.compute_features(summary, instances, "rank", n_samples_per_dimension=20)
    value = rr.compute_features(summary, instances, "value", n_samples_per_dimension=20)
    np.testing.assert_array_equal(rank.iloc[0], rank.iloc[1])
    assert not np.array_equal(value.iloc[0], value.iloc[1])


def test_compute_features_returns_one_row_per_instance(split: InstanceSplit) -> None:
    """Rows are instance_id, columns follow the first instance's keys."""
    features = rr.compute_features(summary, split, "value", n_samples_per_dimension=5)
    assert list(features.index) == [instance.instance_id for instance in split]
    assert features.index.name == "instance_id"
    assert list(features.columns) == ["y_mean", "y_top_corner", "size"]
    assert features.attrs["feature_type"] == "value"
    assert (features["size"] == 10.0).all()
    assert features.to_numpy().dtype == np.float64


def test_values_that_float_accepts_are_features(split: InstanceSplit) -> None:
    """Ints and NumPy scalars are accepted; the key order is kept."""

    def mixed(X: np.ndarray, y: np.ndarray) -> Mapping[str, float]:
        return {"b": np.float32(1.5), "a": np.int64(2), "c": True}  # pyright: ignore[reportReturnType]

    features = rr.compute_features(mixed, split.train, "rank")
    assert list(features.columns) == ["b", "a", "c"]
    np.testing.assert_array_equal(features.iloc[0], [1.5, 2.0, 1.0])


def test_f_is_evaluated_once_per_instance(
    split: InstanceSplit, monkeypatch: pytest.MonkeyPatch
) -> None:
    """f(X) is evaluated once; the transformed instance uses g(f(X))."""
    calls: list[int] = []
    many = InstanceObjective.many

    def counted(self: InstanceObjective, points: np.ndarray) -> np.ndarray:
        calls.append(points.shape[0])
        return many(self, points)

    monkeypatch.setattr(InstanceObjective, "many", counted)
    raw: dict[tuple[str, int], np.ndarray] = {}
    for feature_type in ("value", "rank"):
        feature_table(
            summary,
            [split.test[0], split.transformed_test[0]],
            feature_type,
            5,
            advance=lambda count: None,
            raw_values=raw,
        )
    assert calls == [10]


def test_different_keys_stop_with_the_three_line_message(split: InstanceSplit) -> None:
    """Every instance must give the same features."""
    calls = {"count": 0}

    def shrinking(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        calls["count"] += 1
        # 4 calls in the pre-check (f01 and f10, value and rank), then the instances.
        size = 16 if calls["count"] <= 5 else 14
        return {f"feature_{index}": 0.0 for index in range(size)}

    with pytest.raises(rr.RubberRulerError) as raised:
        rr.compute_features(shrinking, split, "rank")
    assert str(raised.value) == (
        "feature_extractor returned 14 features on f10_i001_d2 (rank) "
        "but 16 on f01_i001_d2 (rank).\n"
        "Why: every instance must give the same features.\n"
        'Fix: return a dict with fixed keys; use float("nan") for values you '
        "cannot compute."
    )


def test_renamed_keys_stop(split: InstanceSplit) -> None:
    """The same number of features with other names stops too."""
    calls = {"count": 0}

    def renaming(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        calls["count"] += 1
        return {"a": 0.0, "b" if calls["count"] <= 3 else "c": 1.0}

    with pytest.raises(rr.RubberRulerError, match='"b" missing and "c" new') as raised:
        rr.compute_features(renaming, split, "value")
    assert_three_lines(raised.value)


def test_an_extractor_that_raises_stops_with_the_instance(
    split: InstanceSplit,
) -> None:
    """The error names the instance and keeps the exception as __cause__."""
    calls = {"count": 0}

    def fragile(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        # 4 calls in the pre-check (f01 and f10, value and rank), then the instances.
        calls["count"] += 1
        if calls["count"] > 6:
            raise ZeroDivisionError("no")
        return {"a": 1.0}

    with pytest.raises(
        rr.RubberRulerError, match=r"raised ZeroDivisionError on f01_i002_d2 \(value\)"
    ) as raised:
        rr.compute_features(fragile, split, "value")
    assert_three_lines(raised.value)
    assert isinstance(raised.value.__cause__, ZeroDivisionError)


@pytest.mark.parametrize(
    ("extractor", "message"),
    [
        (
            lambda X, y: np.array([1.0, 2.0]),
            "returned a ndarray on f01_i001_d2 \\(value\\)",
        ),
        (lambda X, y: {1: 0.5}, "returned the key 1 on"),
        (lambda X, y: {"a": "high"}, "returned 'high' for \"a\""),
        (lambda X, y: {"a": 0.0, "b": 1.0} if y.min() > 0 else {"a": 0.0}, "(rank)"),
        (lambda X, y: 1 / 0, "raised ZeroDivisionError on f01_i001_d2 \\(value\\)"),
    ],
)
def test_broken_extractors_stop_in_the_pre_check(
    split: InstanceSplit, extractor: object, message: str
) -> None:
    """The pre-check stops them on the first instance."""
    with pytest.raises(rr.RubberRulerError, match=message) as raised:
        rr.compute_features(extractor, split, "rank")  # pyright: ignore[reportArgumentType]
    assert_three_lines(raised.value)


def test_the_pre_check_tries_the_first_instance_of_each_function(
    split: InstanceSplit,
) -> None:
    """An extractor that fails only on another function stops in the
    pre-check, before any instance is computed."""
    calls = {"count": 0}

    def fails_on_f10(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        calls["count"] += 1  # f01 value, f01 rank, then f10 value
        if calls["count"] == 3:
            raise ValueError("a plateau")
        return {"a": 1.0}

    with pytest.raises(
        rr.RubberRulerError, match=r"raised ValueError on f10_i001_d2 \(value\)"
    ) as raised:
        rr.compute_features(fails_on_f10, split, "rank")
    assert_three_lines(raised.value)
    assert calls["count"] == 3


def test_keys_must_match_between_feature_types(split: InstanceSplit) -> None:
    """value and rank must give the same keys on the first instance."""

    def depends_on_type(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        is_rank = bool(np.all(y > 0))
        return {"rank_only": 1.0} if is_rank else {"value_only": 1.0}

    with pytest.raises(
        rr.RubberRulerError, match="on f01_i001_d2 \\(rank\\)"
    ) as raised:
        rr.compute_features(depends_on_type, split, "value")
    assert_three_lines(raised.value)


def test_nan_features_and_constant_objectives_are_warned(
    split: InstanceSplit,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """NaN features and constant objectives are counted in warnings."""

    def with_nan(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        return {"a": float("nan") if X[0, 0] > 0 else 1.0}

    with caplog.at_level(logging.WARNING, logger="rubberruler"):
        features = rr.compute_features(with_nan, split, "value")
    nan_count = int(features["a"].isna().sum())
    assert nan_count > 0
    assert [record.getMessage() for record in caplog.records] == [
        f"{nan_count} instances have NaN features"
    ]
    caplog.clear()
    monkeypatch.setattr(
        InstanceObjective, "many", lambda self, points: np.ones(points.shape[0])
    )
    with caplog.at_level(logging.WARNING, logger="rubberruler"):
        features = rr.compute_features(summary, split.train, "value")
    assert [record.getMessage() for record in caplog.records] == [
        "2 instances have a constant objective on the feature sample"
    ]
    assert (features["y_mean"] == 0.0).all()


def test_infinite_feature_values_count_as_nan(
    split: InstanceSplit, caplog: pytest.LogCaptureFixture
) -> None:
    """+inf and -inf become NaN when rubberruler takes them, and
    count in the NaN warning line."""

    def with_infinity(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        return {"plus": float("inf"), "minus": np.float64("-inf"), "finite": 1.0}

    with caplog.at_level(logging.WARNING, logger="rubberruler"):
        features = rr.compute_features(with_infinity, split, "value")
    assert features["plus"].isna().all()
    assert features["minus"].isna().all()
    assert (features["finite"] == 1.0).all()
    assert [record.getMessage() for record in caplog.records] == [
        f"{len(features)} instances have NaN features"
    ]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"feature_type": "values"}, "feature_type='values'"),
        (
            {"feature_type": "rank", "n_samples_per_dimension": 0},
            "n_samples_per_dimension=0",
        ),
    ],
)
def test_bad_settings_stop(
    split: InstanceSplit, arguments: dict[str, object], message: str
) -> None:
    """feature_type and n_samples_per_dimension are checked."""
    with pytest.raises(rr.RubberRulerError, match=message) as raised:
        rr.compute_features(summary, split, **arguments)  # pyright: ignore[reportArgumentType]
    assert_three_lines(raised.value)


def test_feature_rows_follow_the_instances(split: InstanceSplit) -> None:
    """Rows can be selected with the instances of the split."""
    features = rr.compute_features(summary, split, "rank", n_samples_per_dimension=3)
    train = features.loc[[instance.instance_id for instance in split.train]]
    assert isinstance(train, pd.DataFrame)
    assert len(train) == len(split.train)


def test_an_extractor_that_is_not_a_function_stops(split: InstanceSplit) -> None:
    """feature_extractor must be a function."""
    with pytest.raises(rr.RubberRulerError, match="is a str, not a function") as raised:
        rr.compute_features("ela", split, "rank")  # pyright: ignore[reportArgumentType]
    assert_three_lines(raised.value)


def test_a_function_of_another_shape_joins_in_one_line(split: InstanceSplit) -> None:
    """The one-line lambda of examples/README.md wraps a (y, X) -> Series
    function."""
    names = run_readme_block("def my_features")
    features = rr.compute_features(names["feature_extractor"], split, "rank")  # pyright: ignore[reportArgumentType]
    assert list(features.columns) == ["y_mean", "y_std", "best_x0"]
    assert len(features) == len(list(split))


def test_feature_warnings_are_singular_for_one_instance() -> None:
    """The feature warnings use the singular for one instance."""
    assert feature_warnings(["a"], ["b"]) == [
        "1 instance has NaN features",
        "1 instance has a constant objective on the feature sample",
    ]
    assert feature_warnings(["a", "b"], ["c", "d", "e"]) == [
        "2 instances have NaN features",
        "3 instances have a constant objective on the feature sample",
    ]
    assert feature_warnings([], []) == []
