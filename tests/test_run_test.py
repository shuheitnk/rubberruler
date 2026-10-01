import os
from pathlib import Path

import algorithms
import numpy as np
import pandas as pd
import pytest
from helpers import PORTFOLIO, assert_three_lines, extract, tiny_run
from sklearn.ensemble import RandomForestRegressor

import rubberruler as rr
from rubberruler._card import select
from rubberruler._instances import make_split
from rubberruler._performance import run_seed
from rubberruler._run_test import RubberRulerResult, run

STAGES = [
    "[1/4] Preparing instances",
    "[2/4] Measuring performance",
    "[3/4] Computing features",
    "[4/4] Fitting predictors and scoring",
]


@pytest.fixture(scope="module")
def result() -> RubberRulerResult:
    return tiny_run()


def test_run_test_gives_the_answer_and_what_it_rests_on(
    result: RubberRulerResult,
) -> None:
    """The answer, the table, the split, performance, features, config."""
    assert result.selected_feature_type == select(result.table)
    assert list(result.table.index) == ["value", "rank"]
    assert list(result.table.columns) == [
        "r2_test",
        "r2_transformed_test",
        "degradation",
    ]
    np.testing.assert_allclose(
        result.table["degradation"],
        result.table["r2_test"] - result.table["r2_transformed_test"],
    )
    assert (len(result.split.train), len(result.split.test)) == (4, 4)
    assert len(result.split.transformed_test) == 4
    assert result.performance.shape == (12, 3)
    assert set(result.features) == {"value", "rank"}
    assert result.features["rank"].attrs["feature_type"] == "rank"
    assert list(result.features["value"].index) == list(result.performance.index)
    assert result.config == {
        "dimension": 2,
        "train_instance_numbers": (11, 12),
        "test_instance_numbers": (1, 2),
        "budget_per_dimension": 25,
        "n_runs": 3,
        "n_samples_per_dimension": 10,
        "seed": 42,
    }


def test_the_card_is_the_str_and_the_repr(result: RubberRulerResult) -> None:
    """str(result) and repr(result) are the card."""
    text = str(result)
    assert repr(result) == text
    selected = result.selected_feature_type.upper()
    assert text.split("\n")[0] == f"Rubber Ruler Test: use {selected}-based features"
    assert "3 algorithms, 4 train, 4 test, 4 transformed test" in text
    assert all(len(line) <= 80 for line in text.split("\n"))


def test_the_parts_give_the_same_r2(result: RubberRulerResult) -> None:
    """run_test calls the parts in order, so they reproduce its table."""
    for feature_type in ("value", "rank"):
        features = result.features[feature_type]
        fitted = rr.fit_predictor(
            RandomForestRegressor(n_estimators=10, random_state=0),
            features,
            result.performance,
            instances=result.split.train,
        )
        for column, instances in (
            ("r2_test", result.split.test),
            ("r2_transformed_test", result.split.transformed_test),
        ):
            r2 = rr.r2_score(fitted, features, result.performance, instances=instances)
            assert r2 == result.table.loc[feature_type, column]


def test_the_config_repeats_the_test(result: RubberRulerResult) -> None:
    """run_test(..., **result.config) repeats the same test."""
    again = run(
        PORTFOLIO,
        extract,
        RandomForestRegressor(n_estimators=10, random_state=0),
        n_jobs=1,
        cache_dir=None,
        function_ids=(1, 2),
        **result.config,
    )
    pd.testing.assert_frame_equal(again.table, result.table)


def test_n_jobs_does_not_change_the_table(result: RubberRulerResult) -> None:
    """n_jobs=1 and n_jobs=2 give the same table."""
    pd.testing.assert_frame_equal(tiny_run(n_jobs=2).table, result.table)


def test_the_plan_and_the_stages_are_shown_on_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The plan line, then the four stages, on stderr; stdout stays clean."""
    tiny_run()
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = [
        "rubberruler: 108 runs (0 cached, 108 to run) without a cache",
        *STAGES,
    ]
    positions = [captured.err.find(line) for line in lines]
    assert all(position >= 0 for position in positions)
    assert positions == sorted(positions)


def test_a_second_run_test_takes_the_runs_from_the_cache(
    result: RubberRulerResult, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same call again runs nothing and gives the same table."""
    first = tiny_run(cache_dir=tmp_path)
    capsys.readouterr()
    second = tiny_run(cache_dir=tmp_path)
    err = capsys.readouterr().err
    folder = os.path.join(tmp_path, "")
    assert f"rubberruler: 108 runs (108 cached, 0 to run) in {folder}" in err
    pd.testing.assert_frame_equal(first.table, result.table)
    pd.testing.assert_frame_equal(second.table, result.table)


def test_warnings_are_collected_under_the_card() -> None:
    """Failed runs and random_state=None are listed under the card."""
    portfolio = {
        "flaky": algorithms.fails_on_odd_seeds,
        "random": algorithms.random_search,
    }
    result = tiny_run(portfolio, predictor=RandomForestRegressor(n_estimators=10))
    lines = str(result).split("\n")
    warnings = lines[lines.index("Warnings:") + 1 :]
    assert warnings[0].endswith(
        "runs failed; their best value so far was used (+inf if none) (see the log)"
    )
    assert warnings[-1] == (
        "  predictor has random_state=None, so R^2 may change between calls"
    )


def test_one_failed_run_is_written_in_the_singular() -> None:
    """Exactly one failed run gives "1 run failed; its best value ..."."""
    first = make_split(2, (11,), (), seed=42, function_ids=(1,)).train[0]
    unlucky = run_seed(first, "flaky", 2)

    def fails_once(problem: rr.Problem) -> None:
        problem(np.zeros(problem.dimension))
        if problem.seed == unlucky:
            raise RuntimeError("unlucky seed")

    result = tiny_run({"flaky": fails_once, "random": algorithms.random_search})
    lines = str(result).split("\n")
    assert lines[lines.index("Warnings:") + 1 :] == [
        "  1 run failed; its best value so far was used (+inf if none) (see the log)"
    ]


def lazy_extractor(X: np.ndarray, y: np.ndarray) -> object:
    return y


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"extractor": lazy_extractor}, "feature_extractor returned a ndarray"),
        ({"predictor": RandomForestRegressor}, "predictor is the class"),
        (
            {
                "portfolio": {
                    "a": algorithms.random_search,
                    "b": algorithms.returns_without_evaluating,
                }
            },
            "returned without evaluating",
        ),
        ({"dimension": 1}, "dimension=1"),
        ({"n_jobs": 0}, "n_jobs=0"),
        ({"test_instance_numbers": (11,)}, "share 11"),
    ],
)
def test_broken_inputs_stop_before_the_instances_are_prepared(
    capsys: pytest.CaptureFixture[str], changes: dict[str, object], message: str
) -> None:
    """Settings and pre-checks stop before any stage begins."""
    with pytest.raises(rr.RubberRulerError, match=message) as raised:
        tiny_run(**changes)  # pyright: ignore[reportArgumentType]
    assert_three_lines(raised.value)
    assert "[1/4]" not in capsys.readouterr().err


def test_an_extractor_that_fails_on_another_function_stops_before_the_runs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """run_test tries the extractor on the first training instance of each
    function, so one that fails on the second function stops before any run."""
    calls = {"count": 0}

    def fails_on_f02(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        calls["count"] += 1  # f01 value, f01 rank, then f02 value
        if calls["count"] == 3:
            raise ValueError("a plateau")
        return {"mean": float(np.mean(y))}

    with pytest.raises(
        rr.RubberRulerError, match=r"raised ValueError on f02_i011_d2 \(value\)"
    ):
        tiny_run(extractor=fails_on_f02)
    assert "runs (" not in capsys.readouterr().err


def test_equal_performance_everywhere_stops() -> None:
    """When every J is equal, R^2 is nan and there is no answer."""
    portfolio = {"center": algorithms.center, "also center": algorithms.center}
    with pytest.raises(
        rr.RubberRulerError, match=r"R\^2 on the transformed test"
    ) as raised:
        tiny_run(portfolio)
    assert_three_lines(raised.value)
