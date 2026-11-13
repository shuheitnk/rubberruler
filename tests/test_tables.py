import numpy as np
import pandas as pd
import pytest
from helpers import PORTFOLIO, assert_three_lines, extract, tiny_run

import rubberruler as rr
from rubberruler._run_test import RubberRulerResult


@pytest.fixture(scope="module")
def result() -> RubberRulerResult:
    return tiny_run()


def other_extract(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Another extractor, for trying it on the same performance."""
    return {"median": float(np.median(y)), "top": float(np.max(y))}


def test_given_tables_for_every_instance_need_no_portfolio_and_no_extractor(
    result: RubberRulerResult, capsys: pytest.CaptureFixture[str]
) -> None:
    """With the performance and the features of every instance, run_test
    runs nothing, computes nothing, and gives the same table; the card says which
    tables were given."""
    capsys.readouterr()
    again = tiny_run(
        None,
        None,
        performance=result.performance,
        features=result.features,
        **result.config,  # pyright: ignore[reportArgumentType]
    )
    pd.testing.assert_frame_equal(again.table, result.table, check_exact=True)
    pd.testing.assert_frame_equal(again.performance, result.performance)
    assert "there are no runs" in capsys.readouterr().err
    card = str(again)
    assert "performance given for 12 of 12 instances" in card
    assert "value features given for 12 of 12 instances" in card
    assert "rank features given for 12 of 12 instances" in card
    assert all(len(line) <= 80 for line in card.split("\n"))


def test_missing_rows_are_measured_and_computed_as_before(
    result: RubberRulerResult,
) -> None:
    """Given rows are used and only the missing rows are measured or
    computed, so tables taken from a full run give the same table. Rows of other
    instances are not used."""
    transformed = [p.instance_id for p in result.split.transformed_test]
    partial_performance = result.performance.drop(index=transformed)
    partial_performance = pd.concat(
        [partial_performance, partial_performance.iloc[:1].rename(index=lambda _: "x")]
    )
    partial_value = result.features["value"].iloc[:5]
    again = tiny_run(
        performance=partial_performance,
        features={"value": partial_value},
    )
    pd.testing.assert_frame_equal(again.table, result.table, check_exact=True)
    pd.testing.assert_frame_equal(
        again.features["value"], result.features["value"], check_exact=True
    )
    assert "performance given for 8 of 12 instances" in str(again)


def test_another_extractor_on_the_same_performance(
    result: RubberRulerResult,
) -> None:
    """The performance of a run, given with another extractor, needs no
    portfolio: only the features are computed."""
    other = tiny_run(
        None,
        other_extract,
        performance=result.performance,
        **result.config,  # pyright: ignore[reportArgumentType]
    )
    assert list(other.features["rank"].columns) == ["median", "top"]
    pd.testing.assert_frame_equal(other.performance, result.performance)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("no portfolio", "portfolio is None, but performance has no rows for 4 of"),
        ("no extractor", 'feature_extractor is None, but features\\["value"\\]'),
        ("other rows", 'with rows for "f24_i001_d2", none of which is an instance'),
        ("other algorithms", "but the portfolio has"),
        ("raw values", "which is not in \\[0, 1\\]"),
        ("wrong key", "features with the key 'values'"),
        ("wrong attrs", "attrs\\[\"feature_type\"\\] = 'rank'"),
        ("other columns", 'feature_extractor returns "median", "top", but'),
    ],
)
def test_tables_of_another_form_stop_with_three_lines(
    result: RubberRulerResult, change: str, message: str
) -> None:
    """Tables that do not have the form of result.performance and
    result.features stop before any stage, with what differs and how to fix it."""
    transformed = [p.instance_id for p in result.split.transformed_test]
    performance = result.performance
    features: dict[str, pd.DataFrame] = dict(result.features)
    portfolio: object = PORTFOLIO
    extractor: object = extract
    if change == "no portfolio":
        portfolio = None
        performance = performance.drop(index=transformed)
    elif change == "no extractor":
        extractor = None
        features = {"value": features["value"].iloc[:3], "rank": features["rank"]}
    elif change == "other rows":
        performance = performance.iloc[:1].rename(index=lambda _: "f24_i001_d2")
    elif change == "other algorithms":
        performance = performance.rename(columns={"random": "other"})
    elif change == "raw values":
        performance = performance * 100 + 1
    elif change == "wrong key":
        features = {"values": features["value"]}
    elif change == "wrong attrs":
        features = {"value": features["rank"]}
    elif change == "other columns":
        extractor = other_extract
        features = {"value": features["value"].iloc[:3]}
    with pytest.raises(rr.RubberRulerError, match=message) as raised:
        tiny_run(portfolio, extractor, performance=performance, features=features)
    assert_three_lines(raised.value)
