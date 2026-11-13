"""MA-BBOB and the upper bound of instance numbers."""

import os
from pathlib import Path
from typing import Any

import ioh
import numpy as np
import pandas as pd
import pytest
from helpers import PORTFOLIO, assert_three_lines, compute_features, extract, tiny_run
from sklearn.ensemble import RandomForestRegressor

import rubberruler as rr
from rubberruler._card import DEFAULT_CONFIG, RunConfig, render, select
from rubberruler._features import precheck_instances
from rubberruler._instances import InstanceObjective, make_split, transformation_of
from rubberruler._performance import run_seed
from rubberruler._run_test import RubberRulerResult, run
from rubberruler._seeding import derive_seed
from rubberruler._transformations import log_density, sample_psi

# ioh's stubs do not list ManyAffine.
IOH_PROBLEMS: Any = ioh.problem
MA_BBOB_SETTINGS: dict[str, object] = {
    "suite": "ma-bbob",
    "train_instance_numbers": tuple(range(1, 9)),
    "test_instance_numbers": (9, 10, 11),
}


@pytest.fixture(scope="module")
def result() -> RubberRulerResult:
    return tiny_run(**MA_BBOB_SETTINGS)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("domain", "keys", "expected"),
    [
        ("transform", ("ma-bbob", 1, 10), 972812586),
        ("fmax", ("ma-bbob", 1, 10), 1130706760),
        ("features", ("ma-bbob", 1, 10), 4219328329),
        ("run", ("ma-bbob", 1, 10, "cma-es", 0), 113223792),
    ],
)
def test_derive_seed_returns_the_fixed_values_of_ma_bbob(
    domain: str, keys: tuple[str | int, ...], expected: int
) -> None:
    """derive_seed returns the fixed values of MA-BBOB."""
    assert derive_seed(domain, 42, *keys) == expected


def test_psi_of_ma_bbob_matches_the_fixed_values() -> None:
    """m and psi for seed=42, MA-BBOB instance number 1, dimension 10."""
    assert len(log_density(42, None, 1, 10).x) == 7
    psi = sample_psi(42, None, 1, 10)
    for z, expected in (
        (0.25, 0.416098557361484),
        (0.5, 0.5346717129994829),
        (0.75, 0.7681696027474796),
    ):
        assert float(psi(z)) == pytest.approx(expected, abs=1e-9)


def test_f_min_and_f_max_of_ma_bbob_match_the_fixed_values() -> None:
    """f_min is ioh's optimum of MA-BBOB, not 0, and f-hat-max
    is the largest of 10^4 uniform values, for seed 42, instance 1, d = 10."""
    split = make_split(10, (2,), (1,), seed=42, suite="ma-bbob")
    g = transformation_of(split.transformed_test[0])
    assert g is not None
    assert g.f_min == pytest.approx(1.837515859581564e-12, rel=1e-9)
    assert g.f_min == IOH_PROBLEMS.ManyAffine(instance=1, n_variables=10).optimum.y
    assert g.f_max == pytest.approx(38.945837019309465, abs=1e-6)


def test_ma_bbob_instances_have_the_default_sizes_and_ids() -> None:
    """2400, 240, and 240 instances by default, one per instance number,
    in the order of the numbers, without a function ID."""
    split = rr.ma_bbob_instances()
    assert (len(split.train), len(split.test), len(split.transformed_test)) == (
        2400,
        240,
        240,
    )
    assert [p.instance_number for p in split.train] == list(range(1, 2401))
    assert [p.instance_number for p in split.test] == list(range(2401, 2641))
    first, test = split.train[0], split.test[0]
    assert first.instance_id == "mabbob_i001_d10"
    assert (first.suite, first.function_id, first.seed) == ("ma-bbob", None, 42)
    assert test.instance_id == "mabbob_i2401_d10"
    assert split.transformed_test[0].instance_id == "mabbob_i2401_d10~transformed"
    assert split.transformed_test[0].original_instance_id == "mabbob_i2401_d10"
    np.testing.assert_array_equal(first.lower_bounds, np.full(10, -5.0))
    np.testing.assert_array_equal(first.upper_bounds, np.full(10, 5.0))
    unsorted = rr.ma_bbob_instances(
        dimension=2, train_instance_numbers=[1000, 3], test_instance_numbers=[7]
    )
    assert [p.instance_id for p in unsorted.train] == [
        "mabbob_i003_d2",
        "mabbob_i1000_d2",
    ]


def test_bbob_keeps_its_suite_keys_and_ids() -> None:
    """BBOB keeps the keys "bbob", function, instance number, d; MA-BBOB
    uses "ma-bbob", instance number, d; the ids and seeds of the two differ."""
    bbob = make_split(2, (11,), (), seed=42, function_ids=(1,)).train[0]
    ma_bbob = make_split(2, (11,), (), seed=42, suite="ma-bbob").train[0]
    assert (bbob.suite, bbob.function_id, bbob.instance_id) == (
        "bbob",
        1,
        "f01_i011_d2",
    )
    assert run_seed(bbob, "a", 0) == derive_seed("run", 42, "bbob", 1, 11, 2, "a", 0)
    assert run_seed(ma_bbob, "a", 0) == derive_seed("run", 42, "ma-bbob", 11, 2, "a", 0)
    assert ma_bbob.instance_id == "mabbob_i011_d2"


def test_the_objective_is_ma_bbob_of_ioh() -> None:
    """The objective of an MA-BBOB instance is ioh's ManyAffine, for single
    points and for batches; a transformed instance gives g(f)."""
    split = make_split(3, (7,), (8,), seed=5, suite="ma-bbob")
    points = np.random.default_rng(0).uniform(-5, 5, size=(6, 3))
    direct = IOH_PROBLEMS.ManyAffine(instance=7, n_variables=3)
    expected = np.array([direct(point) for point in points])
    objective = InstanceObjective(split.train[0])
    np.testing.assert_array_equal(objective.many(points), expected)
    assert objective.one(points[0]) == expected[0]
    g = transformation_of(split.transformed_test[0])
    assert g is not None
    other = IOH_PROBLEMS.ManyAffine(instance=8, n_variables=3)
    np.testing.assert_array_equal(
        InstanceObjective(split.transformed_test[0]).many(points),
        g(np.array([other(point) for point in points])),
    )


def test_no_value_of_ma_bbob_is_below_f_min() -> None:
    """f_min is the exact minimum, so g never meets z < 0 on MA-BBOB."""
    split = make_split(2, (), range(1, 6), seed=3, suite="ma-bbob")
    points = np.random.default_rng(1).uniform(-5, 5, size=(10_000, 2))
    for original, transformed in zip(split.test, split.transformed_test, strict=True):
        g = transformation_of(transformed)
        assert g is not None
        values = InstanceObjective(original).many(points)
        assert values.min() >= g.f_min
        assert float(g(np.array([g.f_min]))[0]) == 0.0


@pytest.mark.parametrize(
    ("function", "arguments", "message"),
    [
        ("ma", {"dimension": 1}, "dimension=1"),
        ("ma", {"seed": -1}, "seed=-1"),
        ("ma", {"train_instance_numbers": []}, "train_instance_numbers is empty"),
        ("ma", {"train_instance_numbers": [0, 1]}, "contains 0"),
        ("ma", {"train_instance_numbers": [2**31]}, "contains 2147483648"),
        ("ma", {"train_instance_numbers": [3, 3]}, "contains 3 more than once"),
        ("ma", {"train_instance_numbers": [True]}, "contains True"),
        ("ma", {"train_instance_numbers": None}, "is a NoneType"),
        ("ma", {"test_instance_numbers": (5,)}, "share 5"),
        ("bbob", {"test_instance_numbers": [2**31]}, "contains 2147483648"),
    ],
)
def test_bad_settings_stop_on_both_suites(
    function: str, arguments: dict[str, object], message: str
) -> None:
    """Instance numbers outside 1 to 2**31 - 1, and the other
    invalid settings, stop with a three-line error on both suites."""
    make = rr.ma_bbob_instances if function == "ma" else rr.bbob_instances
    with pytest.raises(rr.RubberRulerError, match=message) as raised:
        make(**arguments)  # pyright: ignore[reportArgumentType]
    assert_three_lines(raised.value)


def test_the_largest_instance_number_is_accepted() -> None:
    """2**31 - 1 is a valid instance number on both suites."""
    largest = 2**31 - 1
    for make in (rr.bbob_instances, rr.ma_bbob_instances):
        split = make(
            dimension=2, train_instance_numbers=[largest], test_instance_numbers=[1]
        )
        objective = InstanceObjective(split.train[0])
        assert np.isfinite(objective.one(np.zeros(2)))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"suite": "cec"}, "suite='cec' is not allowed"),
        (
            {
                "suite": "ma-bbob",
                "train_instance_numbers": None,
                "test_instance_numbers": (5,),
            },
            r"share 5\.",
        ),
        (
            {"train_instance_numbers": None, "test_instance_numbers": (11,)},
            r"share 11\.",
        ),
    ],
)
def test_run_test_checks_the_suite_and_takes_its_defaults(
    capsys: pytest.CaptureFixture[str], changes: dict[str, object], message: str
) -> None:
    """An invalid suite stops, and None takes the default instance numbers
    of the suite (1 to 2400 on MA-BBOB, 11 to 110 on BBOB), before any stage."""
    with pytest.raises(rr.RubberRulerError, match=message) as raised:
        tiny_run(**changes)  # pyright: ignore[reportArgumentType]
    assert_three_lines(raised.value)
    assert "[1/4]" not in capsys.readouterr().err


def test_run_test_on_ma_bbob_gives_an_answer(result: RubberRulerResult) -> None:
    """run_test on MA-BBOB uses one instance per instance number,
    and result.config holds the suite and the numbers used."""
    assert result.selected_feature_type == select(result.table)
    assert (len(result.split.train), len(result.split.test)) == (8, 3)
    assert len(result.split.transformed_test) == 3
    assert result.performance.shape == (14, 3)
    assert all(name.startswith("mabbob_i") for name in result.performance.index)
    assert result.config == {
        "suite": "ma-bbob",
        "dimension": 2,
        "train_instance_numbers": tuple(range(1, 9)),
        "test_instance_numbers": (9, 10, 11),
        "budget_per_dimension": 25,
        "n_runs": 3,
        "n_samples_per_dimension": 10,
        "seed": 42,
        "instances": None,
    }


def test_the_config_of_ma_bbob_repeats_the_test(result: RubberRulerResult) -> None:
    """run_test(..., **result.config) repeats the test on MA-BBOB."""
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


def test_the_card_names_the_suite(result: RubberRulerResult) -> None:
    """The card shows suite=ma-bbob and compares the instance numbers with
    the defaults of MA-BBOB."""
    lines = str(result).split("\n")
    conditions = lines[lines.index("3 transformed test instances") + 1 :]
    assert conditions == [
        "seed=42, suite=ma-bbob (default bbob), dimension=2 (default 10),",
        "train_instance_numbers=1-8 (default 1-2400),",
        "test_instance_numbers=9-11 (default 2401-2640),",
        "budget_per_dimension=25 (default 1000), n_runs=3 (default 31),",
        "n_samples_per_dimension=10 (default 50)",
    ]


def test_the_card_hides_the_default_numbers_of_ma_bbob() -> None:
    """With the defaults of MA-BBOB, only the suite follows the seed."""
    config = RunConfig(
        **{
            **DEFAULT_CONFIG,
            "suite": "ma-bbob",
            "train_instance_numbers": tuple(range(1, 2401)),
            "test_instance_numbers": tuple(range(2401, 2641)),
        }
    )
    table = pd.DataFrame(
        {"r2_test": [0.7, 0.6], "r2_transformed_test": [0.4, 0.5]},
        index=["value", "rank"],
    ).assign(degradation=lambda frame: frame["r2_test"] - frame["r2_transformed_test"])
    text = render(
        table,
        select(table),
        n_algorithms=3,
        set_sizes=(2400, 240, 240),
        config=config,
        warnings=(),
    )
    assert text.endswith(
        "3 algorithms, 2400 training instances, 240 test instances,\n"
        "240 transformed test instances\n"
        "seed=42, suite=ma-bbob (default bbob)"
    )


def test_the_extractor_precheck_takes_the_first_24_ma_bbob_instances() -> None:
    """The first instance of each BBOB function and the first 24 MA-BBOB
    instances, in the order given."""
    bbob = make_split(2, (11, 12), (), seed=1, function_ids=(1, 2)).train
    ma_bbob = make_split(2, range(1, 31), (), seed=1, suite="ma-bbob").train
    chosen = precheck_instances([*bbob, *ma_bbob])
    assert [p.instance_id for p in chosen] == [
        "f01_i011_d2",
        "f02_i011_d2",
        *(f"mabbob_i{number:03d}_d2" for number in range(1, 25)),
    ]


def test_an_extractor_that_fails_on_the_24th_instance_stops_before_the_runs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """run_test on MA-BBOB tries the extractor on the first 24 training
    instances, so one that fails on the 24th stops before any run."""
    calls = {"count": 0}

    def fails_on_the_24th(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        calls["count"] += 1  # value and rank of each instance in turn
        if calls["count"] == 47:
            raise ValueError("a plateau")
        return {"mean": float(np.mean(y))}

    with pytest.raises(
        rr.RubberRulerError, match=r"raised ValueError on mabbob_i024_d2 \(value\)"
    ):
        tiny_run(
            extractor=fails_on_the_24th,
            suite="ma-bbob",
            train_instance_numbers=range(1, 31),
            test_instance_numbers=(31,),
        )
    assert "runs (" not in capsys.readouterr().err


def test_runs_of_both_suites_share_one_cache(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Runs of MA-BBOB do not replace runs of BBOB in one cache_dir."""
    tiny_run(cache_dir=tmp_path)
    tiny_run(cache_dir=tmp_path, **MA_BBOB_SETTINGS)  # pyright: ignore[reportArgumentType]
    capsys.readouterr()
    tiny_run(cache_dir=tmp_path)
    tiny_run(cache_dir=tmp_path, **MA_BBOB_SETTINGS)  # pyright: ignore[reportArgumentType]
    err = capsys.readouterr().err
    folder = os.path.join(tmp_path, "")
    assert f"rubberruler: 108 runs (108 cached, 0 to run) in {folder}" in err
    assert f"rubberruler: 126 runs (126 cached, 0 to run) in {folder}" in err


def test_the_components_accept_both_suites_in_one_call() -> None:
    """The component functions do not check that one call uses one suite."""
    bbob = make_split(2, (11,), (), seed=1, function_ids=(1,)).train
    ma_bbob = make_split(2, (11,), (), seed=1, suite="ma-bbob").train
    features = compute_features(
        extract, [*bbob, *ma_bbob], "value", n_samples_per_dimension=5
    )
    assert list(features.index) == ["f01_i011_d2", "mabbob_i011_d2"]


def test_a_suite_of_own_functions_is_pointed_to_objective_instances(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A dict of the user's functions as suite stops with the type
    instead of the whole dict, and the Fix line names rr.objective_instances."""
    with pytest.raises(rr.RubberRulerError, match="suite received a dict") as raised:
        tiny_run(suite={"sphere": np.square})  # pyright: ignore[reportArgumentType]
    assert_three_lines(raised.value)
    assert "rr.objective_instances" in str(raised.value)
    assert "[1/4]" not in capsys.readouterr().err
