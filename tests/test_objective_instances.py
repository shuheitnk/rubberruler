import numpy as np
import pandas as pd
import pytest
from helpers import PORTFOLIO, assert_three_lines, extract, tiny_run

import rubberruler as rr
from rubberruler._instances import transformation_of
from rubberruler._seeding import generator

D = 2
LOWER = np.full(D, -5.0)
UPPER = np.full(D, 5.0)


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def rastrigin(x: np.ndarray) -> float:
    return float(10 * x.size + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def shifted(shift: float) -> object:
    def objective(x: np.ndarray) -> float:
        return float(np.sum((x - shift) ** 2) + np.sum(np.abs(x)))

    return objective


def objectives(count: int) -> dict[str, tuple[object, np.ndarray, np.ndarray]]:
    made: dict[str, tuple[object, np.ndarray, np.ndarray]] = {
        "sphere": (sphere, LOWER, UPPER),
        "rastrigin": (rastrigin, LOWER, UPPER),
    }
    for index in range(count - 2):
        made[f"shifted{index:02d}"] = (shifted(index / 4 - 2), LOWER, UPPER)
    return made


def test_the_objectives_are_split_one_in_four_by_the_seed() -> None:
    """One objective in 4, rounded half up and at least one, is a test
    instance, chosen by the seed among the sorted names; ids are the names."""
    for count, tests in ((2, 1), (5, 1), (6, 2), (24, 6), (26, 7)):
        split = rr.objective_instances(objectives(count))  # pyright: ignore[reportArgumentType]
        assert (len(split.train), len(split.test)) == (count - tests, tests)
        ids = [p.instance_id for p in (*split.train, *split.test)]
        assert sorted(ids) == sorted(objectives(count))
    split = rr.objective_instances(objectives(12), split_seed=1)  # pyright: ignore[reportArgumentType]
    again = rr.objective_instances(objectives(12), split_seed=1)  # pyright: ignore[reportArgumentType]
    assert [p.instance_id for p in split.test] == [p.instance_id for p in again.test]
    tests = {
        rr.objective_instances(objectives(12), split_seed=seed).test[0].instance_id  # pyright: ignore[reportArgumentType]
        for seed in range(8)
    }
    assert len(tests) > 1
    first = split.train[0]
    assert (first.suite, first.function_id, first.dimension) == ("objectives", None, D)
    assert [p.instance_id for p in split.train] == sorted(
        p.instance_id for p in split.train
    )
    transformed = split.transformed_test[0]
    assert transformed.instance_id == split.test[0].instance_id + "~transformed"
    assert transformed.original_instance_id == split.test[0].instance_id


def test_g_is_fitted_as_by_transform_objective_with_the_keys_of_the_instance() -> None:
    """f_min and f_max are the smallest and largest of 1000 x d values at the
    "fmax" sample of the instance keys, or the optimal value if given. The sample
    uses the master seed 42 of the instances, whatever split_seed is."""
    split = rr.objective_instances(
        {"sphere": (sphere, LOWER, UPPER), "rastrigin": (rastrigin, LOWER, UPPER)},
        split_seed=3,
    )  # pyright: ignore[reportArgumentType]
    test = split.test[0]
    name = test.instance_id
    function = {"sphere": sphere, "rastrigin": rastrigin}[name]
    rng = generator("fmax", 42, "objectives", name, D)
    values = [
        function(point) for point in rng.uniform(LOWER, UPPER, size=(1000 * D, D))
    ]
    g = transformation_of(split.transformed_test[0])
    assert g is not None
    assert (g.f_min, g.f_max) == (min(values), max(values))
    known = rr.objective_instances(
        {"sphere": (sphere, LOWER, UPPER), "rastrigin": (rastrigin, LOWER, UPPER)},  # pyright: ignore[reportArgumentType]
        optimal_values={"sphere": 0.0, "rastrigin": 0.0},
        split_seed=3,
    )
    g_known = transformation_of(known.transformed_test[0])
    assert g_known is not None
    assert (g_known.f_min, g_known.f_max) == (0.0, max(values))


def test_test_names_the_test_objectives() -> None:
    """test names the objectives that become test instances, instead of
    the seed; both sets keep the order of the names."""
    split = rr.objective_instances(objectives(6), test=["shifted03", "rastrigin"])  # pyright: ignore[reportArgumentType]
    assert [p.instance_id for p in split.test] == ["rastrigin", "shifted03"]
    assert [p.instance_id for p in split.train] == [
        "shifted00",
        "shifted01",
        "shifted02",
        "sphere",
    ]
    assert [p.instance_id for p in split.transformed_test] == [
        "rastrigin~transformed",
        "shifted03~transformed",
    ]


def test_the_card_shows_how_the_test_instances_were_chosen() -> None:
    """On your objectives, the conditions of the card show
    split_seed right after the suite, or the number of names given as test, within
    80 characters; result.config keeps it, so **result.config repeats the call."""
    settings = {"train_instance_numbers": None, "test_instance_numbers": None}
    splits = {
        "default": rr.objective_instances(objectives(8)),  # pyright: ignore[reportArgumentType]
        "drawn": rr.objective_instances(objectives(8), split_seed=3),  # pyright: ignore[reportArgumentType]
        "named": rr.objective_instances(objectives(8), test=["sphere", "rastrigin"]),  # pyright: ignore[reportArgumentType]
    }
    assert [split.split_seed for split in splits.values()] == [42, 3, None]
    shown = {
        "default": "seed=42, suite=objectives (default bbob), split_seed=42,",
        "drawn": "seed=42, suite=objectives (default bbob), split_seed=3 (default 42),",
        "named": "seed=42, suite=objectives (default bbob), test=2 names,",
    }
    for name, split in splits.items():
        result = tiny_run(instances=split, **settings)
        lines = str(result).split("\n")
        assert shown[name] in lines
        assert all(len(line) <= 80 for line in lines)
        assert result.config["instances"].split_seed == split.split_seed  # pyright: ignore[reportOptionalMemberAccess]
        again = tiny_run(**result.config)  # pyright: ignore[reportArgumentType]
        assert str(again) == str(result)
    reseeded = tiny_run(instances=splits["drawn"], **{**settings, "seed": 7})
    assert reseeded.config["instances"].split_seed == 3  # pyright: ignore[reportOptionalMemberAccess]


def test_the_seed_of_run_test_applies_to_your_objectives() -> None:
    """On your objectives, the seed of run_test replaces the master
    seed 42 of the instances: the same sets, g fitted anew, and other runs; the card
    shows the seed, and **result.config repeats the call."""
    split = rr.objective_instances(objectives(8))  # pyright: ignore[reportArgumentType]
    settings = {"train_instance_numbers": None, "test_instance_numbers": None}
    default = tiny_run(instances=split, **settings)
    other = tiny_run(instances=split, **{**settings, "seed": 7})
    assert default.config["instances"] is split
    assert [p.instance_id for p in other.split.test] == [
        p.instance_id for p in split.test
    ]
    assert {p.seed for p in other.split} == {7}
    g_default = transformation_of(split.transformed_test[0])
    g_other = transformation_of(other.split.transformed_test[0])
    assert g_default is not None and g_other is not None
    assert float(g_default.psi(0.5)) != float(g_other.psi(0.5))
    assert not other.performance.equals(default.performance)
    assert "seed=7 (default 42)" in str(other)
    again = tiny_run(**other.config)  # pyright: ignore[reportArgumentType]
    pd.testing.assert_frame_equal(again.table, other.table, check_exact=True)


def test_another_split_reuses_the_runs_and_features_of_each_objective(
    tmp_path: object,
) -> None:
    """The runs, the features, and g of an objective do
    not depend on split_seed, so another split reads the runs of every
    objective from the cache, and only the transformed test instances of objectives
    that are new in the test set run."""
    from rubberruler._cache import check_cache_dir
    from rubberruler._performance import plan_runs

    portfolio = {"random": PORTFOLIO["random"], "center": PORTFOLIO["center"]}
    first = rr.objective_instances(objectives(8), split_seed=1)  # pyright: ignore[reportArgumentType]
    second = rr.objective_instances(objectives(8), split_seed=2)  # pyright: ignore[reportArgumentType]
    new = {p.instance_id for p in second.test} - {p.instance_id for p in first.test}
    assert new
    rr.measure_performance(
        portfolio,  # pyright: ignore[reportArgumentType]
        first,
        budget_per_dimension=10,
        n_runs=2,
        n_jobs=1,
        cache_dir=tmp_path,  # pyright: ignore[reportArgumentType]
    )
    plan = plan_runs(
        portfolio,  # pyright: ignore[reportArgumentType]
        list(second),
        budget_per_dimension=10,
        n_runs=2,
        cache_dir=check_cache_dir(tmp_path),
    )
    to_run = {task.instance.instance_id for task in plan.tasks}
    assert to_run == {f"{name}~transformed" for name in new}
    one = rr.compute_features(extract, first, "value", n_jobs=1)
    other = rr.compute_features(extract, second, "value", n_jobs=1)
    shared = one.index.intersection(other.index)
    assert len(shared) == 8 + len(first.test) - len(new)
    pd.testing.assert_frame_equal(one.loc[shared], other.loc[shared], check_exact=True)


def test_the_tables_of_one_split_serve_another() -> None:
    """result.features of one split, given to run_test on another
    split, gives the rows of the shared instances, and the table is the same as
    without it."""
    first = tiny_run(
        instances=rr.objective_instances(objectives(8), split_seed=1),  # pyright: ignore[reportArgumentType]
        train_instance_numbers=None,
        test_instance_numbers=None,
    )
    second_split = rr.objective_instances(objectives(8), split_seed=2)  # pyright: ignore[reportArgumentType]
    settings = {"train_instance_numbers": None, "test_instance_numbers": None}
    fresh = tiny_run(instances=second_split, **settings)
    reused = tiny_run(instances=second_split, features=first.features, **settings)
    pd.testing.assert_frame_equal(reused.table, fresh.table, check_exact=True)
    shared = len(
        first.features["value"].index.intersection(fresh.features["value"].index)
    )
    assert f"value features given for {shared} of 10 instances" in str(reused)


def test_run_test_on_your_objectives_shows_the_suite_and_repeats() -> None:
    """run_test(instances=...) answers on your objectives, the card
    names the suite, and **result.config repeats the call."""
    split = rr.objective_instances(objectives(12))  # pyright: ignore[reportArgumentType]
    result = tiny_run(
        instances=split, train_instance_numbers=None, test_instance_numbers=None
    )
    lines = str(result).split("\n")
    assert "suite=objectives (default bbob)" in lines[-1] or any(
        "suite=objectives (default bbob)" in line for line in lines
    )
    assert "9 training instances, 3 test instances," in str(result)
    config = result.config
    assert config["suite"] == "objectives"
    assert config["instances"] is split
    assert config["dimension"] == D
    again = tiny_run(**config)  # pyright: ignore[reportArgumentType]
    pd.testing.assert_frame_equal(again.table, result.table, check_exact=True)
    assert list(result.performance.index[:2]) == [
        p.instance_id for p in split.train[:2]
    ]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"seed": 7}, r"seed=7 was given together with instances, whose seed is 42"),
        ({"suite": "ma-bbob"}, "suite='ma-bbob' was given together with instances"),
        ({"dimension": 5}, "dimension=5 was given together with instances"),
    ],
)
def test_settings_that_the_instances_decide_stop(
    changes: dict[str, object], message: str
) -> None:
    """A suite, dimension, instance numbers, or seed other than the
    default and other than that of the instances stops before any stage; the seed
    only for instances of BBOB and MA-BBOB."""
    split = (
        rr.bbob_instances(
            dimension=2, train_instance_numbers=(11,), test_instance_numbers=(1,)
        )
        if "seed" in changes
        else rr.objective_instances(objectives(3))  # pyright: ignore[reportArgumentType]
    )
    settings = {
        "dimension": 2,
        "train_instance_numbers": None,
        "test_instance_numbers": None,
        **changes,
    }
    with pytest.raises(rr.RubberRulerError, match=message) as raised:
        tiny_run(instances=split, **settings)
    assert_three_lines(raised.value)


def test_suite_objectives_without_instances_stops() -> None:
    """suite="objectives" needs instances from rr.objective_instances."""
    with pytest.raises(rr.RubberRulerError, match="received no instances") as raised:
        tiny_run(suite="objectives")
    assert_three_lines(raised.value)


def raises_on_the_right(x: np.ndarray) -> float:
    if x[0] > 4.9:
        raise ValueError("outside the domain of the model")
    return float(np.sum(x**2))


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"objectives": {"sphere": (sphere, LOWER, UPPER)}}, "has 1 objective."),
        ({"objectives": [sphere, rastrigin]}, "objectives is a list."),
        (
            {
                "objectives": {
                    "a": (sphere, LOWER, UPPER),
                    "b": (sphere, np.full(3, -1), np.full(3, 1)),
                }
            },
            'objectives has d=2 for "a" and d=3 for "b"',
        ),
        (
            {"objectives": {"": (sphere, LOWER, UPPER), "b": (sphere, LOWER, UPPER)}},
            "has the name ''",
        ),
        (
            {
                "objectives": {
                    "a~transformed": (sphere, LOWER, UPPER),
                    "b": (sphere, LOWER, UPPER),
                }
            },
            "has the name",
        ),
        (
            {"objectives": {"a": sphere, "b": (sphere, LOWER, UPPER)}},
            r'objectives\["a"\] is not',
        ),
        (
            {"objectives": {"a": (sphere, UPPER, LOWER), "b": (sphere, LOWER, UPPER)}},
            "is not below",
        ),
        (
            {
                "objectives": {
                    "a": (sphere, LOWER, UPPER),
                    "b": (sphere, LOWER, UPPER),
                },
                "optimal_values": {"c": 0.0},
            },
            "optimal_values has 'c', which is not in objectives",
        ),
        (
            {
                "objectives": {
                    "a": (sphere, LOWER, UPPER),
                    "b": (sphere, LOWER, UPPER),
                },
                "optimal_values": {"a": 3.0, "b": 3.0},
            },
            "optimal_values gives 3.0 for",
        ),
        ({"objectives": objectives(4), "test": "sphere"}, "test is a str."),
        (
            {"objectives": objectives(4), "test": ["sphere", "cube"]},
            "test has 'cube', which is not in objectives",
        ),
        (
            {"objectives": objectives(4), "test": ["sphere", "sphere"]},
            "test has 'sphere' more than once",
        ),
        ({"objectives": objectives(4), "test": []}, "test has 0 of the 4 objectives"),
        (
            {"objectives": objectives(2), "test": ["sphere", "rastrigin"]},
            "test has 2 of the 2 objectives",
        ),
        (
            {"objectives": objectives(4), "test": ["sphere"], "split_seed": 7},
            "received both test and split_seed=7",
        ),
        ({"objectives": objectives(4), "split_seed": -1}, "split_seed=-1 is not"),
        (
            {
                "objectives": {
                    "a": (raises_on_the_right, LOWER, UPPER),
                    "b": (sphere, LOWER, UPPER),
                },
                "split_seed": 0,
            },
            'objective "a" raised ValueError',
        ),
    ],
)
def test_wrong_objectives_stop_with_three_lines(
    arguments: dict[str, object], message: str
) -> None:
    """Each wrong input stops with the three-line error."""
    objectives_argument = arguments.pop("objectives")
    with pytest.raises(rr.RubberRulerError, match=message) as raised:
        split = rr.objective_instances(objectives_argument, **arguments)  # pyright: ignore[reportArgumentType]
        # An objective that raises only at some points is found when g is fitted
        # (on a test objective) or in the runs; both "a" and "b" are tried here.
        for instance in (*split.train, *split.test):
            rr.transform(instance)
    assert_three_lines(raised.value)


def test_a_changed_objective_runs_again(tmp_path: object) -> None:
    """The runs of an objective are cached under its identity:
    the same objective is read from the cache, and a changed one runs again."""
    first = rr.objective_instances(objectives(3))  # pyright: ignore[reportArgumentType]
    portfolio = {"random": PORTFOLIO["random"], "center": PORTFOLIO["center"]}
    settings = {
        "budget_per_dimension": 10,
        "n_runs": 2,
        "n_jobs": 1,
        "cache_dir": tmp_path,
    }
    before = rr.measure_performance(portfolio, first, **settings)  # pyright: ignore[reportArgumentType]
    cached = rr.measure_performance(portfolio, first, **settings)  # pyright: ignore[reportArgumentType]
    pd.testing.assert_frame_equal(before, cached, check_exact=True)
    changed = objectives(3)
    changed["sphere"] = (lambda x: float(np.sum(x**2)) + 100.0, LOWER, UPPER)
    second = rr.objective_instances(changed)  # pyright: ignore[reportArgumentType]
    plan_counts = []
    from rubberruler._cache import check_cache_dir
    from rubberruler._performance import plan_runs

    for split in (first, second):
        plan = plan_runs(
            portfolio,  # pyright: ignore[reportArgumentType]
            list(split),
            budget_per_dimension=10,
            n_runs=2,
            cache_dir=check_cache_dir(tmp_path),
        )
        plan_counts.append(len(plan.cached))
    assert plan_counts[0] == len(list(first)) * 2 * 2
    assert plan_counts[1] < plan_counts[0]
    assert rr.compute_features(extract, first, "rank", n_jobs=1).shape[0] == 4
