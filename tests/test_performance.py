import _thread
import logging
import subprocess
import sys
import threading
import time
from collections.abc import Callable

import algorithms
import numpy as np
import pandas as pd
import psutil
import pytest
from helpers import assert_three_lines
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import rubberruler as rr
from rubberruler import _performance
from rubberruler._instances import (
    InstanceObjective,
    InstanceSplit,
    make_split,
    transformation_of,
)
from rubberruler._performance import (
    average_superiority,
    failure_warning,
    run_once,
    run_seed,
    run_task,
    superiority,
)

run_values = st.one_of(
    st.floats(-1e6, 1e6), st.just(float("inf")), st.sampled_from([0.0, 1.0])
)


@pytest.fixture(scope="module")
def split() -> InstanceSplit:
    return make_split(2, (1,), (2,), seed=3, function_ids=(1, 15))


@given(
    first=arrays(np.float64, st.integers(1, 8), elements=run_values),
    second=arrays(np.float64, st.integers(1, 8), elements=run_values),
)
def test_superiority_of_a_pair_adds_up_to_1(
    first: np.ndarray, second: np.ndarray
) -> None:
    """PS(a, b) + PS(b, a) = 1."""
    assert superiority(first, second) + superiority(second, first) == pytest.approx(1)


@given(
    runs=arrays(
        np.float64,
        st.tuples(st.integers(2, 6), st.integers(1, 6)),
        elements=run_values,
    )
)
def test_performance_of_a_portfolio_adds_up_to_half_its_size(runs: np.ndarray) -> None:
    """The sum of J(p, a) over the portfolio is |A| / 2."""
    j = average_superiority(runs)
    assert j.sum() == pytest.approx(runs.shape[0] / 2)
    assert np.all((j >= 0) & (j <= 1))


def test_superiority_counts_ties_as_half_including_inf() -> None:
    """Equal values, and +inf with +inf, count as 0.5."""
    inf = float("inf")
    assert superiority(np.array([1.0, inf]), np.array([1.0, inf])) == 0.5
    assert superiority(np.array([1.0]), np.array([2.0, inf])) == 1.0
    assert superiority(np.array([inf]), np.array([inf])) == 0.5


def test_a_run_records_the_smallest_value_within_the_budget(
    split: InstanceSplit,
) -> None:
    """A run's value is the smallest value seen within budget B."""
    instance = split.train[0]
    record, failure = run_once(
        algorithms.random_search, InstanceObjective(instance), instance, 20, seed=5
    )
    assert failure is None
    assert record.evaluations == 20
    rng = np.random.default_rng(5)
    points = rng.uniform(instance.lower_bounds, instance.upper_bounds, size=(20, 2))
    assert record.best == InstanceObjective(instance).many(points).min()


def test_failed_runs_keep_the_best_value_so_far(split: InstanceSplit) -> None:
    """A failed run keeps its best value, or +inf if it never evaluated."""
    instance = split.train[0]
    objective = InstanceObjective(instance)
    record, failure = run_once(
        algorithms.raises_after_three, objective, instance, 20, seed=0
    )
    assert failure == "ValueError: boom"
    assert record.evaluations == 3
    assert record.best == objective.one(np.ones(2))
    record, failure = run_once(
        algorithms.returns_without_evaluating, objective, instance, 20, seed=0
    )
    assert failure == "returned without evaluating problem"
    assert record.best == float("inf")


def test_the_budget_ends_algorithms_that_catch_every_exception(
    split: InstanceSplit,
) -> None:
    """The budget exception is not an Exception, so the run still ends."""
    instance = split.train[0]
    record, failure = run_once(
        algorithms.swallows_exceptions, InstanceObjective(instance), instance, 30, 1
    )
    assert failure is None
    assert record.evaluations == 30


def test_an_ioh_style_algorithm_runs_without_glue(split: InstanceSplit) -> None:
    """bounds.lb, meta_data.n_variables, and state.evaluations work."""
    instance = split.train[0]
    record, failure = run_once(
        algorithms.ioh_style_search, InstanceObjective(instance), instance, 100, 1
    )
    assert failure is None
    assert record.evaluations == 10


def test_run_seeds_come_from_derive_seed() -> None:
    """The run seed is derive_seed("run", ...), the same after transform."""
    split = make_split(10, (11,), (1,), seed=42, function_ids=(1,))
    assert run_seed(split.test[0], "cma-es", 0) == 374886278
    assert run_seed(split.transformed_test[0], "cma-es", 0) == 374886278


def test_global_generators_are_seeded_before_each_run(split: InstanceSplit) -> None:
    """An algorithm that uses numpy.random repeats its runs exactly."""
    instance = split.train[0]
    objective = InstanceObjective(instance)
    first, _ = run_once(algorithms.global_random_search, objective, instance, 15, 9)
    np.random.seed(123)  # noqa: NPY002
    second, _ = run_once(algorithms.global_random_search, objective, instance, 15, 9)
    assert first.best == second.best


def test_measure_performance_gives_j_for_each_instance_and_algorithm(
    split: InstanceSplit,
) -> None:
    """One row per instance_id and one column per algorithm."""
    portfolio = {
        "random": algorithms.random_search,
        "batch": algorithms.batch_random_search,
        "center": algorithms.center,
    }
    performance = rr.measure_performance(
        portfolio, split, budget_per_dimension=10, n_runs=3, n_jobs=1, cache_dir=None
    )
    assert list(performance.columns) == ["random", "batch", "center"]
    assert performance.index.name == "instance_id"
    assert list(performance.index) == [instance.instance_id for instance in split]
    assert performance.to_numpy().dtype == np.float64
    np.testing.assert_allclose(performance.sum(axis=1), 1.5)


def test_transformed_runs_see_g_of_the_original_values(split: InstanceSplit) -> None:
    """The same seeds evaluate the same points on both instances."""
    original, transformed = split.test[0], split.transformed_test[0]
    g = transformation_of(transformed)
    assert g is not None
    budget = 25
    for search in (algorithms.random_search, algorithms.batch_random_search):
        before = run_task(original, "a", search, budget, (0, 1, 2))
        after = run_task(transformed, "a", search, budget, (0, 1, 2))
        for run_before, run_after in zip(before, after, strict=True):
            assert run_after.best == g(np.array([run_before.best]))[0]
    portfolio = {
        "random": algorithms.random_search,
        "batch": algorithms.batch_random_search,
    }
    table = rr.measure_performance(
        portfolio,
        [original, transformed],
        budget_per_dimension=10,
        n_runs=5,
        n_jobs=1,
        cache_dir=None,
    )
    np.testing.assert_array_equal(
        table.loc[original.instance_id], table.loc[transformed.instance_id]
    )


def test_n_jobs_does_not_change_the_performance(split: InstanceSplit) -> None:
    """n_jobs=1 and n_jobs=2 give the same J."""
    portfolio = {
        "random": algorithms.random_search,
        "global": algorithms.global_random_search,
        "batch": algorithms.batch_random_search,
    }
    results = [
        rr.measure_performance(
            portfolio,
            split,
            budget_per_dimension=10,
            n_runs=4,
            n_jobs=n_jobs,
            cache_dir=None,
        )
        for n_jobs in (1, 2)
    ]
    pd.testing.assert_frame_equal(results[0], results[1])


def loky_workers() -> list[psutil.Process]:
    """The worker processes of joblib's loky backend under this process."""
    workers: list[psutil.Process] = []
    for child in psutil.Process().children(recursive=True):
        try:
            command = " ".join(child.cmdline())
        except psutil.Error:
            continue
        if "loky" in command and "resource_tracker" not in command:
            workers.append(child)
    return workers


def test_no_worker_is_left_after_the_runs(split: InstanceSplit) -> None:
    """The worker processes end with the runs instead of idling (they
    would hold memory, and on Windows the working folder)."""
    portfolio = {"random": algorithms.random_search, "center": algorithms.center}
    rr.measure_performance(
        portfolio, split, budget_per_dimension=5, n_runs=2, n_jobs=2, cache_dir=None
    )
    assert loky_workers() == []


def test_failed_runs_are_logged_and_counted(
    split: InstanceSplit, caplog: pytest.LogCaptureFixture
) -> None:
    """The first failed runs are logged, then the warning with the count."""
    portfolio = {
        "flaky": algorithms.fails_on_odd_seeds,
        "random": algorithms.random_search,
    }
    with caplog.at_level(logging.WARNING, logger="rubberruler"):
        rr.measure_performance(
            portfolio, split, budget_per_dimension=5, n_runs=8, n_jobs=1, cache_dir=None
        )
    failed = [
        (instance.instance_id, run)
        for instance in split
        for run in range(8)
        if run_seed(instance, "flaky", run) % 2 == 1
    ]
    messages = [record.getMessage() for record in caplog.records]
    first_id, first_run = failed[0]
    assert messages[: min(len(failed), 10)] == [
        f'run {run} of "flaky" on {instance_id} failed: RuntimeError: odd seed'
        for instance_id, run in failed[:10]
    ]
    assert messages[-1] == (
        f"{len(failed)} runs failed; their best value so far was used "
        "(+inf if there was none; see the log)"
    )
    assert f'run {first_run} of "flaky" on {first_id}' in messages[0]


@pytest.mark.parametrize(
    ("algorithm", "n_jobs", "message"),
    [
        (algorithms.returns_without_evaluating, 1, "returned without evaluating"),
        (algorithms.raises_after_three, 1, "raised ValueError"),
    ],
)
def test_broken_algorithms_stop_before_the_runs(
    split: InstanceSplit, algorithm: object, n_jobs: int, message: str
) -> None:
    """The pre-check stops broken algorithms before any run."""
    algorithms.calls["counted_search"] = 0
    portfolio = {"counted": algorithms.counted_search, "broken": algorithm}
    with pytest.raises(rr.RubberRulerError, match=message) as raised:
        rr.measure_performance(
            portfolio,  # pyright: ignore[reportArgumentType]
            split,
            budget_per_dimension=10,
            n_runs=3,
            n_jobs=n_jobs,
            cache_dir=None,
        )
    assert_three_lines(raised.value)
    assert '"broken"' in str(raised.value)
    assert algorithms.calls["counted_search"] == 1


def test_the_original_exception_is_kept_as_the_cause(split: InstanceSplit) -> None:
    """The pre-check error keeps the algorithm's exception as __cause__."""
    portfolio = {"a": algorithms.random_search, "b": algorithms.raises_after_three}
    with pytest.raises(rr.RubberRulerError) as raised:
        rr.measure_performance(portfolio, split, n_jobs=1, cache_dir=None)
    assert isinstance(raised.value.__cause__, ValueError)


def test_algorithms_that_cannot_be_pickled_stop_only_with_workers(
    split: InstanceSplit,
) -> None:
    """With n_jobs other than 1, algorithms must go through cloudpickle."""
    lock = threading.Lock()

    def locked_search(problem: rr.Problem) -> None:
        with lock:
            algorithms.random_search(problem)

    portfolio = {"random": algorithms.random_search, "locked": locked_search}
    with pytest.raises(rr.RubberRulerError, match="cannot be sent") as raised:
        rr.measure_performance(
            portfolio, split, budget_per_dimension=5, n_runs=1, n_jobs=2, cache_dir=None
        )
    assert_three_lines(raised.value)
    rr.measure_performance(
        portfolio, split, budget_per_dimension=5, n_runs=1, n_jobs=1, cache_dir=None
    )


def test_slow_algorithms_stop_after_the_time_limit(
    split: InstanceSplit, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-check run must finish within the time limit."""
    monkeypatch.setattr(_performance, "PRECHECK_SECONDS", 0.2)
    portfolio = {"random": algorithms.random_search, "sleeps": algorithms.sleeps}
    with pytest.raises(rr.RubberRulerError, match="did not finish within") as raised:
        rr.measure_performance(portfolio, split, n_jobs=1, cache_dir=None)
    assert_three_lines(raised.value)


def pre_check_threads_end(seconds: float) -> bool:
    """Whether the threads of pre-check runs are gone within ``seconds``."""
    deadline = time.monotonic() + seconds
    while any(t.name == "rubberruler-pre-check" for t in threading.enumerate()):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def sleeps_near_a_lock() -> Callable[[rr.Problem], None]:
    """An endless algorithm that cloudpickle cannot send: it refers to a lock."""
    lock = threading.Lock()

    def algorithm(problem: rr.Problem) -> None:
        assert not lock.locked()
        algorithms.sleeps(problem)

    return algorithm


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_the_time_limit_stops_algorithms_that_cannot_be_pickled(
    split: InstanceSplit, monkeypatch: pytest.MonkeyPatch, n_jobs: int
) -> None:
    """The limit needs no other process, so it holds for any n_jobs."""
    monkeypatch.setattr(_performance, "PRECHECK_SECONDS", 0.2)
    portfolio = {"random": algorithms.random_search, "locked": sleeps_near_a_lock()}
    with pytest.raises(rr.RubberRulerError, match="did not finish within"):
        rr.measure_performance(portfolio, split, n_jobs=n_jobs, cache_dir=None)


def test_the_time_limit_holds_in_any_thread(
    split: InstanceSplit, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The limit holds when rubberruler is called from another thread."""
    monkeypatch.setattr(_performance, "PRECHECK_SECONDS", 0.2)
    portfolio = {"random": algorithms.random_search, "sleeps": algorithms.sleeps}
    errors: list[rr.RubberRulerError] = []

    def call() -> None:
        try:
            rr.measure_performance(portfolio, split, n_jobs=1, cache_dir=None)
        except rr.RubberRulerError as error:
            errors.append(error)

    caller = threading.Thread(target=call)
    caller.start()
    caller.join(timeout=30)
    assert not caller.is_alive()
    assert len(errors) == 1


@pytest.mark.parametrize("algorithm", [algorithms.sleeps, algorithms.evaluates_slowly])
def test_a_pre_check_run_past_the_time_limit_ends(
    split: InstanceSplit, monkeypatch: pytest.MonkeyPatch, algorithm: object
) -> None:
    """After the limit, the run's thread ends soon instead of running on.

    sleeps never evaluates again, so only the exception sent to its thread stops it;
    evaluates_slowly would take a second for its budget, but its next evaluation
    raises.
    """
    monkeypatch.setattr(_performance, "PRECHECK_SECONDS", 0.2)
    portfolio = {"random": algorithms.random_search, "slow": algorithm}
    with pytest.raises(rr.RubberRulerError, match="did not finish within"):
        rr.measure_performance(portfolio, split, n_jobs=1, cache_dir=None)  # pyright: ignore[reportArgumentType]
    assert pre_check_threads_end(0.5)


ASLEEP_IN_C = """
import time

import rubberruler as rr
from rubberruler import _performance
from rubberruler._instances import make_split

_performance.PRECHECK_SECONDS = 1
split = make_split(2, (1,), (2,), seed=3, function_ids=(1,))
portfolio = {
    "asleep": lambda problem: time.sleep(3600),
    "center": lambda problem: problem([0.0, 0.0]),
}
try:
    rr.measure_performance(portfolio, split, n_jobs=1, cache_dir=None)
except rr.RubberRulerError as error:
    print(str(error).splitlines()[0])
"""


def test_a_run_asleep_in_c_does_not_keep_the_process_alive() -> None:
    """The exception sent to the thread cannot reach time.sleep(3600), but
    the thread is a daemon, so the script ends right after the error."""
    output = subprocess.run(
        [sys.executable, "-c", ASLEEP_IN_C],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    ).stdout
    assert output.startswith('algorithm "asleep" did not finish within 1 seconds')


def test_ctrl_c_during_the_pre_check_ends_its_run(split: InstanceSplit) -> None:
    """Ctrl-C while the pre-check waits raises KeyboardInterrupt at once."""
    portfolio = {"random": algorithms.random_search, "sleeps": algorithms.sleeps}
    timer = threading.Timer(0.3, _thread.interrupt_main)
    start = time.monotonic()
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            rr.measure_performance(portfolio, split, n_jobs=1, cache_dir=None)
    finally:
        timer.cancel()
    assert time.monotonic() - start < 5
    assert pre_check_threads_end(1)


@pytest.mark.slow
def test_an_endless_algorithm_stops_after_60_seconds(split: InstanceSplit) -> None:
    """The real limit, with workers and an algorithm they cannot take."""
    portfolio = {"random": algorithms.random_search, "locked": sleeps_near_a_lock()}
    start = time.monotonic()
    with pytest.raises(
        rr.RubberRulerError, match="did not finish within 60 seconds"
    ) as raised:
        rr.measure_performance(portfolio, split, n_jobs=-1, cache_dir=None)
    assert 60 <= time.monotonic() - start < 70
    assert_three_lines(raised.value)


@pytest.mark.parametrize(
    ("portfolio", "message"),
    [
        ([algorithms.random_search], "portfolio is a list"),
        ({"random": algorithms.random_search}, "portfolio has 1 algorithm"),
        ({"": algorithms.random_search, "b": algorithms.center}, "the name ''"),
        ({"a": algorithms.random_search, "b": "center"}, 'portfolio\\["b"\\] is a str'),
    ],
)
def test_bad_portfolios_stop(
    split: InstanceSplit, portfolio: object, message: str
) -> None:
    """portfolio needs at least 2 named functions."""
    with pytest.raises(rr.RubberRulerError, match=message) as raised:
        rr.measure_performance(portfolio, split, cache_dir=None)  # pyright: ignore[reportArgumentType]
    assert_three_lines(raised.value)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"budget_per_dimension": 0}, "budget_per_dimension=0"),
        ({"n_runs": 0}, "n_runs=0"),
        ({"n_jobs": 0}, "n_jobs=0"),
        ({"n_jobs": 1.5}, "n_jobs=1.5"),
    ],
)
def test_bad_settings_stop(
    split: InstanceSplit, arguments: dict[str, object], message: str
) -> None:
    """Settings outside their allowed values stop."""
    portfolio = {"a": algorithms.random_search, "b": algorithms.center}
    with pytest.raises(rr.RubberRulerError, match=message) as raised:
        rr.measure_performance(portfolio, split, cache_dir=None, **arguments)  # pyright: ignore[reportArgumentType]
    assert_three_lines(raised.value)


@pytest.mark.parametrize(
    ("count", "line"),
    [
        (
            1,
            "1 run failed; its best value so far was used "
            "(+inf if there was none; see the log)",
        ),
        (
            2,
            "2 runs failed; their best value so far was used "
            "(+inf if there was none; see the log)",
        ),
    ],
)
def test_the_failure_warning_is_singular_for_one_run(count: int, line: str) -> None:
    """The warning about failed runs uses the singular for one run."""
    assert failure_warning(count) == line
