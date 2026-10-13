"""Runs of the algorithms and the performance J."""

import contextlib
import ctypes
import logging
import os
import random
import sys
import threading
import time
import warnings
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cloudpickle  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pandas as pd
import psutil  # pyright: ignore[reportMissingTypeStubs]
from joblib import Parallel  # pyright: ignore[reportMissingTypeStubs]
from joblib.externals.loky import (  # pyright: ignore[reportMissingTypeStubs]
    get_reusable_executor,  # pyright: ignore[reportUnknownVariableType]
)
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits  # pyright: ignore[reportMissingTypeStubs]

from rubberruler._cache import RunResult, RunStore, check_cache_dir, identify
from rubberruler._console import progress, say
from rubberruler._errors import check_int, error
from rubberruler._instances import InstanceObjective, ProblemInstance, instance_tuple
from rubberruler._problem import (
    Algorithm,
    BudgetExhausted,
    Objective,
    Problem,
    RunRecord,
)
from rubberruler._seeding import derive_seed

logger = logging.getLogger("rubberruler")

LOGGED_FAILURES = 10
MESSAGE_LENGTH = 500
PRECHECK_BUDGET_PER_DIMENSION = 10
PRECHECK_SECONDS = 60
# How often the pre-check looks at the time, so that Ctrl-C works soon on every OS.
WAIT_SECONDS = 0.1
# How often a worker checks that the process that started it is still there.
PARENT_CHECK_SECONDS = 1.0


@dataclass(frozen=True)
class Task:
    """The runs of one algorithm on one instance, sent to a worker together."""

    instance: ProblemInstance
    algorithm: str
    run_numbers: tuple[int, ...]


@dataclass(frozen=True)
class Plan:
    """The runs to do: those in the cache, and the tasks that are left."""

    instances: tuple[ProblemInstance, ...]
    names: tuple[str, ...]
    budget_per_dimension: int
    n_runs: int
    cached: dict[tuple[str, str, int], RunResult]
    tasks: tuple[Task, ...]
    store: RunStore | None
    identifiers: dict[str, str]
    identity_warnings: tuple[str, ...]
    location: str | None

    @property
    def total(self) -> int:
        return len(self.instances) * len(self.names) * self.n_runs


@dataclass(frozen=True)
class Measurement:
    """J of every instance and algorithm, with the failed runs and the warnings."""

    performance: pd.DataFrame
    failures: tuple[RunResult, ...]
    warnings: tuple[str, ...]
    identity_warnings: tuple[str, ...]


class PrecheckTimeout(BaseException):
    """Stops the pre-check run of an algorithm after PRECHECK_SECONDS."""


class StoppableObjective:
    """The objective of a pre-check run; once ``stopped``, every evaluation raises."""

    def __init__(self, objective: Objective) -> None:
        self._objective = objective
        self.stopped = False

    def one(self, x: NDArray[np.float64], /) -> float:
        if self.stopped:
            raise PrecheckTimeout
        return self._objective.one(x)

    def many(self, points: NDArray[np.float64], /) -> NDArray[np.float64]:
        if self.stopped:
            raise PrecheckTimeout
        return self._objective.many(points)


def measure_performance(
    portfolio: Mapping[str, Algorithm],
    instances: Iterable[ProblemInstance],
    *,
    budget_per_dimension: int = 1000,
    n_runs: int = 31,
    n_jobs: int = -1,
    cache_dir: str | os.PathLike[str] | None = ".rubberruler",
) -> pd.DataFrame:
    """Measure the performance J(p, a) of every algorithm on every instance.

    Each algorithm runs ``n_runs`` times on each instance with the budget
    ``budget_per_dimension`` x d. A run's value is the smallest objective value it
    saw within the budget. J(p, a) is the average probability that a run of ``a``
    beats a run of another algorithm, so it is in [0, 1] and larger is better.

    Finished runs are kept in ``cache_dir``: the same call after an interruption
    resumes, and changing an algorithm runs only that algorithm again.

    Args:
        portfolio: Algorithm names mapped to algorithms. Each algorithm takes a
            ``rr.Problem``. At least 2.
        instances: The instances, such as ``split`` from ``rr.bbob_instances()``.
        budget_per_dimension: The budget of one run divided by d.
        n_runs: The number of runs of each algorithm on each instance.
        n_jobs: The number of worker processes; -1 uses every CPU.
        cache_dir: The folder that keeps finished runs; None keeps nothing.

    Returns:
        A DataFrame of J with one row per ``instance_id`` and one column per
        algorithm name.
    """
    algorithms = check_portfolio(portfolio)
    selected = instance_tuple(instances, "measure_performance")
    budget_per_dimension = check_budget_per_dimension(budget_per_dimension)
    n_runs = check_n_runs(n_runs)
    n_jobs = check_n_jobs(n_jobs)
    folder = check_cache_dir(cache_dir)
    precheck_algorithms(algorithms, selected[0], n_jobs=n_jobs)
    plan = plan_runs(
        algorithms,
        selected,
        budget_per_dimension=budget_per_dimension,
        n_runs=n_runs,
        cache_dir=folder,
    )
    say(plan_line(plan))
    measurement = run_plan(
        plan, algorithms, n_jobs=n_jobs, description="Measuring performance"
    )
    for line in (*measurement.warnings, *measurement.identity_warnings):
        logger.warning(line)
    return measurement.performance


def plan_runs(
    portfolio: Mapping[str, Algorithm],
    instances: Sequence[ProblemInstance],
    *,
    budget_per_dimension: int,
    n_runs: int,
    cache_dir: Path | None,
) -> Plan:
    """Find the cached runs and group the others into tasks."""
    identities = (
        {name: identify(name, algorithm) for name, algorithm in portfolio.items()}
        if cache_dir is not None
        else {}
    )
    identifiers = {name: identity.identifier for name, identity in identities.items()}
    store = RunStore(cache_dir) if cache_dir is not None else None
    stored = store.load(identifiers) if store is not None else {}
    cached: dict[tuple[str, str, int], RunResult] = {}
    tasks: list[Task] = []
    for instance in instances:
        budget = budget_per_dimension * instance.dimension
        for name in portfolio:
            missing: list[int] = []
            for run_number in range(n_runs):
                key = (instance.instance_id, instance.seed, name, budget, run_number)
                if key in stored:
                    cached[instance.instance_id, name, run_number] = stored[key]
                else:
                    missing.append(run_number)
            if missing:
                tasks.append(Task(instance, name, tuple(missing)))
    warnings = [
        identity.warning for identity in identities.values() if identity.warning
    ]
    ordered = sorted(warnings, key=lambda line: "name only" not in line)
    return Plan(
        instances=tuple(instances),
        names=tuple(portfolio),
        budget_per_dimension=budget_per_dimension,
        n_runs=n_runs,
        cached=cached,
        tasks=tuple(tasks),
        store=store,
        identifiers=identifiers,
        identity_warnings=tuple(ordered),
        location=None if cache_dir is None else os.path.join(str(cache_dir), ""),
    )


def plan_line(plan: Plan) -> str:
    """The plan line, "rubberruler: N runs (C cached, R to run) in DIR/".

    When only some runs are cached, a second line names the algorithms with runs to
    do, such as one whose source changed.
    """
    cached = len(plan.cached)
    counts = f"{plan.total:,} runs ({cached:,} cached, {plan.total - cached:,} to run)"
    if plan.location is None:
        line = f"rubberruler: {counts} without a cache"
    else:
        line = f"rubberruler: {counts} in {plan.location}"
    if 0 < cached < plan.total:
        to_run = dict.fromkeys(plan.names, 0)
        for task in plan.tasks:
            to_run[task.algorithm] += len(task.run_numbers)
        names = ", ".join(
            f"{name} {count:,}" for name, count in to_run.items() if count
        )
        line += f"\nrubberruler: to run: {names}"
    return line


def run_plan(
    plan: Plan, portfolio: Mapping[str, Algorithm], *, n_jobs: int, description: str
) -> Measurement:
    """Run the tasks of the plan, keep each finished task, and compute J.

    On Ctrl-C, the runs finished so far stay saved, one line tells how to resume,
    and KeyboardInterrupt is raised again.
    """
    results = dict(plan.cached)
    saved = len(plan.cached)
    to_run = sum(len(task.run_numbers) for task in plan.tasks)
    # Closing the generator on any exit makes joblib stop its workers right away;
    # left open until the interpreter exits, idle workers would delay the exit.
    tasks = execute(plan.tasks, portfolio, plan.budget_per_dimension, n_jobs)
    try:
        with progress(description, to_run) as advance, closing_quietly(tasks):
            for finished in tasks:
                if plan.store is not None:
                    try:
                        plan.store.save(finished, plan.identifiers)
                    except KeyboardInterrupt:
                        plan.store.save(finished, plan.identifiers)
                        saved += len(finished)
                        raise
                    saved += len(finished)
                for result in finished:
                    results[result.instance_id, result.algorithm, result.run_number] = (
                        result
                    )
                advance(len(finished))
    except KeyboardInterrupt:
        if plan.store is None:
            say(
                "rubberruler: stopped. No runs are saved because cache_dir is None.",
                style="yellow",
            )
        else:
            say(
                f"rubberruler: stopped. {saved:,} of {plan.total:,} runs are saved. "
                "Run the same call again to resume.",
                style="yellow",
            )
        raise
    ordered = [
        results[instance.instance_id, name, run_number]
        for instance in plan.instances
        for name in plan.names
        for run_number in range(plan.n_runs)
    ]
    failures = tuple(result for result in ordered if result.failed)
    log_failures(failures)
    lines = (failure_warning(len(failures)),) if failures else ()
    return Measurement(
        performance_table(plan.instances, plan.names, plan.n_runs, results),
        failures,
        lines,
        plan.identity_warnings,
    )


@contextlib.contextmanager
def closing_quietly(
    tasks: Generator[list[RunResult], None, None],
) -> Generator[Generator[list[RunResult], None, None], None, None]:
    """Close the generator on exit, without joblib's warning about dropped tasks.

    After Ctrl-C, joblib warns that finished tasks were not used and suggests changing
    the input; the "stopped" line already says how many runs are saved.
    """
    try:
        yield tasks
    finally:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"\d+ tasks have been successfully executed",
                category=UserWarning,
            )
            tasks.close()


def execute(
    tasks: Sequence[Task],
    portfolio: Mapping[str, Algorithm],
    budget_per_dimension: int,
    n_jobs: int,
) -> Generator[list[RunResult], None, None]:
    """Run the tasks in worker processes and yield each task's results when done.

    Each call is the (function, args, kwargs) tuple that ``joblib.delayed`` makes.
    Closing this generator closes joblib's, which stops the remaining tasks. After
    the last task, the workers are shut down instead of idling for the next call.
    """
    calls = (
        (
            run_task,
            (
                task.instance,
                task.algorithm,
                portfolio[task.algorithm],
                budget_per_dimension * task.instance.dimension,
                task.run_numbers,
                os.getpid(),
            ),
            dict[str, object](),
        )
        for task in tasks
    )
    parallel = Parallel(n_jobs=n_jobs, return_as="generator_unordered")
    yield from cast(Iterator[list[RunResult]], parallel(calls))
    if n_jobs != 1:
        stop_workers()


def stop_workers() -> None:
    """Shut down joblib's worker processes once the runs are done.

    joblib keeps idle workers for the next call; they would hold their memory while
    the features are computed, and on Windows keep the working folder in use.
    """
    reusable = cast(Callable[..., Any], get_reusable_executor)
    reusable(reuse=True).shutdown(wait=True)


def run_task(
    instance: ProblemInstance,
    name: str,
    algorithm: Algorithm,
    budget: int,
    run_numbers: tuple[int, ...],
    parent: int | None = None,
) -> list[RunResult]:
    """Run one algorithm on one instance for each run number (in a worker).

    ``parent`` is the process that sends the tasks; a worker exits once it is gone.
    """
    if parent is not None and parent != os.getpid():
        watch_parent(parent)
    results: list[RunResult] = []
    with threadpool_limits(limits=1):
        objective = InstanceObjective(instance)
        for run_number in run_numbers:
            record, failure = run_once(
                algorithm,
                objective,
                instance,
                budget,
                run_seed(instance, name, run_number),
            )
            results.append(
                RunResult(
                    instance_id=instance.instance_id,
                    seed=instance.seed,
                    algorithm=name,
                    budget=budget,
                    run_number=run_number,
                    best=record.best,
                    evaluations=record.evaluations,
                    failed=failure is not None,
                    message=failure or "",
                )
            )
    return results


_watched_parent: list[int] = []


def watch_parent(parent: int) -> None:
    """Exit this worker as soon as the parent process is gone (one watch per worker).

    A parent killed with kill -9 cannot stop its workers, and loky's workers then
    wait forever to send their results. On POSIX a worker is adopted by a new parent
    when its parent dies; on Windows, psutil tells whether the parent still exists.
    """
    if parent in _watched_parent:
        return
    _watched_parent.append(parent)

    def parent_is_there() -> bool:
        if sys.platform == "win32":
            return psutil.pid_exists(parent)  # pyright: ignore[reportUnknownMemberType]
        return os.getppid() == parent

    def watch() -> None:
        while parent_is_there():
            time.sleep(PARENT_CHECK_SECONDS)
        os._exit(1)

    threading.Thread(target=watch, name="rubberruler-parent", daemon=True).start()


def run_once(
    algorithm: Algorithm,
    objective: Objective,
    instance: ProblemInstance,
    budget: int,
    seed: int,
) -> tuple[RunRecord, str | None]:
    """Run the algorithm once; return the record and why the run failed, if it did.

    A run fails if the algorithm raises an exception (its value is then the smallest
    value so far) or returns without evaluating the problem (its value is +inf).
    """
    record = RunRecord()
    problem = new_problem(objective, record, instance, budget, seed)
    seed_global_generators(seed)
    try:
        algorithm(problem)
    except BudgetExhausted:
        pass
    except Exception as exception:
        return record, describe(exception)
    if record.evaluations == 0:
        return record, "returned without evaluating problem"
    return record, None


def new_problem(
    objective: Objective,
    record: RunRecord,
    instance: ProblemInstance,
    budget: int,
    seed: int,
) -> Problem:
    return Problem(
        objective,
        record,
        instance_id=instance.instance_id,
        lower_bounds=instance.lower_bounds,
        upper_bounds=instance.upper_bounds,
        budget=budget,
        seed=seed,
    )


def run_seed(instance: ProblemInstance, algorithm: str, run_number: int) -> int:
    """The seed of a run. A transformed instance uses the seed of its original."""
    return derive_seed(
        "run",
        instance.seed,
        "bbob",
        instance.function_id,
        instance.instance_number,
        instance.dimension,
        algorithm,
        run_number,
    )


def seed_global_generators(seed: int) -> None:
    """Seed random, numpy.random, and torch (if imported) for the next run."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)


def superiority(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    """PS(a, a'): the mean over run pairs of [r < r'] + 0.5 [r = r']."""
    rows = first[:, np.newaxis]
    columns = second[np.newaxis, :]
    wins = np.count_nonzero(rows < columns)
    ties = np.count_nonzero(rows == columns)
    return (wins + 0.5 * ties) / (first.size * second.size)


def average_superiority(runs: NDArray[np.float64]) -> NDArray[np.float64]:
    """J(p, a) for each algorithm: the mean of PS(a, a') over the others a'.

    ``runs`` has one row per algorithm and one column per run.
    """
    count = runs.shape[0]
    return np.array(
        [
            sum(superiority(runs[a], runs[b]) for b in range(count) if b != a)
            / (count - 1)
            for a in range(count)
        ],
        dtype=np.float64,
    )


def performance_table(
    instances: Sequence[ProblemInstance],
    names: Sequence[str],
    n_runs: int,
    results: Mapping[tuple[str, str, int], RunResult],
) -> pd.DataFrame:
    """J as a DataFrame: one row per instance_id, one column per algorithm."""
    values = np.empty((len(instances), len(names)), dtype=np.float64)
    for row, instance in enumerate(instances):
        runs = np.array(
            [
                [results[instance.instance_id, name, run].best for run in range(n_runs)]
                for name in names
            ],
            dtype=np.float64,
        )
        values[row] = average_superiority(runs)
    index = pd.Index(
        [instance.instance_id for instance in instances], name="instance_id"
    )
    return pd.DataFrame(values, index=index, columns=list(names))


def precheck_algorithms(
    portfolio: Mapping[str, Algorithm], instance: ProblemInstance, *, n_jobs: int
) -> None:
    """Stop if an algorithm cannot finish a short run.

    Each algorithm runs once on ``instance`` with the budget 10 x d and seed 0. It
    must finish within 60 seconds, evaluate the problem at least once, and not
    raise. With n_jobs other than 1 it must also go through cloudpickle. The run
    takes place in another thread of this process, so the time limit holds on
    every OS, from any thread, and for algorithms that cannot be pickled.
    """
    budget = PRECHECK_BUDGET_PER_DIMENSION * instance.dimension
    objective = InstanceObjective(instance)
    where = f"on {instance.instance_id} in the pre-check run (budget {budget}, seed 0)"
    for name, algorithm in portfolio.items():
        record = RunRecord()
        stoppable = StoppableObjective(objective)
        problem = new_problem(stoppable, record, instance, budget, 0)
        seed_global_generators(0)
        try:
            raised = run_with_time_limit(
                algorithm, problem, stoppable, PRECHECK_SECONDS
            )
        except PrecheckTimeout:
            raise error(
                f'algorithm "{name}" did not finish within {PRECHECK_SECONDS} '
                f"seconds {where}.",
                why="with the full budget, every run would take far longer.",
                fix="let the algorithm stop when problem(x) raises after the budget "
                "is used up, and look for an endless loop.",
            ) from None
        if isinstance(raised, Exception):
            raise error(
                f'algorithm "{name}" raised {type(raised).__name__} {where}.',
                why="a run that raises cannot show how well the algorithm performs.",
                fix="fix the algorithm so that this run finishes; the original "
                "exception is shown above.",
            ) from raised
        if raised is not None and not isinstance(raised, BudgetExhausted):
            raise raised  # such as SystemExit, as if the run were in this thread
        if record.evaluations == 0:
            raise error(
                f'algorithm "{name}" returned without evaluating problem {where}.',
                why="a run's value is the smallest value that problem(x) returned, "
                "so every run must call problem(x).",
                fix="call problem(x) in the algorithm; wrap a function of another "
                "shape in one line, such as "
                "lambda p: my_es(p, p.dimension, p.budget, p.seed).",
            )
        if n_jobs != 1:
            try:
                cloudpickle.dumps(algorithm)  # pyright: ignore[reportUnknownMemberType]
            except Exception as exception:
                raise error(
                    f'algorithm "{name}" cannot be sent to worker processes '
                    f"({describe(exception)}).",
                    why=f"n_jobs={n_jobs} runs the algorithms in other processes, "
                    "which needs cloudpickle.",
                    fix="keep open files, locks, and connections out of the "
                    "algorithm, or pass n_jobs=1.",
                ) from exception


def run_with_time_limit(
    algorithm: Algorithm,
    problem: Problem,
    objective: StoppableObjective,
    seconds: float,
) -> BaseException | None:
    """Run ``algorithm(problem)`` in another thread; return what it raised, if any.

    This thread waits in short steps, so Ctrl-C reaches it soon on every OS. After
    ``seconds``, or on Ctrl-C, the run is told to stop and PrecheckTimeout
    (or KeyboardInterrupt) is raised here without waiting for the run to end.
    """
    raised: list[BaseException] = []
    done = threading.Event()

    def target() -> None:
        try:
            try:
                algorithm(problem)
            except BaseException as exception:
                raised.append(exception)
            finally:
                done.set()
        except PrecheckTimeout:
            pass  # sent by stop_run just as the run ended by itself

    thread = threading.Thread(target=target, name="rubberruler-pre-check", daemon=True)
    deadline = time.monotonic() + seconds
    thread.start()
    try:
        while not done.wait(WAIT_SECONDS):
            if time.monotonic() >= deadline:
                raise PrecheckTimeout
    except BaseException:
        stop_run(thread, objective, done)
        raise
    return raised[0] if raised else None


def stop_run(
    thread: threading.Thread, objective: StoppableObjective, done: threading.Event
) -> None:
    """Make a pre-check run end: its next evaluation raises, and so does its thread.

    The exception sent to the thread arrives at its next Python instruction; a run
    inside a long C call ends when the call returns. The thread is a daemon, so it
    never keeps the interpreter from exiting.
    """
    objective.stopped = True
    if thread.ident is not None and not done.is_set():
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(thread.ident), ctypes.py_object(PrecheckTimeout)
        )


def check_portfolio(portfolio: object) -> dict[str, Algorithm]:
    """Return the portfolio as a dict of at least 2 named functions, or stop."""
    example = '{"de": de, "nelder-mead": nelder_mead}'
    if not isinstance(portfolio, Mapping):
        raise error(
            f"portfolio is a {type(portfolio).__name__}.",
            why="portfolio must map algorithm names to algorithms.",
            fix=f"pass a dict such as {example}.",
        )
    items: list[tuple[object, object]] = list(portfolio.items())  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    if len(items) < 2:
        raise error(
            f"portfolio has {len(items)} algorithm{'' if len(items) == 1 else 's'}.",
            why="J compares each algorithm with the others, so it needs at least 2.",
            fix="add another algorithm to portfolio, such as a random search.",
        )
    checked: dict[str, Algorithm] = {}
    for name, algorithm in items:
        if not isinstance(name, str) or not name:
            raise error(
                f"portfolio has the name {name!r}.",
                why="algorithm names must be non-empty strings.",
                fix='use names such as "cma-es".',
            )
        if not callable(algorithm):
            raise error(
                f'portfolio["{name}"] is a {type(algorithm).__name__}, not a function.',
                why="each algorithm must be a function that takes a rr.Problem.",
                fix=f"pass the function itself, such as {example}.",
            )
        checked[name] = algorithm
    return checked


def check_budget_per_dimension(value: object) -> int:
    return check_int(
        "budget_per_dimension",
        value,
        minimum=1,
        example="an int such as budget_per_dimension=1000",
    )


def check_n_runs(value: object) -> int:
    return check_int("n_runs", value, minimum=1, example="an int such as n_runs=31")


def check_n_jobs(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value != 0:
        return value
    raise error(
        f"n_jobs={value!r} is not allowed.",
        why="n_jobs must be a non-zero int; -1 uses every CPU.",
        fix="pass n_jobs=-1 to use every CPU, or n_jobs=1 to run in this process.",
    )


def describe(exception: BaseException) -> str:
    text = f"{type(exception).__name__}: {exception}"
    return text if len(text) <= MESSAGE_LENGTH else text[: MESSAGE_LENGTH - 3] + "..."


def failure_warning(count: int) -> str:
    """The warning line about failed runs (singular for one run)."""
    if count == 1:
        return (
            "1 run failed; its best value so far was used (+inf if none) (see the log)"
        )
    return (
        f"{count} runs failed; their best value so far was used (+inf if none) "
        "(see the log)"
    )


def log_failures(failures: Sequence[RunResult]) -> None:
    """Log the first LOGGED_FAILURES failed runs: algorithm, instance, run, error."""
    for failure in failures[:LOGGED_FAILURES]:
        logger.warning(
            'run %d of "%s" on %s failed: %s',
            failure.run_number,
            failure.algorithm,
            failure.instance_id,
            failure.message,
        )
