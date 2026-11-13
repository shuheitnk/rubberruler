"""Algorithms for the tests.

They live in a module on sys.path so that worker processes can import them.
"""

import time
from typing import Any

import numpy as np

from rubberruler import Problem

calls = {"counted_search": 0}


def random_search(problem: Problem) -> None:
    """Sample uniformly until the budget ends; the points depend only on the seed."""
    rng = np.random.default_rng(problem.seed)
    while True:
        problem(rng.uniform(problem.lower_bounds, problem.upper_bounds))


def batch_random_search(problem: Problem) -> None:
    """Sample 7 points at a time until the budget ends."""
    rng = np.random.default_rng(problem.seed)
    while True:
        problem(
            rng.uniform(
                problem.lower_bounds, problem.upper_bounds, size=(7, problem.dimension)
            )
        )


def global_random_search(problem: Problem) -> None:
    """Sample with numpy's global generator, which rubberruler seeds before a run."""
    while True:
        problem(np.random.uniform(problem.lower_bounds, problem.upper_bounds))  # noqa: NPY002


def center(problem: Problem) -> float:
    """Evaluate the center of the search space once and return its value."""
    return problem((problem.lower_bounds + problem.upper_bounds) / 2)


def ioh_style_search(problem: Any) -> None:
    """Written for IOHexperimenter: it uses bounds.lb, meta_data, and state only."""
    rng = np.random.default_rng(1)
    while problem.state.evaluations < 5 * problem.meta_data.n_variables:
        problem(list(rng.uniform(problem.bounds.lb, problem.bounds.ub)))


def returns_without_evaluating(problem: Problem) -> None:
    """Never calls problem."""


def raises_after_three(problem: Problem) -> None:
    for _ in range(3):
        problem(np.ones(problem.dimension))
    raise ValueError("boom")


def fails_on_odd_seeds(problem: Problem) -> None:
    """Passes the pre-check (seed 0) and fails in the runs whose seed is odd."""
    problem(np.zeros(problem.dimension))
    if problem.seed % 2 == 1:
        raise RuntimeError("odd seed")


def swallows_exceptions(problem: Problem) -> None:
    """Catches every Exception; the budget exception still ends the run."""
    rng = np.random.default_rng(problem.seed)
    while True:
        try:
            problem(rng.uniform(problem.lower_bounds, problem.upper_bounds))
        except Exception:
            pass


def sleeps(problem: Problem) -> None:
    problem(np.zeros(problem.dimension))
    while True:
        time.sleep(0.05)


def evaluates_slowly(problem: Problem) -> None:
    """Evaluates one point every 0.05 seconds until the budget ends."""
    while True:
        problem(np.zeros(problem.dimension))
        time.sleep(0.05)


def counted_search(problem: Problem) -> None:
    """random_search that counts its runs in this process."""
    calls["counted_search"] += 1
    random_search(problem)
