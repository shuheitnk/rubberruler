"""Inputs of the wrong types; pyright reports each line marked expect-error.

tests/test_typing.py checks this folder with its own pyrightconfig.json. It is never
run.
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor

import rubberruler as rr


def random_search(problem: rr.Problem) -> None:
    problem(np.zeros(problem.dimension))


def needs_the_budget(problem: rr.Problem, budget: int) -> None:
    problem(np.zeros(budget))


def returns_an_array(
    X: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    return y[:3]


def extractor(X: NDArray[np.float64], y: NDArray[np.float64]) -> dict[str, float]:
    return {"mean": float(np.mean(y))}


def mismatches() -> None:
    predictor = RandomForestRegressor(random_state=0)
    rr.run_test(
        {"random": random_search, "budget": needs_the_budget},  # expect-error
        extractor,
        predictor,
    )
    rr.run_test(
        {"random": random_search},
        returns_an_array,  # expect-error
        predictor,
    )
    rr.run_test(
        {"random": random_search},
        extractor,
        predictor,
        dimension="10",  # expect-error
    )
    rr.run_test(
        {"random": random_search},
        extractor,
        predictor,
        verbose=True,  # expect-error
    )
    split = rr.bbob_instances()
    rr.compute_features(extractor, split, "values")  # expect-error
    rr.transform(split)  # expect-error
