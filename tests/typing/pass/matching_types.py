"""Inputs of the right types, which pyright accepts.

pyright checks this file. It is never run.
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

import rubberruler as rr


def random_search(problem: rr.Problem) -> None:
    rng = np.random.default_rng(problem.seed)
    while True:
        problem(rng.uniform(problem.lower_bounds, problem.upper_bounds))


def returns_a_value(problem: rr.Problem) -> float:
    return problem([0.0] * problem.dimension)


def extractor(X: NDArray[np.float64], y: NDArray[np.float64]) -> dict[str, float]:
    return {"mean": float(np.mean(y)), "size": X.shape[0]}


def uses_every_public_name() -> None:
    result = rr.run_test(
        portfolio={
            "random": random_search,
            "value": returns_a_value,
            "lambda": lambda p: p(np.zeros(p.dimension)),
        },
        feature_extractor=extractor,
        predictor=make_pipeline(SimpleImputer(), RandomForestRegressor(random_state=0)),
    )
    portfolio = {"random": random_search, "value": returns_a_value}
    again = rr.run_test(
        portfolio,
        extractor,
        RandomForestRegressor(random_state=0),
        n_jobs=1,
        cache_dir=None,
        **result.config,
    )
    selected: Literal["value", "rank"] = again.selected_feature_type
    split = rr.bbob_instances(dimension=2, train_instance_numbers=[3, 4], seed=1)
    transformed = rr.transform(split.test[0])
    performance = rr.measure_performance(portfolio, split, n_jobs=1)
    features = rr.compute_features(extractor, split, selected)
    fitted = rr.fit_predictor(
        RandomForestRegressor(random_state=0),
        features,
        performance,
        instances=split.train,
    )
    r2: float = rr.r2_score(fitted, features, performance, instances=[transformed])
    try:
        print(r2)
    except rr.RubberRulerError as error:
        print(error)
