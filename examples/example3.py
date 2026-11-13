"""Makes training data from transformed objectives with ``rr.transform_objective``.

This is augmentation: rubber rulers on your own objectives, as additional training
data. Run it with ``python example3.py``. It takes about half a minute.

The portfolio and the extractor are those of example1.py. Each objective gets ten
transformed objectives from rr.transform_objective; rubberruler then runs the
portfolio and computes the features on all of them, as it does for the test.
"""

import numpy as np
from scipy.optimize import Bounds, differential_evolution, minimize
from scipy.stats import kurtosis, skew

import rubberruler as rr


def random_search(problem):
    rng = np.random.default_rng(problem.seed)
    while True:
        problem(rng.uniform(problem.lower_bounds, problem.upper_bounds))


def de(problem):
    bounds = Bounds(problem.lower_bounds, problem.upper_bounds)
    rng = np.random.default_rng(problem.seed)
    differential_evolution(problem, bounds, maxiter=10**6, polish=False, seed=rng)


def nelder_mead(problem):
    rng = np.random.default_rng(problem.seed)
    x0 = rng.uniform(problem.lower_bounds, problem.upper_bounds)
    minimize(problem, x0, method="Nelder-Mead", options={"maxfev": problem.budget})


portfolio = {"random-search": random_search, "de": de, "nelder-mead": nelder_mead}


def simple_features(X, y):
    def r2_of_fit(design):
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        return float(1 - np.var(y - design @ coefficients) / np.var(y))

    ones = np.ones((len(y), 1))
    return {
        "skewness": float(skew(y)),
        "kurtosis": float(kurtosis(y)),
        "linear_r2": r2_of_fit(np.hstack([ones, X])),
        "quadratic_r2": r2_of_fit(np.hstack([ones, X, X**2])),
    }


def rastrigin(x):
    return 10 * x.shape[-1] + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=-1)


def rosenbrock(x):
    return np.sum(
        100 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (1 - x[..., :-1]) ** 2,
        axis=-1,
    )


d = 5
bounds = (np.full(d, -5.0), np.full(d, 5.0))
objectives = {"rastrigin": (rastrigin, *bounds), "rosenbrock": (rosenbrock, *bounds)}

# Ten transformed objectives of each objective, each with its own seed.
augmented = {
    f"{name}-{k}": (
        rr.transform_objective(objective, lower, upper, seed=10 * i + k),
        lower,
        upper,
    )
    for i, (name, (objective, lower, upper)) in enumerate(objectives.items())
    for k in range(10)
}

# rubberruler runs the portfolio and computes the features on every objective.
split = rr.objective_instances({**objectives, **augmented})
instances = [*split.train, *split.test]
performance = rr.measure_performance(
    portfolio, instances, budget_per_dimension=100, n_runs=5
)
features = {
    feature_type: rr.compute_features(simple_features, instances, feature_type)
    for feature_type in ("value", "rank")
}

print(f"{len(instances)} instances, J of {performance.shape[1]} algorithms")
for name in objectives:
    rows = [row for row in performance.index if row.split("-")[0] == name]
    for feature_type, table in features.items():
        values = table.loc[rows, "kurtosis"]
        print(
            f"{name:10} {feature_type:5} kurtosis "
            f"from {values.min():6.3f} to {values.max():6.3f}"
        )
