"""Runs the test on 40 objective functions of your own instead of BBOB.

Run it with ``python example2.py``. It takes about 8 seconds on 24 CPUs.

The portfolio, the extractor, and the predictor are those of example1.py. The
problems are 40 random functions of one family, as a generator of problems in your
research would make them: rr.objective_instances splits them into 30 training and 10
test instances and transforms the test instances with random rubber rulers.
"""

import numpy as np
from scipy.optimize import Bounds, differential_evolution, minimize
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestRegressor

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


predictor = RandomForestRegressor(random_state=0)

d = 5
lower_bounds = np.full(d, -5.0)
upper_bounds = np.full(d, 5.0)


def problem(k):
    """The k-th problem of your family: a rotated ellipsoid with ripples."""
    rng = np.random.default_rng(k)
    rotation = np.linalg.qr(rng.normal(size=(d, d)))[0]
    scales = 10 ** rng.uniform(0, 3, d)
    ripples = rng.uniform(0, 10)

    def objective(x):
        z = rotation @ x
        return float(scales @ z**2 + ripples * np.sum(1 - np.cos(2 * np.pi * z)))

    return objective


objectives = {
    f"problem{k:02d}": (problem(k), lower_bounds, upper_bounds) for k in range(40)
}
split = rr.objective_instances(objectives)
result = rr.run_test(
    portfolio,
    simple_features,
    predictor,
    instances=split,
    budget_per_dimension=100,
    n_runs=5,
)
print(result)
