"""Quickstart: the Rubber Ruler Test on a small portfolio, with small settings.

It needs scikit-learn too: ``pip install rubberruler scikit-learn``. Run it with
``python quickstart.py``. While it runs, stderr shows the plan, the four stages, and
where the figure went:

    rubberruler: 43,200 runs (0 cached, 43,200 to run) in .rubberruler/
    [1/4] Preparing instances
    [2/4] Measuring performance  40%|████      | 17,355/43,200 [00:05<00:05]
    [3/4] Computing features
    [4/4] Fitting predictors and scoring
    rubberruler: figure saved to .rubberruler/rubber_ruler_test.png and .svg

Then ``print(result)`` shows the card. The README says how to read it.
"""

import numpy as np
from scipy.optimize import Bounds, differential_evolution, minimize
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestRegressor

import rubberruler as rr


# 1. A portfolio. Each algorithm minimizes `problem`;
#    rubberruler ends the run when the budget is spent.
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


# 2. A feature extractor. It gets a sample X and its objective values y,
#    rescaled (value-based) or ranked (rank-based), and returns named features.
#    These four are ELA-style.
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


# 3. A performance predictor: any regressor with fit and predict.
predictor = RandomForestRegressor(random_state=0)

result = rr.run_test(
    portfolio,
    simple_features,
    predictor,
    # Small settings for a first try.
    # Without them, run_test takes about an hour on a 16-core PC.
    dimension=2,
    budget_per_dimension=100,
    n_runs=5,
)
print(result)
