"""Runs the test with GA, PSO, and Nelder-Mead from pymoo as the portfolio.

Install pymoo next to rubberruler. It works on Python 3.11 and 3.12, and next to
pflacco too:

    pip install rubberruler scikit-learn pymoo==0.6.2

The feature extractor and the predictor are those of example1.py. It runs with the
default settings, which follow the paper except that the sample points are uniform
random rather than a Latin hypercube and the test uses one seed rather than 11. They
take long: a pymoo run of 10,000 evaluations at d = 10 takes 0.3 to 0.9 s, more than ten
times as long as ModularCMAES, so these three algorithms take two to four hours on 16
cores. To try it first, give run_test at the end small settings: dimension=2,
budget_per_dimension=100, n_runs=5 (about a minute).
"""

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import Problem as PymooProblem
from pymoo.optimize import minimize as pymoo_minimize
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestRegressor

import rubberruler as rr


class PymooView(PymooProblem):
    """The problem as pymoo sees it. pymoo passes the solutions together, (n, d)."""

    def __init__(self, problem):
        super().__init__(
            n_var=problem.dimension,
            n_obj=1,
            xl=problem.lower_bounds,
            xu=problem.upper_bounds,
        )
        self.problem = problem

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = self.problem(X)


def pymoo_algorithm(make_algorithm):
    """A pymoo algorithm, made new by make_algorithm() for each run."""

    def algorithm(problem):
        pymoo_minimize(
            PymooView(problem),
            make_algorithm(),
            ("n_eval", problem.budget),
            seed=problem.seed,
            verbose=False,
        )

    return algorithm


portfolio = {
    "ga": pymoo_algorithm(GA),
    "pso": pymoo_algorithm(PSO),
    "nelder-mead": pymoo_algorithm(NelderMead),
}


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
result = rr.run_test(portfolio, simple_features, predictor)
print(result)
