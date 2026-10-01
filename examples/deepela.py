"""DeepELA's features as the extractor, with the quickstart's portfolio and predictor.

DeepELA is not on PyPI. Fetch its repository at a fixed commit, install what it
imports, and put the folder on PYTHONPATH. It works on Python 3.11 and 3.12, so it can
go next to pflacco too:

    commit=c0368677461f96a17536111ee947bf111e8b8425
    git init deep_ela
    git -C deep_ela fetch --depth 1 https://github.com/mvseiler/deep_ela $commit
    git -C deep_ela checkout FETCH_HEAD
    pip install rubberruler scikit-learn torch lightning einops pyyaml
    export PYTHONPATH="$PWD/deep_ela"

It runs with the default settings, the settings of the paper. The portfolio is the
quickstart's, so after the quickstart with the same settings every run comes from the
cache, and the time goes to the features: DeepELA runs one instance after another in
this process, 5,760 times for 2,880 instances and both feature types, which took 31
minutes at d = 10 on 16 cores, where torch used 12 threads. With an NVIDIA GPU, pass
device="cuda" to load_large_50d_v1: on an RTX 4070 Ti the features took 72 seconds,
with values up to about 1e-5 away from the CPU's. To try it first, give run_test at
the end small settings: dimension=2, train_instance_numbers=range(11, 31),
budget_per_dimension=100, n_runs=5 (about two minutes).
"""

import deep_ela  # github.com/mvseiler/deep_ela at c036867, on PYTHONPATH
import numpy as np
import torch
from scipy.optimize import Bounds, differential_evolution, minimize
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

model = deep_ela.load_large_50d_v1(stride=1)


def deep_ela_features(X, y):
    if X.shape[1] > 11:
        raise ValueError(
            f"DeepELA's large models take at most 11 dimensions, got {X.shape[1]}"
        )
    torch.manual_seed(0)  # predict() permutes the input slots with torch's global RNG
    return model(X, y)


predictor = RandomForestRegressor(random_state=0)
result = rr.run_test(portfolio, deep_ela_features, predictor)
print(result)
