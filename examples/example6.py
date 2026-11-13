"""Runs the test with the 48 learned features of DeepELA as the extractor.

DeepELA is not on PyPI. Fetch its repository at a fixed commit, install what it
imports, and put the folder on PYTHONPATH. It works on Python 3.11 and 3.12, so it can
go next to pflacco too:

    commit=c0368677461f96a17536111ee947bf111e8b8425
    git init deep_ela
    git -C deep_ela fetch --depth 1 https://github.com/mvseiler/deep_ela $commit
    git -C deep_ela checkout FETCH_HEAD
    pip install rubberruler scikit-learn torch lightning einops pyyaml
    export PYTHONPATH="$PWD/deep_ela"

It runs with the default settings, which follow the paper except that the sample points
are uniform random rather than a Latin hypercube and the test uses one seed rather than
11. The portfolio and the predictor are those of example1.py, so after example1.py with
the same settings every run comes from the cache, and the time goes to the features:
DeepELA runs 5,760 times for 2,880 instances and both feature types, in n_jobs worker
processes, each with its own copy of torch and the model. With n_jobs=4, which fits in
16 GB, this took 41 minutes at d = 10 on 24 CPUs and at most 2.9 GB in all; with every
CPU (n_jobs=-1), 23 minutes and 13 GB. Raise n_jobs if you have the memory. With an
NVIDIA GPU, pass device="cuda" to load_large_50d_v1: on an RTX 4070 Ti the features took
72 seconds, with values up to about 1e-5 away from the CPU's. To try it first, give
run_test at the end small settings: dimension=2, train_instance_numbers=range(11, 31),
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
            "DeepELA's large models take at most 11 dimensions, "
            f"but received {X.shape[1]}"
        )
    torch.manual_seed(0)  # predict() permutes the input slots with torch's global RNG
    return model(X, y)


predictor = RandomForestRegressor(random_state=0)
# Each worker loads torch and its own copy of the model (about 0.5 GB); 4 fit in 16 GB.
result = rr.run_test(portfolio, deep_ela_features, predictor, n_jobs=4)
print(result)
