"""Runs the test with a feature-free model that learns J from the sample itself.

Feature-free methods of algorithm selection, such as point-cloud transformers, give
the sampled points and their objective values to the model, without computing
features. With rubberruler, the extractor returns the sample itself, and the model
reads it back as points. Value-based, the model then receives the objective values
rescaled to [0, 1]; rank-based, their ranks, rescaled to [0, 1] in the same way.

    pip install rubberruler torch

Run it with ``python example7.py``. With the small settings of example1.py, it takes
about two minutes, most of it for training the networks on the CPU.
"""

import numpy as np
import torch
from scipy.optimize import Bounds, differential_evolution, minimize

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


def sample(X, y):
    """The sample itself: the coordinates and the value of every point, in order."""
    points = np.column_stack([X, y])
    return {f"p{k}": value for k, value in enumerate(points.ravel())}


class DeepSets:
    """A small network that encodes each point and averages over the points, so its
    prediction does not depend on the order of the points."""

    def __init__(self, point_size):
        self.point_size = point_size

    def points(self, X):
        rows = torch.as_tensor(X, dtype=torch.float32)
        return rows.reshape(len(X), -1, self.point_size)

    def forward(self, points):
        return self.head(self.encoder(points).mean(dim=1)).squeeze(1)

    def fit(self, X, y):
        torch.manual_seed(0)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.point_size, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)
        )
        parameters = [*self.encoder.parameters(), *self.head.parameters()]
        optimizer = torch.optim.Adam(parameters, lr=0.01)
        points = self.points(X)
        targets = torch.as_tensor(y, dtype=torch.float32)
        for _ in range(200):
            optimizer.zero_grad()
            torch.nn.functional.mse_loss(self.forward(points), targets).backward()
            optimizer.step()
        return self

    def predict(self, X):
        with torch.no_grad():
            return self.forward(self.points(X)).numpy()


d = 2
result = rr.run_test(
    portfolio,
    sample,
    DeepSets(point_size=d + 1),
    dimension=d,
    budget_per_dimension=100,
    n_runs=5,
)
print(result)
