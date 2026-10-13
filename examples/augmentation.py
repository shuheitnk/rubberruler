"""Augmentation: ten rubber rulers on your own objective, and what they do to features.

Run it with ``python augmentation.py``. It takes a few seconds.
"""

import numpy as np
from scipy.stats import kurtosis, rankdata

import rubberruler as rr


def rastrigin(x):
    return 10 * x.shape[-1] + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=-1)


d = 5
lower_bounds = np.full(d, -5.12)
upper_bounds = np.full(d, 5.12)
copies = [
    rr.transform_objective(rastrigin, lower_bounds, upper_bounds, seed=k)
    for k in range(10)
]


# Two features of the quickstart, with y value-based or rank-based as in rubberruler.
def features(X, y, feature_type):
    if feature_type == "value":
        y = (y - y.min()) / (y.max() - y.min())
    else:
        y = rankdata(y) / len(y)
    design = np.hstack([np.ones((len(y), 1)), X, X**2])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    quadratic_r2 = 1 - np.var(y - design @ coefficients) / np.var(y)
    return {"kurtosis": kurtosis(y), "quadratic_r2": quadratic_r2}


X = np.random.default_rng(0).uniform(lower_bounds, upper_bounds, size=(50 * d, d))
for feature_type in ("value", "rank"):
    original = features(X, rastrigin(X), feature_type)
    transformed = [features(X, copy(X), feature_type) for copy in copies]
    for name, value in original.items():
        values = [copy_features[name] for copy_features in transformed]
        print(
            f"{feature_type:5} {name:12}  original {value:6.3f}"
            f"  copies {min(values):6.3f} to {max(values):6.3f}"
        )
