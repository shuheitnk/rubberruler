"""Decide with BBOB whether to compute features value-based or rank-based.

Stretch the ruler. Keep the features that still measure.
"""

from rubberruler._errors import RubberRulerError
from rubberruler._features import compute_features
from rubberruler._instances import bbob_instances, transform
from rubberruler._objective import transform_objective
from rubberruler._performance import measure_performance
from rubberruler._prediction import fit_predictor, r2_score
from rubberruler._problem import Problem
from rubberruler._run_test import run_test

__all__ = [
    "Problem",
    "RubberRulerError",
    "bbob_instances",
    "compute_features",
    "fit_predictor",
    "measure_performance",
    "r2_score",
    "run_test",
    "transform",
    "transform_objective",
]
