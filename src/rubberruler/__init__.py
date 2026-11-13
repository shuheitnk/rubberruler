"""Decide with BBOB, MA-BBOB, or your own objective functions whether to compute
features value-based or rank-based.

Stretch the ruler. Keep the features that still measure.
"""

from rubberruler._errors import RubberRulerError
from rubberruler._features import compute_features
from rubberruler._instances import bbob_instances, ma_bbob_instances, transform
from rubberruler._objective import objective_instances, transform_objective
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
    "ma_bbob_instances",
    "measure_performance",
    "objective_instances",
    "r2_score",
    "run_test",
    "transform",
    "transform_objective",
]
