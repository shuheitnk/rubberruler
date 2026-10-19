"""The strictly increasing transformation g of the objective."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline, PchipInterpolator

from rubberruler._seeding import generator, instance_keys

GRID_POINTS = 2001


def log_density(
    seed: int, function_id: int | None, instance_number: int, dimension: int
) -> CubicSpline:
    """Draw the spline s whose exp is the density of psi of one instance.

    ``function_id`` is None on MA-BBOB.
    """
    keys = instance_keys(function_id, instance_number, dimension)
    return draw_log_density(generator("transform", seed, *keys))


def draw_log_density(rng: np.random.Generator) -> CubicSpline:
    """Draw the spline s whose exp is the density of psi from ``rng``."""
    m = int(rng.integers(5, 9))
    knots = np.linspace(0, 1, m)
    heights = rng.normal(0.0, 1.0, m)
    return CubicSpline(knots, heights)


def psi_from(s: CubicSpline) -> PchipInterpolator:
    """Integrate exp(s) into psi, strictly increasing from 0 to 1."""
    z = np.linspace(0, 1, GRID_POINTS)
    g = cumulative_trapezoid(np.exp(s(z)), z, initial=0)
    g = g / g[-1]
    g[0] = 0.0
    g[-1] = 1.0
    return PchipInterpolator(z, g)


def sample_psi(
    seed: int, function_id: int | None, instance_number: int, dimension: int
) -> PchipInterpolator:
    """Draw psi for one instance. The same arguments always give the same psi."""
    return psi_from(log_density(seed, function_id, instance_number, dimension))


def objective_psi(seed: int) -> PchipInterpolator:
    """Draw psi for rr.transform_objective. It depends only on ``seed``."""
    return psi_from(draw_log_density(generator("transform", seed, "objective")))


@dataclass(frozen=True)
class StrictlyIncreasingTransformation:
    """g(y): rescale y by f_min and f_max, apply psi up to 1, and the identity above."""

    psi: PchipInterpolator
    f_min: float
    f_max: float

    def __call__(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        z = np.maximum((y - self.f_min) / (self.f_max - self.f_min), 0.0)
        inside = np.asarray(self.psi(np.minimum(z, 1.0)), dtype=np.float64)
        return np.where(z <= 1.0, inside, z)


@dataclass(frozen=True)
class ObjectiveTransformation:
    """g(y) of rr.transform_objective: psi(z) for 0 < z < 1, and z elsewhere.

    z = (y - f_min) / (f_max - f_min). NaN stays NaN, and +-inf stays +-inf.
    """

    psi: PchipInterpolator
    f_min: float
    f_max: float

    def __call__(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        z = (y - self.f_min) / (self.f_max - self.f_min)
        inside = (z > 0.0) & (z < 1.0)
        stretched = np.asarray(self.psi(np.where(inside, z, 0.0)), dtype=np.float64)
        return np.where(inside, stretched, z)
