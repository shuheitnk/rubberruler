"""The strictly increasing transformation g of the objective."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline, PchipInterpolator

from rubberruler._seeding import generator

GRID_POINTS = 2001


def log_density(
    seed: int, function_id: int, instance_number: int, dimension: int
) -> CubicSpline:
    """Draw the spline s whose exp is the density of psi."""
    rng = generator("transform", seed, "bbob", function_id, instance_number, dimension)
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
    seed: int, function_id: int, instance_number: int, dimension: int
) -> PchipInterpolator:
    """Draw psi for one instance. The same arguments always give the same psi."""
    return psi_from(log_density(seed, function_id, instance_number, dimension))


@dataclass(frozen=True)
class MonotoneTransformation:
    """g(y): rescale y by f_min and f_max, apply psi up to 1, and the identity above."""

    psi: PchipInterpolator
    f_min: float
    f_max: float

    def __call__(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        z = np.maximum((y - self.f_min) / (self.f_max - self.f_min), 0.0)
        inside = np.asarray(self.psi(np.minimum(z, 1.0)), dtype=np.float64)
        return np.where(z <= 1.0, inside, z)
