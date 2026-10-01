"""Problem: the objective function as one run of an algorithm sees it."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rubberruler._errors import error


class BudgetExhausted(BaseException):
    """Ends a run when the algorithm calls the problem after the budget is used up.

    It derives from BaseException so that ``except Exception`` in an algorithm lets
    it through. rubberruler catches it and ends the run normally.
    """


class Objective(Protocol):
    """Evaluates the objective of one instance: f, or g(f) if it is transformed."""

    def one(self, x: NDArray[np.float64], /) -> float: ...

    def many(self, points: NDArray[np.float64], /) -> NDArray[np.float64]: ...


@dataclass
class RunRecord:
    """What one run has seen: the number of evaluations and the smallest value."""

    evaluations: int = 0
    best: float = float("inf")


class Bounds:
    """``problem.bounds``: the search range under the IOHexperimenter names."""

    def __init__(self, lb: NDArray[np.float64], ub: NDArray[np.float64]) -> None:
        self.lb = lb
        self.ub = ub


class MetaData:
    """``problem.meta_data`` with the IOHexperimenter name ``n_variables``."""

    def __init__(self, n_variables: int) -> None:
        self.n_variables = n_variables


class State:
    """``problem.state`` with the IOHexperimenter names ``evaluations`` and
    ``optimum_found``."""

    def __init__(self, evaluations: int) -> None:
        self.evaluations = evaluations

    @property
    def optimum_found(self) -> bool:
        """Always False: rubberruler does not show the optimum, or reaching it.

        ModDE reads it in every generation.
        """
        return False


class Problem:
    """The objective function as one run of an algorithm sees it.

    ``problem(x)`` evaluates one point of shape (d,) and returns a float, or n points
    of shape (n, d) and returns an array of shape (n,). On a transformed instance the
    value is g(f(x)). After ``budget`` evaluations, calling ``problem`` again ends the
    run with an exception that ``except Exception`` does not catch; the run's value
    is the smallest value seen within the budget. Points outside the search range
    can be evaluated too, and they count toward the budget.

    rubberruler makes one for each run and passes it to the algorithm; you do not
    make one yourself.

    Attributes:
        dimension: d.
        lower_bounds: The lower end of the search range, an array of shape (d,).
        upper_bounds: The upper end of the search range, an array of shape (d,).
        budget: The maximum number of evaluations in this run.
        seed: The seed for this run, from 0 to 2**32 - 1.
        evaluations: The number of evaluations so far.
        bounds: ``bounds.lb`` and ``bounds.ub``, as in IOHexperimenter.
        meta_data: ``meta_data.n_variables``, as in IOHexperimenter.
        state: ``state.evaluations`` and ``state.optimum_found`` (always False), as
            in IOHexperimenter.
    """

    def __init__(
        self,
        objective: Objective,
        record: RunRecord,
        *,
        instance_id: str,
        lower_bounds: ArrayLike,
        upper_bounds: ArrayLike,
        budget: int,
        seed: int,
    ) -> None:
        self._objective = objective
        self._record = record
        self._instance_id = instance_id
        self._lower_bounds = np.array(lower_bounds, dtype=np.float64)
        self._upper_bounds = np.array(upper_bounds, dtype=np.float64)
        self._budget = budget
        self._seed = seed
        self._bounds = Bounds(self._lower_bounds, self._upper_bounds)
        self._meta_data = MetaData(self.dimension)

    @property
    def dimension(self) -> int:
        return self._lower_bounds.shape[0]

    @property
    def lower_bounds(self) -> NDArray[np.float64]:
        return self._lower_bounds

    @property
    def upper_bounds(self) -> NDArray[np.float64]:
        return self._upper_bounds

    @property
    def budget(self) -> int:
        return self._budget

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def evaluations(self) -> int:
        return self._record.evaluations

    @property
    def bounds(self) -> Bounds:
        return self._bounds

    @property
    def meta_data(self) -> MetaData:
        return self._meta_data

    @property
    def state(self) -> State:
        return State(self._record.evaluations)

    @overload
    def __call__(self, x: Sequence[float], /) -> float: ...

    @overload
    def __call__(self, x: Sequence[Sequence[float]], /) -> NDArray[np.float64]: ...

    @overload
    def __call__(self, x: NDArray[Any], /) -> Any: ...

    def __call__(self, x: ArrayLike, /) -> float | NDArray[np.float64]:
        points = np.asarray(x, dtype=np.float64)
        if points.ndim == 1 and points.shape[0] == self.dimension:
            return self._evaluate_one(points)
        if points.ndim == 2 and points.shape[1] == self.dimension:
            return self._evaluate_many(points)
        raise error(
            f"problem(x) got x of shape {points.shape} on {self._instance_id}, "
            f"whose dimension is {self.dimension}.",
            why="x must be one point of shape (d,) or n points of shape (n, d).",
            fix=f"pass x with shape ({self.dimension},) or (n, {self.dimension}).",
        )

    def __repr__(self) -> str:
        return (
            f"Problem(dimension={self.dimension}, budget={self._budget}, "
            f"seed={self._seed}, evaluations={self._record.evaluations})"
        )

    def _evaluate_one(self, x: NDArray[np.float64]) -> float:
        if self._record.evaluations >= self._budget:
            raise BudgetExhausted
        value = self._objective.one(x)
        self._record.evaluations += 1
        if value < self._record.best:
            self._record.best = value
        return value

    def _evaluate_many(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        if points.shape[0] == 0:
            return np.empty(0, dtype=np.float64)
        remaining = self._budget - self._record.evaluations
        if remaining <= 0:
            raise BudgetExhausted
        count = min(points.shape[0], remaining)
        values = self._objective.many(points[:count])
        self._record.evaluations += count
        smallest = float(np.fmin.reduce(values))
        if smallest < self._record.best:
            self._record.best = smallest
        if count < points.shape[0]:
            raise BudgetExhausted
        return values


Algorithm = Callable[[Problem], object]
