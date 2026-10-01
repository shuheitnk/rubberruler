import numpy as np
import pytest
from helpers import assert_three_lines

from rubberruler import RubberRulerError
from rubberruler._instances import InstanceObjective, make_split, transformation_of
from rubberruler._problem import BudgetExhausted, Problem, RunRecord


class SumOfSquares:
    def one(self, x: np.ndarray, /) -> float:
        return float(np.sum(x**2))

    def many(self, points: np.ndarray, /) -> np.ndarray:
        return np.sum(points**2, axis=1)


def make_problem(budget: int = 5, dimension: int = 2) -> tuple[Problem, RunRecord]:
    record = RunRecord()
    problem = Problem(
        SumOfSquares(),
        record,
        instance_id="f01_i001_d2",
        lower_bounds=np.full(dimension, -5.0),
        upper_bounds=np.full(dimension, 5.0),
        budget=budget,
        seed=7,
    )
    return problem, record


def test_one_point_gives_a_float_and_n_points_give_an_array() -> None:
    """problem(x) is a float for (d,) and an (n,) array for (n, d)."""
    problem, _ = make_problem()
    value = problem(np.array([1.0, 2.0]))
    assert type(value) is float
    assert value == 5.0
    values = problem(np.array([[1.0, 0.0], [0.0, 3.0]]))
    assert isinstance(values, np.ndarray)
    assert values.dtype == np.float64
    np.testing.assert_array_equal(values, [1.0, 9.0])
    assert problem([1, 1]) == 2.0


def test_problem_has_its_attributes_and_the_ioh_names() -> None:
    """dimension, bounds, budget, seed, evaluations, and the IOH names."""
    problem, _ = make_problem(budget=5, dimension=3)
    assert problem.dimension == 3
    np.testing.assert_array_equal(problem.lower_bounds, [-5.0, -5.0, -5.0])
    np.testing.assert_array_equal(problem.upper_bounds, [5.0, 5.0, 5.0])
    assert problem.budget == 5
    assert problem.seed == 7
    assert problem.evaluations == 0
    problem(np.zeros(3))
    assert problem.evaluations == 1
    assert problem.bounds.lb is problem.lower_bounds
    assert problem.bounds.ub is problem.upper_bounds
    assert problem.meta_data.n_variables == 3
    assert problem.state.evaluations == 1


def test_state_optimum_found_is_always_false_and_read_only() -> None:
    """state.optimum_found (ModDE reads it) is False, even at the optimum."""
    problem, _ = make_problem()
    assert problem.state.optimum_found is False
    problem(np.zeros(2))
    assert problem.state.optimum_found is False
    with pytest.raises(AttributeError):
        problem.state.optimum_found = True  # pyright: ignore[reportAttributeAccessIssue]


def test_attributes_cannot_be_changed() -> None:
    """An algorithm cannot raise its own budget."""
    problem, _ = make_problem()
    with pytest.raises(AttributeError):
        problem.budget = 10  # pyright: ignore[reportAttributeAccessIssue]


def test_calls_after_the_budget_raise_past_except_exception() -> None:
    """After the budget, problem raises an exception that is not Exception."""
    problem, record = make_problem(budget=2)
    problem(np.array([1.0, 1.0]))
    problem(np.array([2.0, 2.0]))
    with pytest.raises(BudgetExhausted):
        try:
            problem(np.array([0.0, 0.0]))
        except Exception:  # an algorithm that swallows its errors
            pytest.fail("the budget exception was caught by except Exception")
    assert not issubclass(BudgetExhausted, Exception)
    with pytest.raises(BudgetExhausted):
        problem(np.array([0.0, 0.0]))
    assert record.evaluations == 2
    assert record.best == 2.0


def test_a_batch_that_crosses_the_budget_records_the_first_points() -> None:
    """A batch past the budget records the points within it, then stops."""
    problem, record = make_problem(budget=3)
    problem(np.array([3.0, 0.0]))
    with pytest.raises(BudgetExhausted):
        problem(np.array([[2.0, 0.0], [1.0, 0.0], [0.0, 0.0]]))
    assert record.evaluations == 3
    assert record.best == 1.0


def test_best_ignores_nan_values() -> None:
    """The run's value is the smallest value seen, and NaN is not a value."""
    problem, record = make_problem(budget=5)
    problem(np.array([np.nan, 0.0]))
    problem(np.array([[np.nan, 1.0], [2.0, 0.0]]))
    assert record.evaluations == 3
    assert record.best == 4.0


def test_an_empty_batch_uses_no_budget() -> None:
    """problem(x) with n = 0 returns an empty array."""
    problem, record = make_problem(budget=1)
    assert problem(np.empty((0, 2))).shape == (0,)
    assert record.evaluations == 0


@pytest.mark.parametrize("shape", [(3,), (2, 3), (1, 2, 2), ()])
def test_a_wrong_shape_stops_with_the_three_line_error(shape: tuple[int, ...]) -> None:
    """A point of the wrong shape gives a three-line RubberRulerError."""
    problem, record = make_problem()
    with pytest.raises(RubberRulerError, match="f01_i001_d2") as raised:
        problem(np.zeros(shape))
    assert_three_lines(raised.value)
    assert record.evaluations == 0


def test_problem_evaluates_bbob_and_the_transformed_objective() -> None:
    """The value is f(x), or g(f(x)) on a transformed instance."""
    split = make_split(2, (1,), (2,), seed=42, function_ids=(3,))
    original, transformed = split.test[0], split.transformed_test[0]
    g = transformation_of(transformed)
    assert g is not None
    x = np.array([0.5, -1.5])
    points = np.array([[0.5, -1.5], [4.0, 4.0]])
    f = InstanceObjective(original)
    raw = f.many(points)
    assert f.one(x) == raw[0]
    problem = Problem(
        InstanceObjective(transformed),
        RunRecord(),
        instance_id=transformed.instance_id,
        lower_bounds=transformed.lower_bounds,
        upper_bounds=transformed.upper_bounds,
        budget=10,
        seed=0,
    )
    assert problem(x) == g(raw[:1])[0]
    np.testing.assert_array_equal(problem(points), g(raw))
