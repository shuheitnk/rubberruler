import dataclasses
import pickle

import numpy as np
import pytest
from helpers import assert_three_lines

import rubberruler as rr
from rubberruler._instances import (
    InstanceObjective,
    InstanceSplit,
    make_split,
    original_of,
    transformation_of,
)


@pytest.fixture(scope="module")
def default_split() -> InstanceSplit:
    return rr.bbob_instances()


def test_default_split_has_the_paper_sizes(default_split: InstanceSplit) -> None:
    """24 functions x (100 train, 10 test, 10 transformed test) at d = 10."""
    assert len(default_split.train) == 2400
    assert len(default_split.test) == 240
    assert len(default_split.transformed_test) == 240
    everything = list(default_split)
    assert everything == [
        *default_split.train,
        *default_split.test,
        *default_split.transformed_test,
    ]
    first = default_split.train[0]
    assert first.instance_id == "f01_i011_d10"
    assert (first.function_id, first.instance_number, first.dimension) == (1, 11, 10)
    assert first.seed == 42
    assert default_split.train[100].instance_id == "f02_i011_d10"
    assert default_split.test[0].instance_id == "f01_i001_d10"


def test_instance_ids_follow_the_fixed_format() -> None:
    """f"f{fid:02d}_i{iid:03d}_d{d}", and ~transformed after a transform."""
    split = make_split(2, (3, 1000), (5,), seed=1, function_ids=(5,))
    assert [p.instance_id for p in split.train] == ["f05_i003_d2", "f05_i1000_d2"]
    original, transformed = split.test[0], split.transformed_test[0]
    assert transformed.instance_id == "f05_i005_d2~transformed"
    assert transformed.original_instance_id == "f05_i005_d2"
    assert original.original_instance_id == "f05_i005_d2"
    assert transformed.is_transformed and not original.is_transformed
    assert original_of(transformed) == original


def test_instance_numbers_are_sorted() -> None:
    """Instances are ordered by function and then by instance number."""
    split = rr.bbob_instances(
        dimension=2, train_instance_numbers=[13, 11, 12], test_instance_numbers=[2, 1]
    )
    numbers = [p.instance_number for p in split.train[:3]]
    assert numbers == [11, 12, 13]
    assert [p.instance_number for p in split.test[:2]] == [1, 2]


def test_transform_matches_the_fixed_values() -> None:
    """f_min (ioh's optimum) and f-hat-max (the largest of 10^4
    uniform values) for seed 42, function 1, instance 1, d = 10."""
    split = make_split(10, (11,), (1,), seed=42, function_ids=(1,))
    g = transformation_of(split.transformed_test[0])
    assert g is not None
    assert g.f_min == pytest.approx(79.48, abs=1e-6)
    assert g.f_max == pytest.approx(332.54400819810616, abs=1e-6)
    assert transformation_of(split.test[0]) is None


def test_transform_gives_the_same_transformation_to_the_same_instance() -> None:
    """g depends only on the master seed, function, instance number, d."""
    first = make_split(2, (11,), (1,), seed=5, function_ids=(7,))
    second = make_split(2, (11,), (1,), seed=5, function_ids=(7,))
    other_seed = make_split(2, (11,), (1,), seed=6, function_ids=(7,))
    y = np.linspace(-10.0, 400.0, 50)
    g1 = transformation_of(first.transformed_test[0])
    g2 = transformation_of(rr.transform(second.test[0]))
    g3 = transformation_of(other_seed.transformed_test[0])
    assert g1 is not None and g2 is not None and g3 is not None
    np.testing.assert_array_equal(g1(y), g2(y))
    assert not np.array_equal(g1(y), g3(y))


def test_transforming_twice_stops() -> None:
    """Transforming a transformed instance is an error."""
    split = make_split(2, (11,), (1,), seed=5, function_ids=(1,))
    with pytest.raises(rr.RubberRulerError, match="already transformed") as raised:
        rr.transform(split.transformed_test[0])
    assert_three_lines(raised.value)


def test_instances_are_read_only_and_not_callable() -> None:
    """ProblemInstance is read-only and cannot evaluate the objective."""
    instance = make_split(2, (11,), (1,), seed=5, function_ids=(1,)).train[0]
    assert not callable(instance)
    with pytest.raises(dataclasses.FrozenInstanceError):
        instance.seed = 3  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(ValueError, match="read-only"):
        instance.lower_bounds[0] = 0.0
    np.testing.assert_array_equal(instance.lower_bounds, [-5.0, -5.0])
    np.testing.assert_array_equal(instance.upper_bounds, [5.0, 5.0])


def test_instances_survive_pickling() -> None:
    """Instances can be sent to worker processes."""
    split = make_split(2, (11,), (1,), seed=5, function_ids=(1,))
    transformed = pickle.loads(pickle.dumps(split.transformed_test[0]))
    assert transformed == split.transformed_test[0]
    points = np.array([[0.1, 0.2], [3.0, -4.0]])
    np.testing.assert_array_equal(
        InstanceObjective(transformed).many(points),
        InstanceObjective(split.transformed_test[0]).many(points),
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"dimension": 1}, "dimension=1"),
        ({"dimension": 2.0}, "dimension=2.0"),
        ({"dimension": True}, "dimension=True"),
        ({"seed": -1}, "seed=-1"),
        ({"seed": 2**32}, "seed=4294967296"),
        ({"train_instance_numbers": []}, "train_instance_numbers is empty"),
        ({"train_instance_numbers": [0, 11]}, "contains 0"),
        ({"train_instance_numbers": [11, 11]}, "contains 11 more than once"),
        ({"test_instance_numbers": "1"}, "test_instance_numbers is a str"),
        ({"test_instance_numbers": [1.0]}, "contains 1.0"),
        ({"test_instance_numbers": range(5, 20)}, "share 11, 12, 13, 14, 15, and"),
    ],
)
def test_bad_settings_stop_with_a_three_line_error(
    arguments: dict[str, object], message: str
) -> None:
    """Settings outside their allowed values stop."""
    with pytest.raises(rr.RubberRulerError, match=message) as raised:
        rr.bbob_instances(**arguments)  # pyright: ignore[reportArgumentType]
    assert_three_lines(raised.value)


def test_transform_rejects_other_objects() -> None:
    """transform takes only a ProblemInstance."""
    with pytest.raises(rr.RubberRulerError, match="transform got str") as raised:
        rr.transform("f01_i001_d10")  # pyright: ignore[reportArgumentType]
    assert_three_lines(raised.value)


@pytest.mark.parametrize(
    ("instances", "message"),
    [
        (5, "got instances of type int"),
        ([], "got no instances"),
        (["f01_i001_d2"], "got str in instances"),
        ("twice", "got f01_i001_d2 twice in instances"),
    ],
)
def test_instances_that_are_not_instances_stop(instances: object, message: str) -> None:
    """Instances must be ProblemInstance objects, each once."""
    split = make_split(2, (1,), (2,), seed=3, function_ids=(1,))
    if instances == "twice":
        instances = [split.train[0], split.train[0]]
    portfolio = {"a": lambda p: p(p.lower_bounds), "b": lambda p: p(p.upper_bounds)}
    with pytest.raises(rr.RubberRulerError, match=message) as raised:
        rr.measure_performance(portfolio, instances, cache_dir=None)  # pyright: ignore[reportArgumentType]
    assert_three_lines(raised.value)
