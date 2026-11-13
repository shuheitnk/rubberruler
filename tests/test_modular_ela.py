"""examples/example5.py: ModularCMAES, ModDE, and pflacco's ELA features.

These tests need pflacco, ModularCMAES, and ModDE, which run in the pflacco environment
(Python 3.11; the pflacco job of CI). Elsewhere they are skipped.
"""

import math
import runpy
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

import rubberruler
from rubberruler import _run_test

for module in ("pflacco", "modcma", "modde"):
    pytest.importorskip(module)

EXAMPLE = Path(__file__).parents[1] / "examples" / "example5.py"
# d = 10 (at d = 2 pflacco's ela_level fails on f7), 4 functions, 5 training and 3 test
# instances each: 44 instances, 880 runs, and the ELA features of 88.
FUNCTION_IDS = (1, 8, 15, 22)
BUDGET_PER_DIMENSION = 100
N_RUNS = 5


def run_example(monkeypatch: pytest.MonkeyPatch, run_test: object) -> dict[str, object]:
    """Run the example as a script, with ``rr.run_test`` replaced by ``run_test``."""
    monkeypatch.setattr(rubberruler, "run_test", run_test)
    return runpy.run_path(str(EXAMPLE), run_name="__main__")


def test_ela_features_leave_out_the_costs_and_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No key has .costs_ (such as seconds), and the same input repeats."""
    names = run_example(monkeypatch, lambda *args, **kwargs: "(not run)")
    ela_features = cast(
        Callable[[NDArray[np.float64], NDArray[np.float64]], dict[str, float]],
        names["ela_features"],
    )
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(500, 10))
    y = np.sum((10 * X - 5) ** 2, axis=1)
    y = (y - y.min()) / (y.max() - y.min())
    first = ela_features(X, y)
    second = ela_features(X, y)
    assert len(first) == 55
    assert [name for name in first if ".costs_" in name] == []
    assert list(first) == list(second)
    for name, value in first.items():
        both_nan = math.isnan(value) and math.isnan(second[name])
        assert value == second[name] or both_nan, (name, value, second[name])


@pytest.mark.slow
def test_the_example_decides_and_its_runs_are_cached(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With small settings the example prints a verdict, and
    measure_performance then takes every run of its portfolio from the cache."""

    def small_run_test(
        portfolio: object, feature_extractor: object, predictor: object
    ) -> _run_test.RubberRulerResult:
        return _run_test.run(
            portfolio,  # pyright: ignore[reportArgumentType]
            feature_extractor,  # pyright: ignore[reportArgumentType]
            predictor,
            dimension=10,
            train_instance_numbers=range(11, 16),
            test_instance_numbers=range(1, 4),
            budget_per_dimension=BUDGET_PER_DIMENSION,
            n_runs=N_RUNS,
            n_samples_per_dimension=50,
            seed=42,
            n_jobs=-1,
            cache_dir=tmp_path,
            function_ids=FUNCTION_IDS,
        )

    names = run_example(monkeypatch, small_run_test)
    assert capsys.readouterr().out.startswith("Rubber Ruler Test: use ")
    result = cast(_run_test.RubberRulerResult, names["result"])
    portfolio = cast(
        Mapping[str, Callable[[rubberruler.Problem], object]], names["portfolio"]
    )
    split = result.split
    instances = [*split.train, *split.test, *split.transformed_test]
    rubberruler.measure_performance(
        portfolio,
        instances,
        budget_per_dimension=BUDGET_PER_DIMENSION,
        n_runs=N_RUNS,
        cache_dir=tmp_path,
    )
    runs = f"{len(instances) * len(portfolio) * N_RUNS:,}"
    assert (
        f"rubberruler: {runs} runs ({runs} cached, 0 to run)" in capsys.readouterr().err
    )
