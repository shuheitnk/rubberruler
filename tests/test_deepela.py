"""examples/deepela.py: DeepELA's features with the quickstart's portfolio.

These tests need DeepELA (github.com/mvseiler/deep_ela at c036867, on PYTHONPATH) and
PyTorch, which run in the DeepELA environment (Python 3.12; the deepela job of CI).
Elsewhere they are skipped.
"""

import math
import runpy
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from helpers import assert_three_lines
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor

import rubberruler
from rubberruler import _run_test
from rubberruler._features import preprocess

pytest.importorskip("deep_ela")

ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "examples" / "deepela.py"
QUICKSTART = ROOT / "examples" / "quickstart.py"
# d = 2, 4 functions, 5 training and 3 test instances each: 44 instances, 396 runs.
FUNCTION_IDS = (1, 8, 15, 22)

Extractor = Callable[[NDArray[np.float64], NDArray[np.float64]], Mapping[str, float]]


def run_example(monkeypatch: pytest.MonkeyPatch, run_test: object) -> dict[str, object]:
    """Run the example as a script, with ``rr.run_test`` replaced by ``run_test``."""
    monkeypatch.setattr(rubberruler, "run_test", run_test)
    return runpy.run_path(str(EXAMPLE), run_name="__main__")


def test_deep_ela_features_are_48_finite_values_that_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """48 finite values under fixed keys, the same values for the same input
    (torch.manual_seed before each call), and other values for rank-based y."""
    names = run_example(monkeypatch, lambda *args, **kwargs: "(not run)")
    deep_ela_features = cast(Extractor, names["deep_ela_features"])
    X = np.random.default_rng(0).uniform(-5, 5, size=(500, 10))
    y = np.sum(X**2, axis=1)
    value = deep_ela_features(X, preprocess(y, "value")[0])
    again = deep_ela_features(X, preprocess(y, "value")[0])
    rank = deep_ela_features(X, preprocess(y, "rank")[0])
    assert list(value) == [f"deep_ela.large_50d_v1.X{i}" for i in range(48)]
    assert all(math.isfinite(number) for number in value.values())
    assert value == again
    assert np.mean([abs(value[name] - rank[name]) for name in value]) > 0.05


def test_twelve_dimensions_stop_in_the_precheck(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """At d = 12 the extractor raises ValueError (in the large models the
    objective value would share a slot with the last coordinate), and run_test stops
    in the pre-check, before any run, with that ValueError as the cause."""
    names = run_example(monkeypatch, lambda *args, **kwargs: "(not run)")
    with pytest.raises(rubberruler.RubberRulerError) as raised:
        _run_test.run_test(
            cast(
                Mapping[str, Callable[[rubberruler.Problem], object]],
                names["portfolio"],
            ),
            cast(Extractor, names["deep_ela_features"]),
            RandomForestRegressor(random_state=0),
            dimension=12,
            cache_dir=tmp_path,
        )
    assert_three_lines(raised.value)
    assert str(raised.value).startswith(
        "feature_extractor raised ValueError on f01_i011_d12 (value)."
    )
    assert isinstance(raised.value.__cause__, ValueError)
    assert "at most 11 dimensions, but received 12" in str(raised.value.__cause__)
    assert not (tmp_path / "runs").exists()


@pytest.mark.slow
def test_after_the_quickstart_only_the_features_are_computed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With the same small settings, the example takes every run of
    the quickstart's portfolio from the cache, and prints a verdict."""

    def small_run_test(
        portfolio: object, feature_extractor: object, predictor: object, **_: object
    ) -> _run_test.RubberRulerResult:
        return _run_test.run(
            portfolio,  # pyright: ignore[reportArgumentType]
            feature_extractor,  # pyright: ignore[reportArgumentType]
            predictor,
            dimension=2,
            train_instance_numbers=range(11, 16),
            test_instance_numbers=range(1, 4),
            budget_per_dimension=50,
            n_runs=3,
            n_samples_per_dimension=50,
            seed=42,
            n_jobs=-1,
            cache_dir=tmp_path,
            function_ids=FUNCTION_IDS,
        )

    monkeypatch.setattr(rubberruler, "run_test", small_run_test)
    runpy.run_path(str(QUICKSTART), run_name="__main__")
    capsys.readouterr()
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    out, err = capsys.readouterr()
    assert out.startswith("Rubber Ruler Test: use ")
    assert "rubberruler: 396 runs (396 cached, 0 to run)" in err
