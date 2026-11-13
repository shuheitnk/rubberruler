"""examples/example4.py: GA, PSO, and Nelder-Mead from pymoo.

pymoo is in the uv group examples and in the pflacco environment, so this runs in both
CI jobs; where pymoo is not installed, it is skipped.
"""

import runpy
from pathlib import Path

import pytest

import rubberruler
from rubberruler import _run_test

pytest.importorskip("pymoo")

EXAMPLE = Path(__file__).parents[1] / "examples" / "example4.py"


@pytest.mark.slow
def test_the_pymoo_example_decides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With small settings (4 functions, 44 instances, 396 runs of 100
    evaluations in worker processes), the example prints a verdict."""

    def small_run_test(
        portfolio: object, feature_extractor: object, predictor: object
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
            function_ids=(1, 8, 15, 22),
        )

    monkeypatch.setattr(rubberruler, "run_test", small_run_test)
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    assert capsys.readouterr().out.startswith("Rubber Ruler Test: use ")
