"""Checks, a smallest run, and README access shared by the tests."""

import re
from collections.abc import Mapping
from pathlib import Path

import algorithms
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import rubberruler as rr
from rubberruler._run_test import RubberRulerResult, run
from rubberruler._seeding import generator

README = Path(__file__).parents[1] / "README.md"
EXAMPLES_README = Path(__file__).parents[1] / "examples" / "README.md"

PORTFOLIO = {
    "random": algorithms.random_search,
    "batch": algorithms.batch_random_search,
    "center": algorithms.center,
}


def assert_three_lines(raised: BaseException) -> None:
    """The message is three lines: what happened, Why:, and Fix:."""
    lines = str(raised).split("\n")
    assert len(lines) == 3, lines
    assert lines[0] and not lines[0].startswith(("Why:", "Fix:")), lines
    assert lines[1].startswith("Why: "), lines
    assert lines[2].startswith("Fix: "), lines


def extract(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """A small feature extractor for the runs of the tests."""
    return {
        "mean": float(np.mean(y)),
        "spread": float(np.std(y)),
        "corner": float(y[np.argmin(X.sum(axis=1))]),
    }


def tiny_run(
    portfolio: Mapping[str, object] | None = PORTFOLIO,
    extractor: object = extract,
    predictor: object = None,
    **overrides: object,
) -> RubberRulerResult:
    """The smallest run: d = 2, 2 functions, 2 instances per set, budget 50, 3 runs."""
    settings: dict[str, object] = {
        "dimension": 2,
        "train_instance_numbers": (11, 12),
        "test_instance_numbers": (1, 2),
        "budget_per_dimension": 25,
        "n_runs": 3,
        "n_samples_per_dimension": 10,
        "seed": 42,
        "n_jobs": 1,
        "cache_dir": None,
        **overrides,
    }
    if predictor is None:
        predictor = RandomForestRegressor(n_estimators=10, random_state=0)
    return run(
        portfolio,  # pyright: ignore[reportArgumentType]
        extractor,  # pyright: ignore[reportArgumentType]
        predictor,
        function_ids=(1, 2),
        **settings,  # pyright: ignore[reportArgumentType]
    )


def readme_python_blocks(readme: Path = README) -> list[str]:
    """The ```python blocks of README.md (or of examples/README.md), in order."""
    return re.findall(r"```python\n(.*?)```", readme.read_text("utf-8"), re.DOTALL)


def readme_card() -> str:
    """The card printed in the README (the ```text block that starts with it)."""
    blocks = re.findall(r"```text\n(.*?)\n```", README.read_text("utf-8"), re.DOTALL)
    (card,) = [block for block in blocks if block.startswith("Rubber Ruler Test:")]
    return card


def run_readme_block(marker: str) -> dict[str, object]:
    """Run the one block of examples/README.md that contains ``marker`` and return
    its names."""
    blocks = readme_python_blocks(EXAMPLES_README)
    (code,) = [block for block in blocks if marker in block]
    names: dict[str, object] = {}
    exec(compile(code, "examples/README.md", "exec"), names)
    return names


def compute_features(*arguments: object, **settings: object) -> pd.DataFrame:
    """rr.compute_features in this process (n_jobs=1), unless the test says otherwise.

    Extractors defined in a test module cannot be loaded by worker processes, which
    import the test modules under other names.
    """
    settings.setdefault("n_jobs", 1)
    return rr.compute_features(*arguments, **settings)  # pyright: ignore[reportArgumentType]


# The first coordinate of the feature sample of f01_i012_d2 with seed 7.
FAILING_POINT = generator("features", 7, "bbob", 1, 12, 2).uniform(
    np.full(2, -5.0), np.full(2, 5.0), size=(100, 2)
)[0, 0]


def fails_after_the_first(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """An extractor that fails only on f01_i012_d2 of seed 7, which the pre-check
    (the first instance of each function) does not try."""
    if X[0, 0] == FAILING_POINT:
        return {"mean": 1 / 0}
    return {"mean": float(np.mean(y))}
