"""Checks, a smallest run, and README access shared by the tests."""

import re
from collections.abc import Mapping
from pathlib import Path

import algorithms
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from rubberruler._run_test import RubberRulerResult, run

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
    portfolio: Mapping[str, object] = PORTFOLIO,
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
