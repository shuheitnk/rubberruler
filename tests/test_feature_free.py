import importlib.util
import runpy
from pathlib import Path

import pytest
from helpers import EXAMPLES_README, readme_python_blocks, tiny_run

import rubberruler

ROOT = Path(__file__).parents[1]

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="needs torch"
)


def test_a_feature_free_model_is_tested_through_the_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """examples/example7.py runs with run_test shrunk: the extractor returns the
    sample, and the network reads it back as points of size d + 1. The extractor in
    examples/README.md is that of the script."""
    example = ROOT / "examples" / "example7.py"
    blocks = readme_python_blocks(EXAMPLES_README)
    (block,) = [code for code in blocks if "def sample" in code]
    assert block in example.read_text("utf-8")

    def smallest_run_test(
        portfolio: object, extractor: object, predictor: object, **_: object
    ) -> object:
        return tiny_run(portfolio, extractor, predictor)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(rubberruler, "run_test", smallest_run_test)
    result = runpy.run_path(str(example))["result"]
    assert str(result).startswith("Rubber Ruler Test: use ")
    features = result.features["value"]
    # tiny_run takes 10 points per dimension at d = 2: 20 points of size 3.
    assert features.shape[1] == 60
