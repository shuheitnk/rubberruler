import os
import subprocess
import sys
from pathlib import Path

import rubberruler

TESTS = Path(__file__).parent

# A whole run with stdout and stderr in cp932 (the code page of Japanese Windows),
# both strict: Python writes stderr with backslashreplace, which would hide a character
# that cp932 cannot encode.
CP932_RUN = """
import sys
import traceback

sys.stderr.reconfigure(errors="strict")

import numpy as np
from helpers import tiny_run
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

import rubberruler as rr


def extract(X, y):
    corner = float(y[np.argmin(X.sum(axis=1))])
    return {"mean": float(np.mean(y)), "corner": corner if corner > 0.2 else np.nan}


result = tiny_run(
    extractor=extract,
    predictor=make_pipeline(SimpleImputer(), RandomForestRegressor(n_estimators=10)),
    cache_dir=".rubberruler",
)
print(result)
features = result.features["rank"]
fitted = rr.fit_predictor(
    make_pipeline(SimpleImputer(), RandomForestRegressor(n_estimators=5)),
    features,
    result.performance,
    instances=result.split.train,
)
rr.r2_score(fitted, features, result.performance * 0 + 0.5, instances=result.split.test)
try:
    tiny_run(extractor=lambda X, y: [1.0])
except rr.RubberRulerError:
    traceback.print_exc()
"""


def test_the_package_source_is_ascii() -> None:
    """The card, warnings, errors, and docstrings are ASCII."""
    for path in sorted(Path(rubberruler.__file__).parent.glob("*.py")):
        text = path.read_text("utf-8")
        assert text.isascii(), (path.name, sorted({c for c in text if not c.isascii()}))


def test_a_whole_run_prints_in_cp932(tmp_path: Path) -> None:
    """Nothing fails with stdout and stderr in cp932.

    The card with its warnings goes to stdout; the plan, the progress, the figure line,
    logged warnings, and the traceback of a RubberRulerError go to stderr.
    """
    env = {
        **os.environ,
        "PYTHONIOENCODING": "cp932",
        "PYTHONPATH": os.pathsep.join([str(TESTS), os.environ.get("PYTHONPATH", "")]),
    }
    env.pop("TQDM_DISABLE", None)
    completed = subprocess.run(
        [sys.executable, "-c", CP932_RUN],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=300,
        check=False,
    )
    stderr = completed.stderr.decode("cp932", errors="replace")
    assert completed.returncode == 0, stderr
    stdout = completed.stdout.decode("ascii")
    assert "  R^2 on the transformed test instances: " in stdout
    assert (
        "  predictor has random_state=None, so R^2 may change between calls" in stdout
    )
    assert " NaN features" in stdout
    for text in (
        "rubberruler: 108 runs (0 cached, 108 to run) in .rubberruler",
        "[2/4] Measuring performance",
        "rubberruler: figure saved to .rubberruler",
        "R^2 is undefined because all performance values are equal; returned nan",
        "rubberruler.RubberRulerError: feature_extractor returned a list on ",
    ):
        assert text in stderr
    # File paths in the traceback are the machine's; everything else is ASCII.
    lines = [line for line in stderr.splitlines() if 'File "' not in line]
    assert "\n".join(lines).isascii()
