import json
import subprocess
import sys
from pathlib import Path

import pytest

FAIL = Path(__file__).parent / "typing" / "fail"


@pytest.mark.locked
def test_pyright_reports_the_marked_lines_and_no_others() -> None:
    """Wrong input types are errors at the call, with no import needed."""
    completed = subprocess.run(
        [sys.executable, "-m", "pyright", "-p", str(FAIL), "--outputjson"],
        capture_output=True,
        text=True,
        check=False,
    )
    report = json.loads(completed.stdout)
    reported = {
        (Path(item["file"]).name, item["range"]["start"]["line"] + 1)
        for item in report["generalDiagnostics"]
        if item["severity"] == "error"
    }
    expected = {
        (path.name, number)
        for path in sorted(FAIL.glob("*.py"))
        for number, line in enumerate(path.read_text("utf-8").splitlines(), start=1)
        if line.rstrip().endswith("# expect-error")
    }
    assert len(expected) == 6
    assert reported == expected
