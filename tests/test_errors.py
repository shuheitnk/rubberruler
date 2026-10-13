import os
import subprocess
import sys
from pathlib import Path

import rubberruler
from rubberruler._errors import error


def test_the_error_is_shown_as_rubberruler_error() -> None:
    """The traceback shows rubberruler.RubberRulerError."""
    assert rubberruler.RubberRulerError.__module__ == "rubberruler"
    assert issubclass(rubberruler.RubberRulerError, Exception)


def test_error_has_what_why_and_fix_lines() -> None:
    """The message is three lines: what, Why:, Fix:."""
    message = str(error("x went wrong on f01_i001_d2.", why="y.", fix="z."))
    assert message.split("\n") == ["x went wrong on f01_i001_d2.", "Why: y.", "Fix: z."]


def import_with_a_broken_ioh(folder: Path, message: str) -> list[str]:
    """The last lines of `import rubberruler` when importing ioh raises this."""
    (folder / "ioh").mkdir()
    (folder / "ioh" / "__init__.py").write_text(
        f"raise ImportError({message!r})\n", encoding="utf-8"
    )
    completed = subprocess.run(
        [sys.executable, "-c", "import rubberruler"],
        env={**os.environ, "PYTHONPATH": str(folder)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    return completed.stderr.strip().splitlines()


def test_a_missing_visual_cpp_runtime_says_what_to_install(tmp_path: Path) -> None:
    """When ioh's compiled part cannot load on Windows, the import error
    says in three lines to install the Visual C++ Redistributable."""
    windows = "DLL load failed while importing iohcpp: The module could not be found."
    lines = import_with_a_broken_ioh(tmp_path, windows)
    assert lines[-3] == (
        "ImportError: ioh, the library behind the BBOB functions, could not load its "
        f"compiled part ({windows})."
    )
    assert lines[-2].startswith("Why: ioh needs the Microsoft Visual C++ ")
    assert lines[-1] == (
        "Fix: install its X64 package from "
        "https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist and "
        "start Python again."
    )


def test_other_import_errors_of_ioh_are_left_as_they_are(tmp_path: Path) -> None:
    """Only the DLL error receives the three lines."""
    lines = import_with_a_broken_ioh(tmp_path, "something else")
    assert lines[-1] == "ImportError: something else"
