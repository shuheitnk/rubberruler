import functools
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
import threading
import types
from collections.abc import Mapping
from pathlib import Path

import algorithms
import numpy as np
import pandas as pd
import pytest
from helpers import assert_three_lines

import rubberruler as rr
from rubberruler import _cache, _performance
from rubberruler._cache import identify, package_version
from rubberruler._instances import InstanceSplit, make_split

PLAN = re.compile(r"rubberruler: ([\d,]+) runs \(([\d,]+) cached, ([\d,]+) to run\)")

SEARCHES = """
import numpy as np


def scale():
    return {scale}


def uses_helper(problem):
    rng = np.random.default_rng(problem.seed)
    while True:
        problem(rng.uniform(problem.lower_bounds, problem.upper_bounds) * scale())


def independent(problem):
    rng = np.random.default_rng(problem.seed)
    while True:
        problem(rng.uniform(problem.lower_bounds, problem.upper_bounds){tail})
"""


@pytest.fixture(scope="module")
def split() -> InstanceSplit:
    return make_split(2, (1,), (2,), seed=3, function_ids=(1, 15))


def load_module(
    folder: Path, name: str, source: str, monkeypatch: pytest.MonkeyPatch
) -> types.ModuleType:
    """Write the module and import it, as a user edits and reruns a script."""
    path = folder / f"{name}.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def measured(
    portfolio: Mapping[str, object],
    split: InstanceSplit,
    cache_dir: Path | None,
    capsys: pytest.CaptureFixture[str],
    n_runs: int = 3,
) -> tuple[pd.DataFrame, tuple[int, int, int]]:
    """measure_performance, and the numbers of its plan line: total, cached, to run."""
    performance = rr.measure_performance(
        portfolio,  # pyright: ignore[reportArgumentType]
        split,
        budget_per_dimension=10,
        n_runs=n_runs,
        n_jobs=1,
        cache_dir=cache_dir,
    )
    err = capsys.readouterr().err
    found = PLAN.search(err)
    assert found is not None, err
    total, cached, to_run = (int(value.replace(",", "")) for value in found.groups())
    return performance, (total, cached, to_run)


def test_the_second_call_uses_the_cache(
    split: InstanceSplit, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Finished runs are kept, and the same call runs nothing again."""
    portfolio = {
        "random": algorithms.random_search,
        "batch": algorithms.batch_random_search,
    }
    first, counts = measured(portfolio, split, tmp_path / "cache", capsys)
    assert counts == (36, 0, 36)
    second, counts = measured(portfolio, split, tmp_path / "cache", capsys)
    assert counts == (36, 36, 0)
    pd.testing.assert_frame_equal(first, second)
    assert list((tmp_path / "cache" / "runs").glob("*.parquet"))


def test_the_plan_line_names_the_folder(
    split: InstanceSplit, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The plan line gives the runs, the cached ones, and the folder."""
    portfolio = {
        "random": algorithms.random_search,
        "batch": algorithms.batch_random_search,
    }
    measured(portfolio, split, tmp_path / "plan", capsys)
    rr.measure_performance(
        portfolio,
        split,
        budget_per_dimension=10,
        n_runs=3,
        n_jobs=1,
        cache_dir=tmp_path / "plan",
    )
    err = capsys.readouterr().err
    folder = os.path.join(tmp_path / "plan", "")
    assert f"rubberruler: 36 runs (36 cached, 0 to run) in {folder}" in err
    rr.measure_performance(
        portfolio, split, budget_per_dimension=10, n_runs=3, n_jobs=1, cache_dir=None
    )
    assert "rubberruler: 36 runs (0 cached, 36 to run) without a cache" in (
        capsys.readouterr().err
    )


def test_an_interrupted_call_resumes_with_the_same_result(
    split: InstanceSplit,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ctrl-C keeps the finished runs; the same call resumes."""
    portfolio = {
        "random": algorithms.random_search,
        "batch": algorithms.batch_random_search,
    }
    expected, _ = measured(portfolio, split, None, capsys)
    run_task = _performance.run_task
    calls = {"count": 0}

    def interrupted(*arguments: object) -> object:
        calls["count"] += 1
        if calls["count"] == 3:
            raise KeyboardInterrupt
        return run_task(*arguments)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(_performance, "run_task", interrupted)
    with pytest.raises(KeyboardInterrupt):
        rr.measure_performance(
            portfolio,
            split,
            budget_per_dimension=10,
            n_runs=3,
            n_jobs=1,
            cache_dir=tmp_path,
        )
    assert capsys.readouterr().err.endswith(
        "rubberruler: stopped. 6 of 36 runs are saved. "
        "Run the same call again to resume.\n"
    )
    monkeypatch.setattr(_performance, "run_task", run_task)
    resumed, counts = measured(portfolio, split, tmp_path, capsys)
    assert counts == (36, 6, 30)
    pd.testing.assert_frame_equal(resumed, expected)


def test_ctrl_c_without_a_cache_says_nothing_is_saved(
    split: InstanceSplit,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With cache_dir=None, the stop line says that no runs are saved."""

    def interrupted(*arguments: object) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(_performance, "run_task", interrupted)
    with pytest.raises(KeyboardInterrupt):
        rr.measure_performance(
            {"random": algorithms.random_search, "batch": algorithms.center},
            split,
            budget_per_dimension=10,
            n_runs=3,
            n_jobs=1,
            cache_dir=None,
        )
    assert capsys.readouterr().err.endswith(
        "rubberruler: stopped. No runs are saved because cache_dir is None.\n"
    )


def test_changing_a_function_runs_only_that_function_again(
    split: InstanceSplit,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An edited algorithm is run again; the others come from the cache."""
    cache = tmp_path / "cache"
    old = load_module(
        tmp_path, "edited", SEARCHES.format(scale=1.0, tail=""), monkeypatch
    )
    portfolio = {"helper": old.uses_helper, "independent": old.independent}
    measured(portfolio, split, cache, capsys)
    new = load_module(
        tmp_path, "edited", SEARCHES.format(scale=1.0, tail=" * 0.5"), monkeypatch
    )
    portfolio = {"helper": new.uses_helper, "independent": new.independent}
    _, counts = measured(portfolio, split, cache, capsys)
    assert counts == (36, 18, 18)


def test_the_plan_names_the_algorithms_that_run_again(
    split: InstanceSplit,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When only some runs are cached, a second line says which
    algorithms run, so an edited one stands out; it is not there otherwise."""
    cache = tmp_path / "cache"

    def plan_lines(module: types.ModuleType) -> list[str]:
        portfolio = {"helper": module.uses_helper, "independent": module.independent}
        rr.measure_performance(
            portfolio,
            split,
            budget_per_dimension=10,
            n_runs=3,
            n_jobs=1,
            cache_dir=cache,
        )
        err = capsys.readouterr().err
        return [line for line in err.splitlines() if line.startswith("rubberruler:")]

    old = load_module(
        tmp_path, "named", SEARCHES.format(scale=1.0, tail=""), monkeypatch
    )
    assert len(plan_lines(old)) == 1
    new = load_module(
        tmp_path, "named", SEARCHES.format(scale=1.0, tail=" * 0.5"), monkeypatch
    )
    lines = plan_lines(new)
    assert lines[1] == "rubberruler: to run: independent 18"
    assert len(plan_lines(new)) == 1


def test_changing_a_global_helper_runs_its_callers_again(
    split: InstanceSplit,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Editing a helper runs again only the algorithms that call it."""
    cache = tmp_path / "cache"
    old = load_module(
        tmp_path, "helped", SEARCHES.format(scale=1.0, tail=""), monkeypatch
    )
    measured(
        {"helper": old.uses_helper, "independent": old.independent},
        split,
        cache,
        capsys,
    )
    new = load_module(
        tmp_path, "helped", SEARCHES.format(scale=0.25, tail=""), monkeypatch
    )
    _, counts = measured(
        {"helper": new.uses_helper, "independent": new.independent},
        split,
        cache,
        capsys,
    )
    assert counts == (36, 18, 18)


def make_search(scale: float) -> object:
    def search(problem: rr.Problem) -> None:
        rng = np.random.default_rng(problem.seed)
        while True:
            problem(rng.uniform(problem.lower_bounds, problem.upper_bounds) * scale)

    return search


def test_closure_values_are_part_of_the_identifier() -> None:
    """make_search(1.0) and make_search(2.0) share their source only."""
    same = identify("search", make_search(1.0))
    assert same == identify("search", make_search(1.0))
    assert same.identifier != identify("search", make_search(2.0)).identifier
    assert same.warning is None
    assert same.identifier != identify("other name", make_search(1.0)).identifier


def test_partials_methods_and_callable_objects_are_identified() -> None:
    """partial arguments and object attributes are part of the identifier."""

    def scaled(problem: rr.Problem, scale: float) -> None:
        problem(np.zeros(problem.dimension) * scale)

    class Search:
        def __init__(self, scale: float) -> None:
            self.scale = scale

        def __call__(self, problem: rr.Problem) -> None:
            problem(np.zeros(problem.dimension) * self.scale)

        def run(self, problem: rr.Problem) -> None:
            self(problem)

    pairs = [
        (functools.partial(scaled, scale=1.0), functools.partial(scaled, scale=2.0)),
        (Search(1.0), Search(2.0)),
        (Search(1.0).run, Search(2.0).run),
    ]
    for first, second in pairs:
        assert identify("a", first) == identify("a", first)
        assert identify("a", first).identifier != identify("a", second).identifier
        assert identify("a", first).warning is None


def test_identifiers_are_the_same_in_another_process() -> None:
    """The same function has the same identifier in another process."""
    code = (
        f"import sys; sys.path.insert(0, {str(Path(__file__).parent)!r}); "
        "import algorithms; from rubberruler._cache import identify; "
        "print(identify('random', algorithms.random_search).identifier)"
    )
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert output == identify("random", algorithms.random_search).identifier


def test_functions_that_cannot_be_identified_are_warned(
    split: InstanceSplit, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Name only, or name and source only, is warned."""
    lock = threading.Lock()

    def locked(problem: rr.Problem) -> None:
        with lock:
            algorithms.random_search(problem)

    assert (
        identify("max", max).warning
        == "max: cached by name only (source not available)"
    )
    assert identify("locked", locked).warning == (
        "locked: cached by name and source only (some values could not be hashed)"
    )
    with caplog.at_level(logging.WARNING, logger="rubberruler"):
        rr.measure_performance(
            {"random": algorithms.random_search, "locked": locked},
            split,
            budget_per_dimension=5,
            n_runs=1,
            n_jobs=1,
            cache_dir=tmp_path,
        )
    assert caplog.messages == [
        "locked: cached by name and source only (some values could not be hashed)"
    ]


def test_libraries_are_identified_by_their_version() -> None:
    """Libraries by package and version, the standard library by Python."""
    assert package_version("numpy.random") == f"numpy {np.__version__}"
    assert package_version("os.path") == f"python {sys.version.split()[0]}"
    assert package_version("not_a_module_anywhere") is None


def install(folder: Path, name: str, *, record: bool, editable: bool) -> None:
    """Put a package and its dist-info into folder, as pip install --target does."""
    (folder / name).mkdir(parents=True)
    (folder / name / "__init__.py").write_text(
        "from math import fsum\n\n\ndef spread(values):\n    return fsum(values)\n",
        encoding="utf-8",
    )
    info = folder / f"{name}-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n", encoding="utf-8"
    )
    (info / "top_level.txt").write_text(f"{name}\n", encoding="utf-8")
    if record:
        (info / "RECORD").write_text(
            f"{name}/__init__.py,,\n{info.name}/METADATA,,\n{info.name}/RECORD,,\n",
            encoding="utf-8",
        )
    if editable:
        (info / "direct_url.json").write_text(
            json.dumps({"url": "file:///project", "dir_info": {"editable": True}}),
            encoding="utf-8",
        )


@pytest.mark.parametrize(
    ("record", "editable", "library"),
    [(True, False, True), (True, True, False), (False, False, False)],
)
def test_libraries_installed_outside_site_packages_are_libraries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record: bool,
    editable: bool,
    library: bool,
) -> None:
    """A package that an installer put outside site-packages (pip install
    --target, uv run --with) is a library, identified by its version without a
    warning; an editable install, or one without a RECORD, is the user's code."""
    name = f"outside_{record}_{editable}".lower()
    install(tmp_path / "target", name, record=record, editable=editable)
    monkeypatch.syspath_prepend(str(tmp_path / "target"))
    monkeypatch.delitem(sys.modules, name, raising=False)
    _cache._distributions.cache_clear()  # pyright: ignore[reportPrivateUsage]
    try:
        user = load_module(
            tmp_path,
            f"uses_{name}",
            f"from {name} import spread\n\n\ndef search(problem):\n"
            "    problem([spread([0.0])] * problem.dimension)\n",
            monkeypatch,
        )
        assert _cache.is_library(name) is library
        name_only = "search: cached by name only (source not available)"
        assert identify("search", user.search).warning == (
            None if library else name_only
        )
    finally:
        _cache._distributions.cache_clear()  # pyright: ignore[reportPrivateUsage]
        _cache.installed_folder.cache_clear()


VIEW = """
import numpy as np


class View:
    scale = 1.0

    def __init__(self, problem):
        self.problem = problem

    def evaluate(self, x):
        return self.problem(x * self.scale){tail}

    @staticmethod
    def start(problem):
        return np.random.default_rng(problem.seed)


def uses_view(problem):
    view = View(problem)
    rng = View.start(problem)
    while True:
        view.evaluate(rng.uniform(problem.lower_bounds, problem.upper_bounds))
"""


def test_classes_of_the_user_are_identified_by_their_methods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An algorithm that uses a class of the user (such as a wrapper of
    the problem for another library) is identified with the class's bases, methods,
    and values, without a warning; changing a method changes the identifier."""
    first = load_module(tmp_path, "view_a", VIEW.format(tail=""), monkeypatch)
    second = load_module(tmp_path, "view_b", VIEW.format(tail=" + 0.0"), monkeypatch)
    identity = identify("view", first.uses_view)
    assert identity.warning is None
    assert identity == identify("view", first.uses_view)
    assert identity.identifier != identify("view", second.uses_view).identifier
    monkeypatch.setattr(first.View, "scale", 2.0)
    assert identity.identifier != identify("view", first.uses_view).identifier


def test_functions_of_the_main_script_stay_user_functions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A function of __main__ keeps its identifier from its source when
    sys.modules["__main__"] is later a library module (as after runpy under
    python -m pytest)."""

    def search(problem: rr.Problem) -> None:
        problem(np.zeros(problem.dimension))

    monkeypatch.setattr(search, "__module__", "__main__")
    script = types.ModuleType("__main__")
    script.__file__ = str(tmp_path / "experiment.py")
    monkeypatch.setitem(sys.modules, "__main__", script)
    as_script = identify("search", search)
    monkeypatch.setitem(sys.modules, "__main__", json)
    assert identify("search", search) == as_script
    assert as_script.warning is None


def test_other_settings_are_not_taken_from_the_cache(
    split: InstanceSplit, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The budget and the master seed are part of the key."""
    portfolio = {
        "random": algorithms.random_search,
        "batch": algorithms.batch_random_search,
    }
    measured(portfolio, split, tmp_path, capsys)
    rr.measure_performance(
        portfolio,
        split,
        budget_per_dimension=11,
        n_runs=3,
        n_jobs=1,
        cache_dir=tmp_path,
    )
    assert "(0 cached, 36 to run)" in capsys.readouterr().err
    other_seed = make_split(2, (1,), (2,), seed=4, function_ids=(1, 15))
    _, counts = measured(portfolio, other_seed, tmp_path, capsys)
    assert counts == (36, 0, 36)
    _, counts = measured(portfolio, split, tmp_path, capsys, n_runs=5)
    assert counts == (60, 36, 24)


@pytest.mark.parametrize("cache_dir", [3, "file.txt/runs"])
def test_a_cache_dir_that_cannot_be_written_stops(
    split: InstanceSplit,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache_dir: object,
) -> None:
    """cache_dir must be a folder that can be written, or None."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "file.txt").write_text("not a folder")
    portfolio = {
        "random": algorithms.random_search,
        "batch": algorithms.batch_random_search,
    }
    with pytest.raises(rr.RubberRulerError, match="cache_dir") as raised:
        rr.measure_performance(portfolio, split, cache_dir=cache_dir)  # pyright: ignore[reportArgumentType]
    assert_three_lines(raised.value)


def test_a_function_whose_source_file_holds_other_text_is_identified(
    tmp_path: Path,
) -> None:
    """Code compiled under the name of a file that holds other text (as
    exec does) still gets an identifier; inspect may fail to read its source."""
    other = tmp_path / "notes.txt"
    other.write_text("it's\n" * 5, encoding="utf-8")
    namespace: dict[str, object] = {}
    source = "def search(problem):\n    problem([0.0, 0.0])\n"
    exec(compile(source, str(other), "exec"), namespace)
    identity = identify("search", namespace["search"])
    assert identity.warning in (
        None,
        "search: cached by name only (source not available)",
    )
