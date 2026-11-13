import ast
import difflib
import importlib.util
import re
import runpy
from pathlib import Path

import pytest
from helpers import EXAMPLES_README, README, readme_card, readme_python_blocks

import rubberruler
from rubberruler import _run_test

ROOT = Path(__file__).parents[1]
QUICKSTART = ROOT / "examples" / "example1.py"
# The tools whose blocks in examples/README.md are taken from an example, and that
# example.
EXAMPLES = {
    "modcma": "example5.py",
    "modde": "example5.py",
    "pflacco": "example5.py",
    "pymoo": "example4.py",
    "deep_ela": "example6.py",
}
OPTIONAL = {"cma", "nevergrad", "torch", *EXAMPLES}


def imported_modules(code: str) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Import):
            modules |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module.split(".")[0])
    return modules


def slug(heading: str) -> str:
    """The anchor GitHub gives a heading: lower case, punctuation dropped, spaces as
    hyphens."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def anchors(markdown: str) -> set[str]:
    """The anchors of the headings of a Markdown text, outside its code blocks."""
    prose = re.sub(r"```.*?```", "", markdown, flags=re.DOTALL)
    return {slug(heading) for heading in re.findall(r"^#+ (.+)$", prose, re.MULTILINE)}


def test_the_readme_code_runs_with_the_smallest_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every block of README.md and then of examples/README.md runs, with
    run_test shrunk to 2 functions at d = 2. Each file starts from the names of
    examples/example1.py, whose last lines are the first block of README.md and
    which examples/README.md continues.

    Blocks that import pycma, Nevergrad, PyTorch, ModularCMAES, ModDE, pflacco, pymoo,
    or DeepELA are skipped when those are not installed.
    """

    def smallest_run_test(
        portfolio: object,
        feature_extractor: object,
        predictor: object,
        instances: object = None,
        performance: object = None,
        features: object = None,
        **_: object,
    ) -> _run_test.RubberRulerResult:
        # Given instances decide the dimension and the instance numbers.
        return _run_test.run(
            portfolio,  # pyright: ignore[reportArgumentType]
            feature_extractor,  # pyright: ignore[reportArgumentType]
            predictor,
            dimension=2 if instances is None else 10,
            train_instance_numbers=(11, 12) if instances is None else None,
            test_instance_numbers=(1, 2) if instances is None else None,
            instances=instances,  # pyright: ignore[reportArgumentType]
            performance=performance,  # pyright: ignore[reportArgumentType]
            features=features,  # pyright: ignore[reportArgumentType]
            budget_per_dimension=10,
            n_runs=2,
            n_samples_per_dimension=10,
            seed=42,
            n_jobs=1,
            cache_dir=tmp_path,
            function_ids=(1, 2),
        )

    monkeypatch.setattr(rubberruler, "run_test", smallest_run_test)
    for readme in (README, EXAMPLES_README):
        namespace = dict(runpy.run_path(str(QUICKSTART), run_name="__main__"))
        for code in readme_python_blocks(readme):
            missing = {
                module
                for module in imported_modules(code)
                if importlib.util.find_spec(module) is None
            }
            assert missing <= OPTIONAL, missing
            if missing:
                continue
            exec(compile(code, readme.name, "exec"), namespace)
    assert capsys.readouterr().out.startswith("Rubber Ruler Test: use ")
    assert isinstance(namespace["r2"], float)  # pyright: ignore[reportPossiblyUnbound]


def test_the_tutorial_is_the_quickstart() -> None:
    """The first README block is the end of examples/example1.py."""
    assert QUICKSTART.read_text("utf-8").endswith(readme_python_blocks()[0])


def test_the_output_of_the_augmentation_is_what_it_prints(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """examples/example3.py prints the output that examples/README.md shows."""
    runpy.run_path(str(ROOT / "examples" / "example3.py"), run_name="__main__")
    printed = capsys.readouterr().out
    assert f"```text\n{printed}```" in EXAMPLES_README.read_text("utf-8")


def test_the_example_of_your_own_objectives_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """examples/example2.py runs on its 40 objectives, with run_test shrunk,
    and answers on the suite "objectives"."""

    def small_run_test(
        portfolio: object, feature_extractor: object, predictor: object, **given: object
    ) -> _run_test.RubberRulerResult:
        return _run_test.run(
            portfolio,  # pyright: ignore[reportArgumentType]
            feature_extractor,  # pyright: ignore[reportArgumentType]
            predictor,
            dimension=10,
            train_instance_numbers=None,
            test_instance_numbers=None,
            instances=given["instances"],  # pyright: ignore[reportArgumentType]
            budget_per_dimension=10,
            n_runs=2,
            n_samples_per_dimension=10,
            seed=42,
            n_jobs=1,
            cache_dir=tmp_path,
            function_ids=(1,),
        )

    monkeypatch.setattr(rubberruler, "run_test", small_run_test)
    names = runpy.run_path(str(ROOT / "examples" / "example2.py"))
    result = names["result"]
    assert str(result).startswith("Rubber Ruler Test: use ")
    assert result.config["suite"] == "objectives"  # pyright: ignore[reportAttributeAccessIssue]
    assert len(result.split.test) == 10  # pyright: ignore[reportAttributeAccessIssue]


def test_the_readme_code_for_the_tools_is_from_their_examples() -> None:
    """The blocks of examples/README.md for ModularCMAES, ModDE, pflacco,
    pymoo, and DeepELA are taken from their examples, which the tests run: the same
    imports, and the rest of the block as it is in the example."""
    blocks = 0
    for code in readme_python_blocks(EXAMPLES_README):
        tools = imported_modules(code) & EXAMPLES.keys()
        if not tools:
            continue
        blocks += 1
        (name,) = {EXAMPLES[tool] for tool in tools}
        example = (ROOT / "examples" / name).read_text("utf-8")
        example_imports = {
            ast.dump(node)
            for node in ast.parse(example).body
            if isinstance(node, ast.Import | ast.ImportFrom)
        }
        imports = [
            node
            for node in ast.parse(code).body
            if isinstance(node, ast.Import | ast.ImportFrom)
        ]
        assert {ast.dump(node) for node in imports} <= example_imports, code
        after_imports = max(node.end_lineno or node.lineno for node in imports)
        rest = "".join(code.splitlines(keepends=True)[after_imports:]).strip("\n")
        assert rest in example, rest
    assert blocks == 5


def test_the_links_into_the_readmes_name_their_sections() -> None:
    """Every link to a section, from README.md into examples/README.md and
    within each of the two files, names one of the headings there."""
    readme = README.read_text("utf-8")
    examples = EXAMPLES_README.read_text("utf-8")
    into_examples = re.findall(r"examples/README\.md#([\w-]+)", readme)
    within_readme = re.findall(r"\]\(#([\w-]+)\)", readme)
    within_examples = re.findall(r"\]\(#([\w-]+)\)", examples)
    assert within_examples
    assert set(into_examples) <= anchors(examples), set(into_examples)
    assert set(within_readme) <= anchors(readme), set(within_readme)
    assert set(within_examples) <= anchors(examples), set(within_examples)


def test_the_table_of_examples_lists_each_example_with_its_docstring() -> None:
    """The table at the top of examples/README.md lists example1.py, example2.py, and
    so on, in order. Each row links to the section "Example N: ..." and says what the
    example does in the first line of its docstring."""
    examples = EXAMPLES_README.read_text("utf-8")
    rows = re.findall(
        r"^\| \[(example(\d+)\.py)\]\(#([\w-]+)\) \| ([^|]+?) \|",
        examples,
        re.MULTILINE,
    )
    files = sorted(
        (ROOT / "examples").glob("example[0-9]*.py"),
        key=lambda path: int(path.stem.removeprefix("example")),
    )
    assert [path.name for path in files] == [
        f"example{number}.py" for number in range(1, len(files) + 1)
    ]
    assert [row[0] for row in rows] == [path.name for path in files]
    for (name, number, anchor, sentence), path in zip(rows, files, strict=True):
        assert anchor.startswith(f"example-{number}-"), anchor
        assert anchor in anchors(examples), anchor
        docstring = ast.get_docstring(ast.parse(path.read_text("utf-8")))
        assert docstring is not None, name
        assert docstring.split("\n")[0].replace("``", "`") == sentence, name


@pytest.mark.slow
@pytest.mark.locked
def test_the_readme_card_is_what_the_tutorial_prints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """examples/example1.py, run as it is, prints exactly the card in the README.

    It runs as a script, so that the cache can read the source of the algorithms.
    The numbers can be compared because uv.lock fixes every version. When raising a
    dependency changes them, paste the new card into the README.
    """
    monkeypatch.chdir(tmp_path)
    names = runpy.run_path(str(QUICKSTART), run_name="__main__")
    capsys.readouterr()
    printed = str(names["result"])
    expected = readme_card()
    if printed != expected:
        difference = "\n".join(
            difflib.unified_diff(
                expected.split("\n"),
                printed.split("\n"),
                "card in README.md",
                "card printed now",
                lineterm="",
            )
        )
        pytest.fail(
            "The card in README.md is not what the tutorial prints now. Re-paste the "
            "README card: run `uv run python examples/example1.py` and copy its card "
            "into README.md.\n" + difference,
            pytrace=False,
        )
