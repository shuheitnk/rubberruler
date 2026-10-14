import logging
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest
from helpers import tiny_run
from matplotlib.colors import to_rgba

from rubberruler._figure import draw, save_figure


def make_table(value: tuple[float, float], rank: tuple[float, float]) -> pd.DataFrame:
    rows = {
        "value": [value[0], value[1], value[0] - value[1]],
        "rank": [rank[0], rank[1], rank[0] - rank[1]],
    }
    return pd.DataFrame.from_dict(
        rows, orient="index", columns=["r2_test", "r2_transformed_test", "degradation"]
    )


def test_run_test_writes_the_png_and_the_svg_into_cache_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """rubber_ruler_test.png and .svg appear in cache_dir, with one line."""
    result = tiny_run(cache_dir=tmp_path)
    png = tmp_path / "rubber_ruler_test.png"
    svg = tmp_path / "rubber_ruler_test.svg"
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert svg.read_text("utf-8").startswith("<?xml")
    folder = os.path.join(tmp_path, "")
    assert capsys.readouterr().err.endswith(
        f"rubberruler: figure saved to {folder}rubber_ruler_test.png and .svg\n"
    )
    assert "figure" not in str(result)


def test_without_cache_dir_no_figure_is_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With cache_dir=None, nothing is written and nothing is said."""
    monkeypatch.chdir(tmp_path)
    tiny_run(cache_dir=None)
    assert list(tmp_path.iterdir()) == []
    assert "figure saved" not in capsys.readouterr().err


def test_the_svg_keeps_the_values_names_and_colors_as_text(tmp_path: Path) -> None:
    """The four values, the block names (each in two lines), and both colors are
    in the SVG."""
    _, svg = save_figure(make_table((0.4119, 0.2318), (0.3018, 0.31)), tmp_path)
    text = svg.read_text("utf-8")
    labels = (
        "0.41",
        "0.23",
        "0.30",
        "0.31",
        ">test<",
        ">transformed test<",
        ">instances<",
    )
    for label in labels:
        assert label in text
    assert "#d55e00" in text.lower()
    assert "#0072b2" in text.lower()


def test_negative_values_use_the_minus_sign_and_bars_stop_at_minus_1(
    tmp_path: Path,
) -> None:
    """U+2212 for negatives; a bar below minus 1 stops there, not its label."""
    table = make_table((0.52, -3.42), (-0.25, 0.1))
    axes = draw(table).axes[0]
    heights = sorted(round(patch.get_height(), 9) for patch in axes.patches)  # pyright: ignore[reportAttributeAccessIssue]
    assert heights == [-1.0, -0.25, 0.1, 0.52]
    texts = {text.get_text() for text in axes.texts}
    assert texts == {"0.52", "\u22123.42", "\u22120.25", "0.10"}
    _, svg = save_figure(table, tmp_path)
    content = svg.read_text("utf-8")
    assert "\u22123.42" in content
    assert "-3.42" not in content


def test_the_y_range_follows_the_values() -> None:
    """Top max(1, max + 0.08) + 0.06; bottom 0, or low minus 14%."""
    positive = draw(make_table((0.97, 0.88), (0.95, 0.96))).axes[0]
    assert positive.get_ylim() == pytest.approx((0.0, 1.11))
    negative = draw(make_table((0.52, -3.42), (-0.25, 0.1))).axes[0]
    top = 1.06
    assert negative.get_ylim() == pytest.approx((-1.0 - 0.14 * (top + 1.0), top))


def test_importing_rubberruler_does_not_import_matplotlib() -> None:
    """matplotlib is loaded only when a figure is drawn."""
    code = "import sys, rubberruler; print('matplotlib' in sys.modules)"
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert output.strip() == "False"


def test_drawing_leaves_the_users_settings_alone(tmp_path: Path) -> None:
    """rcParams and the backend are the same after the figure is written."""
    before = dict(matplotlib.rcParams)
    backend = matplotlib.get_backend()
    save_figure(make_table((0.5, 0.4), (0.45, 0.47)), tmp_path)
    assert dict(matplotlib.rcParams) == before
    assert matplotlib.get_backend() == backend


def test_a_figure_that_cannot_be_written_is_only_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """run_test still returns the result, and the logger warns."""
    (tmp_path / "rubber_ruler_test.png").mkdir()
    with caplog.at_level(logging.WARNING, logger="rubberruler"):
        result = tiny_run(cache_dir=tmp_path)
    assert result.selected_feature_type in ("value", "rank")
    folder = os.path.join(tmp_path, "")
    assert any(
        message.startswith(f"could not save the figure in {folder}: ")
        for message in caplog.messages
    )
    assert "Warnings:" not in str(result)


def test_labels_sit_on_a_white_box_in_front_of_the_lines() -> None:
    """Near R^2 = 1 the reference line breaks behind each value label."""
    axes = draw(make_table((0.95, 0.93), (0.97, 0.95))).axes[0]
    assert len(axes.texts) == 4
    front = max(line.get_zorder() for line in axes.lines)
    for text in axes.texts:
        box = text.get_bbox_patch()
        assert box is not None
        assert box.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
        assert to_rgba(box.get_edgecolor())[3] == 0.0
        style = box.get_boxstyle()
        assert type(style).__name__ == "Square"
        assert style.pad == 0.1  # pyright: ignore[reportAttributeAccessIssue]
        assert text.get_zorder() > front
