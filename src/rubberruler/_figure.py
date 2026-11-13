"""The figure of the answer: R^2 of both feature types as bars, and the row "no
features" as a dashed line in each block.

matplotlib is imported inside the functions, so ``import rubberruler`` does not load
it. The figure is a plain ``matplotlib.figure.Figure`` (no pyplot), and every
setting is changed only inside ``matplotlib.rc_context``: the user's rcParams and
backend stay as they were, and Jupyter shows nothing by itself.
"""

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.typing import RcKeyType

FILE_NAME = "rubber_ruler_test"
FEATURE_TYPES = ("value", "rank")
# The names of the instance sets, in two lines so that they fit their blocks.
BLOCKS = (
    ("r2_test", "test\ninstances"),
    ("r2_transformed_test", "transformed test\ninstances"),
)
EDGE = {"value": "#D55E00", "rank": "#0072B2"}  # Okabe-Ito
FILL = {"value": "#E49659", "rank": "#59A3CD"}  # 35% toward white
INK = "#1B1B1B"
GRID = "#E0E0E0"
# The baseline without features: a gray dashed line across the two bars of a block,
# named in a legend above the axes.
BASELINE = "no features"
BASELINE_COLOR = "#6E6E6E"
BASELINE_DASHES = (0, (4, 2.5))
MINUS = "\u2212"
# A white box behind each value label, so that the lines behind it break there.
LABEL_BOX = {"facecolor": "white", "edgecolor": "none", "boxstyle": "square,pad=0.1"}
SETTINGS: "dict[RcKeyType, Any]" = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "svg.hashsalt": "rubberruler",
}


def save_figure(table: pd.DataFrame, folder: Path) -> tuple[Path, Path]:
    """Write rubber_ruler_test.png and .svg into ``folder``, replacing old ones."""
    import matplotlib

    png = folder / f"{FILE_NAME}.png"
    svg = folder / f"{FILE_NAME}.svg"
    with matplotlib.rc_context(SETTINGS):
        figure = draw(table)
        figure.savefig(png, dpi=200, facecolor="white")
        figure.savefig(svg, facecolor="white", metadata={"Date": None})
    return png, svg


def draw(table: pd.DataFrame) -> "Figure":
    """The bar chart: blocks test instances, transformed test instances; bars
    value, rank; the row "no features", if the table has it, as a dashed line."""
    from matplotlib.figure import Figure

    values = {
        (column, feature_type): float(table.loc[feature_type, column])  # pyright: ignore[reportArgumentType]
        for column, _ in BLOCKS
        for feature_type in FEATURE_TYPES
    }
    baseline = {
        column: float(table.loc[BASELINE, column])  # pyright: ignore[reportArgumentType]
        for column, _ in BLOCKS
        if BASELINE in table.index
    }
    finite = [
        value
        for value in (*values.values(), *baseline.values())
        if math.isfinite(value)
    ]
    top = max(1.0, max(finite, default=0.0) + 0.08) + 0.06
    lowest = min(finite, default=0.0)
    low = max(lowest, -1.0)
    bottom = low - 0.14 * (top - low) if lowest < 0 else 0.0
    gap = 0.02 * (top - bottom)

    figure = Figure(figsize=(8.4, 5.6), facecolor="white")
    axes = figure.add_subplot()
    axes.set_facecolor("white")
    for block, (column, _) in enumerate(BLOCKS):
        for slot, feature_type in enumerate(FEATURE_TYPES):
            value = values[column, feature_type]
            if not math.isfinite(value):
                continue
            x = block * 3 + slot
            height = max(value, -1.0)
            axes.bar(
                x,
                height,
                width=0.74,
                color=FILL[feature_type],
                edgecolor=EDGE[feature_type],
                linewidth=1.8,
                zorder=2,
            )
            above = value >= 0
            axes.text(
                x,
                height + gap if above else height - gap,
                label(value),
                ha="center",
                va="bottom" if above else "top",
                color=EDGE[feature_type],
                fontsize=24,
                fontweight="bold",
                bbox=LABEL_BOX,
                zorder=4,
            )
    for block, (column, _) in enumerate(BLOCKS):
        value = baseline.get(column, math.nan)
        if not math.isfinite(value):
            continue
        height = max(value, -1.0)
        axes.plot(
            [block * 3 - 0.5, block * 3 + 1.5],
            [height, height],
            color=BASELINE_COLOR,
            linestyle=BASELINE_DASHES,
            linewidth=2.4,
            zorder=3.5,
        )
    if axes.lines:
        axes.legend(
            [axes.lines[0]],
            [BASELINE],
            loc="lower right",
            bbox_to_anchor=(1.0, 1.0),
            frameon=False,
            fontsize=18,
            labelcolor=INK,
            handlelength=2.4,
            borderaxespad=0.2,
        )
    for reference in (0.0, 1.0):
        axes.axhline(reference, color=INK, linewidth=1.4, zorder=3)
    axes.set_ylim(bottom, top)
    axes.grid(axis="y", color=GRID, linewidth=1.1)
    axes.set_axisbelow(True)
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.set_xticks(
        [block * 3 + 0.5 for block in range(len(BLOCKS))],
        [name for _, name in BLOCKS],
        fontsize=26,
    )
    axes.set_xticks(
        [block * 3 + slot for block in range(len(BLOCKS)) for slot in (0, 1)],
        [feature_type for _ in BLOCKS for feature_type in FEATURE_TYPES],
        minor=True,
        fontsize=20,
    )
    axes.tick_params(axis="both", which="both", length=0)
    axes.tick_params(axis="x", which="major", pad=38)
    axes.tick_params(axis="y", labelsize=19.2)
    axes.set_ylabel("$R^2$", fontsize=24)
    figure.tight_layout()
    return figure


def label(value: float) -> str:
    """Two decimals, with U+2212 for a negative number (-0.00 is shown as 0.00)."""
    text = f"{value:.2f}"
    if not text.startswith("-"):
        return text
    return text[1:] if float(text) == 0 else MINUS + text[1:]
