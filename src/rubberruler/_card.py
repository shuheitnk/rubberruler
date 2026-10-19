"""The card that print(result) shows, and the panel that rich draws of it."""

import math
import re
import textwrap
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TypedDict

import pandas as pd
from rich import box
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from rubberruler._features import FeatureType
from rubberruler._instances import DEFAULT_INSTANCE_NUMBERS, Suite

WIDTH = 80
SEPARATOR = ", "
# The header of the table, in two lines, so that the names of the instance sets
# fit in 80 characters without abbreviation.
HEADER = (
    ("", "feature type"),
    ("R^2 on", "test instances"),
    ("R^2 on", "transformed test instances"),
    ("", "degradation"),
)
COLUMN_WIDTHS = (12, 14, 26, 11)
COLUMN_GAP = "   "
# Joins the words of a part in parentheses while a warning is wrapped.
NO_BREAK = "\x00"
WEAK_DIFFERENCE = 0.01
WEAK_R2 = 0.1
WEAK_R2_NOTE = (
    "Both R^2 values are below 0.1, so this answer is weak.",
    "Check the extractor and the predictor first.",
)
MEANING: dict[FeatureType, tuple[str, str]] = {
    "rank": (
        "With value-based features, the predictor does not generalize to",
        "strictly increasing transformations of the objective.",
    ),
    "value": (
        "Value-based features still help the predictor after",
        "strictly increasing transformations of the objective.",
    ),
}


class RunConfig(TypedDict):
    """The eight settings that decide the result (``result.config``)."""

    suite: Suite
    dimension: int
    train_instance_numbers: tuple[int, ...]
    test_instance_numbers: tuple[int, ...]
    budget_per_dimension: int
    n_runs: int
    n_samples_per_dimension: int
    seed: int


DEFAULT_CONFIG = RunConfig(
    suite="bbob",
    dimension=10,
    train_instance_numbers=tuple(DEFAULT_INSTANCE_NUMBERS["bbob"][0]),
    test_instance_numbers=tuple(DEFAULT_INSTANCE_NUMBERS["bbob"][1]),
    budget_per_dimension=1000,
    n_runs=31,
    n_samples_per_dimension=50,
    seed=42,
)


def select(table: pd.DataFrame) -> FeatureType:
    """tau*: the larger R^2 on the transformed test instances wins; rank wins
    ties."""
    value = float(table.loc["value", "r2_transformed_test"])  # pyright: ignore[reportArgumentType]
    rank = float(table.loc["rank", "r2_transformed_test"])  # pyright: ignore[reportArgumentType]
    return "rank" if rank >= value else "value"


@dataclass(frozen=True)
class Card:
    """The parts of the card: the answer, the evidence, the meaning, and the
    conditions. The text of print(result) and the panel of rich show the same parts."""

    selected: FeatureType
    evidence: str
    rows: tuple[tuple[str, str, str, str], ...]
    meaning: tuple[str, ...]
    cautions: tuple[str, ...]
    conditions: tuple[str, ...]
    warnings: tuple[str, ...]

    def answer(self) -> str:
        return f"use {self.selected.upper()}-based features"

    def text(self) -> str:
        """The card that print(result) shows."""
        lines = [
            f"Rubber Ruler Test: {self.answer()}",
            f"  {self.evidence}",
            "",
            _table_line(names[0] for names in HEADER),
            _table_line(names[1] for names in HEADER),
            *(_table_line(row) for row in self.rows),
            "",
            *self.meaning,
            *self.cautions,
            "",
            *self.conditions,
        ]
        if self.warnings:
            lines += ["", *self.warning_lines()]
        return "\n".join(lines)

    def warning_lines(self) -> list[str]:
        """ "Warnings:" and each warning, wrapped at 80 with the rest indented. A
        part in parentheses stays on one line."""
        lines = ["Warnings:"]
        for warning in self.warnings:
            wrapped = textwrap.wrap(
                re.sub(r"\([^()]*\)", _keep_together, warning),
                WIDTH,
                initial_indent="  ",
                subsequent_indent="    ",
                break_long_words=False,
                break_on_hyphens=False,
            )
            lines += [line.replace(NO_BREAK, " ") for line in wrapped]
        return lines

    def panel(self) -> Panel:
        """The card as rich draws it, such as with rich.print(result)."""
        table = Table(
            *("\n".join(names) if names[0] else names[1] for names in HEADER),
            box=box.SIMPLE_HEAD,
            show_edge=False,
            pad_edge=False,
        )
        for column in table.columns[1:]:
            column.justify = "right"
        for row in self.rows:
            table.add_row(*row, style="bold" if row[0] == self.selected else None)
        parts: list[RenderableType] = [
            Text(self.answer(), style="bold green"),
            Text(self.evidence),
            Text(""),
            table,
            Text(""),
            Text("\n".join(self.meaning)),
        ]
        if self.cautions:
            parts.append(Text("\n".join(self.cautions), style="yellow"))
        parts += [Text(""), Text("\n".join(self.conditions), style="dim")]
        if self.warnings:
            lines = self.warning_lines()
            parts += [Text(""), Text("\n".join(lines), style="yellow")]
        return Panel(
            Group(*parts), title="Rubber Ruler Test", title_align="left", expand=False
        )


def card(
    table: pd.DataFrame,
    selected: FeatureType,
    *,
    n_algorithms: int,
    set_sizes: tuple[int, int, int],
    config: RunConfig,
    warnings: Sequence[str],
) -> Card:
    """The parts of the card for the table of R^2 and the run."""
    other: FeatureType = "value" if selected == "rank" else "rank"
    chosen = float(table.loc[selected, "r2_transformed_test"])  # pyright: ignore[reportArgumentType]
    rejected = float(table.loc[other, "r2_transformed_test"])  # pyright: ignore[reportArgumentType]
    digits = 3 if number(chosen, 2) == number(rejected, 2) else 2
    relation = ">" if chosen != rejected else "="
    rows: list[tuple[str, str, str, str]] = []
    for feature_type in ("value", "rank"):
        on_test, on_transformed, drop = (
            number(float(table.loc[feature_type, column]), digits)  # pyright: ignore[reportArgumentType]
            for column in ("r2_test", "r2_transformed_test", "degradation")
        )
        rows.append((feature_type, on_test, on_transformed, drop))
    cautions: list[str] = []
    difference = abs(chosen - rejected)
    if difference < WEAK_DIFFERENCE:
        cautions.append(
            f"The difference is {difference:.4f} (below 0.01), so treat this answer "
            "as weak."
        )
    # chosen is the larger R^2 on the transformed test instances (or equal to the
    # other).
    if chosen < WEAK_R2:
        cautions += WEAK_R2_NOTE
    train, test, transformed_test = set_sizes
    return Card(
        selected=selected,
        evidence=f"R^2 on the transformed test instances: {selected} "
        f"{number(chosen, digits)} {relation} {other} {number(rejected, digits)}",
        rows=tuple(rows),
        meaning=MEANING[selected],
        cautions=tuple(cautions),
        conditions=(
            *wrap(
                [
                    f"{n_algorithms} algorithms",
                    f"{train} training instances",
                    f"{test} test instances",
                    f"{transformed_test} transformed test instances",
                ]
            ),
            *wrap(settings(config)),
        ),
        warnings=tuple(warnings),
    )


def render(
    table: pd.DataFrame,
    selected: FeatureType,
    *,
    n_algorithms: int,
    set_sizes: tuple[int, int, int],
    config: RunConfig,
    warnings: Sequence[str],
) -> str:
    """The card that print(result) shows."""
    return card(
        table,
        selected,
        n_algorithms=n_algorithms,
        set_sizes=set_sizes,
        config=config,
        warnings=warnings,
    ).text()


def settings(config: RunConfig) -> list[str]:
    """seed, then each changed setting, in the order of run_test's parameters.

    The instance numbers are compared with the defaults of the suite.
    """
    seed = config["seed"]
    items = [
        f"seed={seed}"
        if seed == DEFAULT_CONFIG["seed"]
        else f"seed={seed} (default {DEFAULT_CONFIG['seed']})"
    ]
    suite = config["suite"]
    if suite != DEFAULT_CONFIG["suite"]:
        items.append(f"suite={suite} (default {DEFAULT_CONFIG['suite']})")
    if config["dimension"] != DEFAULT_CONFIG["dimension"]:
        items.append(
            _changed("dimension", config["dimension"], DEFAULT_CONFIG["dimension"])
        )
    defaults = DEFAULT_INSTANCE_NUMBERS[suite]
    for name, default_range in zip(
        ("train_instance_numbers", "test_instance_numbers"), defaults, strict=True
    ):
        numbers = tuple(sorted(config[name]))
        default = tuple(default_range)
        if numbers != default:
            items.append(f"{name}={_numbers(numbers)} (default {_numbers(default)})")
    for name in ("budget_per_dimension", "n_runs", "n_samples_per_dimension"):
        if config[name] != DEFAULT_CONFIG[name]:
            items.append(_changed(name, config[name], DEFAULT_CONFIG[name]))
    return items


def wrap(items: Sequence[str]) -> list[str]:
    """Join the items with ", "; wrap before an item past 80, keeping "," at the end."""
    lines = [items[0]]
    for item in items[1:]:
        # Below WIDTH, so that the "," of a line that then ends still fits.
        if len(lines[-1]) + len(SEPARATOR) + len(item) < WIDTH:
            lines[-1] += SEPARATOR + item
        else:
            lines[-1] += SEPARATOR.rstrip()
            lines.append(item)
    return lines


def _keep_together(match: re.Match[str]) -> str:
    return match.group().replace(" ", NO_BREAK)


def _table_line(cells: Iterable[str]) -> str:
    """One line of the table: the feature type on the left, the numbers right."""
    first, *rest = cells
    aligned = [first.ljust(COLUMN_WIDTHS[0])]
    aligned += [
        cell.rjust(width) for cell, width in zip(rest, COLUMN_WIDTHS[1:], strict=True)
    ]
    return COLUMN_GAP.join(aligned).rstrip()


def number(value: float, digits: int) -> str:
    """The value with ``digits`` decimals; -0.00 is shown as 0.00."""
    if math.isnan(value):
        return "nan"
    text = f"{value:.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def _changed(name: str, value: int, default: int) -> str:
    return f"{name}={value} (default {default})"


def _numbers(numbers: tuple[int, ...]) -> str:
    """ "11-20" for consecutive numbers, "5" for one, "12 numbers" otherwise."""
    if len(numbers) == 1:
        return str(numbers[0])
    if numbers == tuple(range(numbers[0], numbers[-1] + 1)):
        return f"{numbers[0]}-{numbers[-1]}"
    return f"{len(numbers)} numbers"
