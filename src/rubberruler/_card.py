"""The card that print(result) shows."""

import math
from collections.abc import Sequence
from typing import TypedDict

import pandas as pd

from rubberruler._features import FeatureType

WIDTH = 80
SEPARATOR = ", "
WEAK_DIFFERENCE = 0.01
WEAK_R2 = 0.1
WEAK_R2_NOTE = (
    "Both R^2 values are below 0.1, so this answer is weak.",
    "Check the extractor and the predictor first.",
)
MEANING: dict[FeatureType, tuple[str, str]] = {
    "rank": (
        "With value-based features, the predictor does not generalize to",
        "monotone transformations of the objective.",
    ),
    "value": (
        "Value-based features still help the predictor after",
        "monotone transformations of the objective.",
    ),
}


class RunConfig(TypedDict):
    """The seven settings that decide the result (``result.config``)."""

    dimension: int
    train_instance_numbers: tuple[int, ...]
    test_instance_numbers: tuple[int, ...]
    budget_per_dimension: int
    n_runs: int
    n_samples_per_dimension: int
    seed: int


DEFAULT_CONFIG = RunConfig(
    dimension=10,
    train_instance_numbers=tuple(range(11, 111)),
    test_instance_numbers=tuple(range(1, 11)),
    budget_per_dimension=1000,
    n_runs=31,
    n_samples_per_dimension=50,
    seed=42,
)


def select(table: pd.DataFrame) -> FeatureType:
    """tau*: the larger R^2 on the transformed test wins; rank wins ties."""
    value = float(table.loc["value", "r2_transformed_test"])  # pyright: ignore[reportArgumentType]
    rank = float(table.loc["rank", "r2_transformed_test"])  # pyright: ignore[reportArgumentType]
    return "rank" if rank >= value else "value"


def render(
    table: pd.DataFrame,
    selected: FeatureType,
    *,
    n_algorithms: int,
    set_sizes: tuple[int, int, int],
    config: RunConfig,
    warnings: Sequence[str],
) -> str:
    """The card: the answer, the evidence, the meaning, and the conditions."""
    other: FeatureType = "value" if selected == "rank" else "rank"
    chosen = float(table.loc[selected, "r2_transformed_test"])  # pyright: ignore[reportArgumentType]
    rejected = float(table.loc[other, "r2_transformed_test"])  # pyright: ignore[reportArgumentType]
    digits = 3 if number(chosen, 2) == number(rejected, 2) else 2
    relation = ">" if chosen != rejected else "="
    lines = [
        f"Rubber Ruler Test: use {selected.upper()}-based features",
        f"  R^2 on transformed test: {selected} {number(chosen, digits)} {relation} "
        f"{other} {number(rejected, digits)}",
        "",
        "feature type   R^2 (test)   R^2 (transformed test)   degradation",
    ]
    for feature_type in ("value", "rank"):
        cells = [
            number(float(table.loc[feature_type, column]), digits)  # pyright: ignore[reportArgumentType]
            for column in ("r2_test", "r2_transformed_test", "degradation")
        ]
        lines.append(
            f"{feature_type:<12}   {cells[0]:>10}   {cells[1]:>22}   {cells[2]:>11}"
        )
    lines += ["", *MEANING[selected]]
    difference = abs(chosen - rejected)
    if difference < WEAK_DIFFERENCE:
        lines.append(
            f"The difference is {difference:.4f} (below 0.01), so treat this answer "
            "as weak."
        )
    # chosen is the larger R^2 on the transformed test (or equal to the other).
    if chosen < WEAK_R2:
        lines += WEAK_R2_NOTE
    train, test, transformed_test = set_sizes
    lines += [
        "",
        f"{n_algorithms} algorithms{SEPARATOR}{train} train{SEPARATOR}{test} test"
        f"{SEPARATOR}{transformed_test} transformed test",
        *wrap(settings(config)),
    ]
    if warnings:
        lines += ["", "Warnings:", *(f"  {line}" for line in warnings)]
    return "\n".join(lines)


def settings(config: RunConfig) -> list[str]:
    """seed, then each changed setting, in the order of run_test's parameters."""
    seed = config["seed"]
    items = [
        f"seed={seed}"
        if seed == DEFAULT_CONFIG["seed"]
        else f"seed={seed} (default {DEFAULT_CONFIG['seed']})"
    ]
    if config["dimension"] != DEFAULT_CONFIG["dimension"]:
        items.append(
            _changed("dimension", config["dimension"], DEFAULT_CONFIG["dimension"])
        )
    for name in ("train_instance_numbers", "test_instance_numbers"):
        numbers = tuple(sorted(config[name]))
        default = DEFAULT_CONFIG[name]
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
