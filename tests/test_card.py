import io

import pandas as pd
import pytest
from rich.console import Console

from rubberruler import _card
from rubberruler._card import DEFAULT_CONFIG, RunConfig, render, select

EXAMPLE_CARD = """\
Rubber Ruler Test: use RANK-based features
  R^2 on the transformed test instances: rank 0.63 > value 0.42

                       R^2 on                       R^2 on
feature type   test instances   transformed test instances   degradation
value                    0.71                         0.42          0.29
rank                     0.66                         0.63          0.03

With value-based features, the predictor does not generalize to
strictly increasing transformations of the objective.

3 algorithms, 2400 training instances, 240 test instances,
240 transformed test instances
seed=42, suite=bbob"""


def make_table(value: tuple[float, float], rank: tuple[float, float]) -> pd.DataFrame:
    rows = {
        "value": [value[0], value[1], value[0] - value[1]],
        "rank": [rank[0], rank[1], rank[0] - rank[1]],
    }
    return pd.DataFrame.from_dict(
        rows, orient="index", columns=["r2_test", "r2_transformed_test", "degradation"]
    )


def card(
    table: pd.DataFrame,
    config: RunConfig = DEFAULT_CONFIG,
    warnings: tuple[str, ...] = (),
) -> str:
    return render(
        table,
        select(table),
        n_algorithms=3,
        set_sizes=(2400, 240, 240),
        config=config,
        warnings=warnings,
    )


def test_the_card_matches_the_example() -> None:
    """The card shows the answer, evidence, meaning, and conditions."""
    assert card(make_table((0.71, 0.42), (0.66, 0.63))) == EXAMPLE_CARD


def test_value_wins_with_its_own_meaning() -> None:
    """value is chosen when its transformed R^2 is larger."""
    lines = card(make_table((0.8, 0.7), (0.6, 0.5))).split("\n")
    assert lines[0] == "Rubber Ruler Test: use VALUE-based features"
    assert lines[1] == "  R^2 on the transformed test instances: value 0.70 > rank 0.50"
    assert lines[8:10] == [
        "Value-based features still help the predictor after",
        "strictly increasing transformations of the objective.",
    ]


def test_a_tie_selects_rank_with_an_equals_sign() -> None:
    """Equal R^2 on the transformed test chooses rank."""
    table = make_table((0.9, 0.5), (0.7, 0.5))
    assert select(table) == "rank"
    lines = card(table).split("\n")
    assert (
        lines[1] == "  R^2 on the transformed test instances: rank 0.500 = value 0.500"
    )
    assert (
        lines[10]
        == "The difference is 0.0000 (below 0.01), so treat this answer as weak."
    )


def test_close_values_use_three_digits_and_a_weak_note() -> None:
    """Same at 2 digits gives 3 digits in the table and line 2."""
    lines = card(make_table((0.7123, 0.6312), (0.6655, 0.634))).split("\n")
    assert (
        lines[1] == "  R^2 on the transformed test instances: rank 0.634 > value 0.631"
    )
    assert (
        lines[5]
        == "value                   0.712                        0.631         0.081"
    )
    assert (
        lines[6]
        == "rank                    0.665                        0.634         0.031"
    )
    assert (
        lines[10]
        == "The difference is 0.0028 (below 0.01), so treat this answer as weak."
    )


def test_comparison_uses_unrounded_values() -> None:
    """0.634 vs 0.6344 prints the same but the larger one wins with >."""
    table = make_table((0.5, 0.6344), (0.5, 0.634))
    assert select(table) == "value"
    assert card(table).split("\n")[1] == (
        "  R^2 on the transformed test instances: value 0.634 > rank 0.634"
    )


def test_a_difference_of_at_least_001_is_not_weak() -> None:
    """The weak note appears only below 0.01 (unrounded values)."""
    assert "weak" not in card(make_table((0.5, 0.40), (0.5, 0.42)))
    assert "weak" in card(make_table((0.5, 0.40), (0.5, 0.41)))  # 0.00999...


LOW_R2_NOTE = [
    "Both R^2 values are below 0.1, so this answer is weak.",
    "Check the extractor and the predictor first.",
]


def test_a_small_r2_for_both_feature_types_is_weak() -> None:
    """The larger R^2 on the transformed test below 0.1 adds a weak note of
    two lines."""
    lines = card(make_table((0.3, -0.04), (0.2, 0.05))).split("\n")
    assert (
        lines[1] == "  R^2 on the transformed test instances: rank 0.05 > value -0.04"
    )
    assert lines[10:13] == [*LOW_R2_NOTE, ""]
    shown = card(make_table((0.3, 0.02), (0.2, 0.0999))).split("\n")  # shows 0.10
    assert shown[10:12] == LOW_R2_NOTE
    assert "weak" not in card(make_table((0.3, 0.02), (0.2, 0.1)))


def test_both_weak_notes_can_appear_together() -> None:
    """A small difference and a small R^2 give both notes, in this order."""
    lines = card(make_table((0.3, 0.051), (0.2, 0.05))).split("\n")
    assert lines[10:13] == [
        "The difference is 0.0010 (below 0.01), so treat this answer as weak.",
        *LOW_R2_NOTE,
    ]


def test_changed_settings_follow_the_seed_and_wrap_at_the_separator() -> None:
    """Seed, then changed settings in run_test's order, wrapped at ", "."""
    config = RunConfig(
        suite="bbob",
        dimension=20,
        train_instance_numbers=(20, 11, 12, 13, 14, 15, 16, 17, 18, 19),
        test_instance_numbers=(1, 3, 5),
        budget_per_dimension=100,
        n_runs=31,
        n_samples_per_dimension=50,
        seed=7,
        instances=None,
    )
    lines = card(make_table((0.7, 0.4), (0.6, 0.5)), config).split("\n")
    sizes = "240 transformed test instances"
    conditions = lines[lines.index(sizes) + 1 :]
    assert conditions == [
        "seed=7 (default 42), suite=bbob, dimension=20 (default 10),",
        "train_instance_numbers=11-20 (default 11-110),",
        "test_instance_numbers=3 numbers (default 1-10),",
        "budget_per_dimension=100 (default 1000)",
    ]
    assert all(len(line) <= 80 for line in conditions)


def test_default_instance_numbers_in_another_order_are_not_shown() -> None:
    """Instance numbers are compared as sorted tuples."""
    config = RunConfig(
        suite="bbob",
        dimension=10,
        train_instance_numbers=tuple(reversed(range(11, 111))),
        test_instance_numbers=tuple(range(1, 11)),
        budget_per_dimension=1000,
        n_runs=31,
        n_samples_per_dimension=50,
        seed=42,
        instances=None,
    )
    assert card(make_table((0.7, 0.4), (0.6, 0.5)), config).endswith(
        "\nseed=42, suite=bbob"
    )


def test_a_single_instance_number_is_shown_as_itself() -> None:
    """One instance number is shown as "5", not as a range."""
    config = RunConfig(
        suite="bbob",
        dimension=10,
        train_instance_numbers=tuple(range(11, 111)),
        test_instance_numbers=(5,),
        budget_per_dimension=1000,
        n_runs=31,
        n_samples_per_dimension=50,
        seed=42,
        instances=None,
    )
    assert card(make_table((0.7, 0.4), (0.6, 0.5)), config).endswith(
        "\nseed=42, suite=bbob, test_instance_numbers=5 (default 1-10)"
    )


def test_warnings_come_last() -> None:
    """Warnings are listed under the card."""
    failed = (
        "2 runs failed; their best value so far was used "
        "(+inf if there was none; see the log)"
    )
    text = card(
        make_table((0.71, 0.42), (0.66, 0.63)),
        warnings=(failed, "12 instances have NaN features"),
    )
    assert text == EXAMPLE_CARD + (
        "\n\nWarnings:"
        "\n  2 runs failed; their best value so far was used"
        "\n    (+inf if there was none; see the log)"
        "\n  12 instances have NaN features"
    )


@pytest.mark.parametrize(
    ("value", "rank"),
    [
        ((0.71, 0.42), (0.66, 0.63)),
        ((-12.3456, -3.0), (0.1, -0.004)),
        ((0.5, 0.5), (0.5, 0.5)),
    ],
)
def test_every_line_fits_in_80_characters(
    value: tuple[float, float], rank: tuple[float, float]
) -> None:
    """Each line is at most 80 characters."""
    assert all(len(line) <= 80 for line in card(make_table(value, rank)).split("\n"))


def test_negative_zero_is_shown_as_zero() -> None:
    """A degradation of -0.0005 at 2 digits is 0.00, not -0.00."""
    lines = card(make_table((0.6, 0.5), (0.2995, 0.3))).split("\n")
    assert (
        lines[6]
        == "rank                     0.30                         0.30          0.00"
    )


def weak_card_with_a_warning() -> _card.Card:
    table = make_table((0.08, 0.05), (0.07, 0.055))
    return _card.card(
        table,
        select(table),
        n_algorithms=3,
        set_sizes=(2400, 240, 240),
        config=DEFAULT_CONFIG,
        warnings=(
            "1 run failed; its best value so far was used "
            "(+inf if there was none; see the log)",
        ),
    )


def test_rich_draws_the_parts_of_the_card_in_a_panel() -> None:
    """rich draws what the card says: the answer, the evidence, the table, the
    meaning, the cautions, the conditions, and the warnings."""
    parts = weak_card_with_a_warning()
    screen = io.StringIO()
    Console(file=screen, width=100, color_system=None).print(parts.panel())
    drawn = screen.getvalue()
    lines = parts.text().split("\n")
    assert lines[0] == "Rubber Ruler Test: use RANK-based features"
    assert "Rubber Ruler Test" in drawn
    assert "use RANK-based features" in drawn
    # The header and the rows of the table are spaced differently in the panel.
    assert all(line.strip() in drawn for line in lines[1:3] + lines[7:])
    assert all(cell in drawn for row in parts.rows for cell in row)
    assert "Warnings:" in drawn


def test_the_panel_is_ascii_in_a_console_without_unicode() -> None:
    """In cp932, rich draws the panel and the table with ASCII lines."""
    data = io.BytesIO()
    screen = io.TextIOWrapper(data, encoding="cp932", errors="strict")
    console = Console(file=screen, width=100, color_system=None)
    console.print(weak_card_with_a_warning().panel())
    screen.flush()
    assert data.getvalue().decode("cp932").isascii()


def test_the_panel_fits_an_80_column_terminal_without_breaking_lines() -> None:
    """Inside the panel, the conditions and the warnings are wrapped at 76,
    so that rich does not break them again in a terminal of 80 columns."""
    table = make_table((0.41, 0.23), (0.30, 0.31))
    parts = _card.card(
        table,
        select(table),
        n_algorithms=3,
        set_sizes=(100, 10, 10),
        config=RunConfig(
            suite="ma-bbob",
            dimension=20,
            train_instance_numbers=tuple(range(1, 101)),
            test_instance_numbers=tuple(range(2401, 2411)),
            budget_per_dimension=100,
            n_runs=5,
            n_samples_per_dimension=50,
            seed=7,
            instances=None,
        ),
        warnings=(
            "nelder-mead: cached by name and source only "
            "(some values could not be hashed)",
        ),
    )
    text = parts.text().split("\n")
    # The card of print(result) has lines longer than 76, which the panel rewraps.
    assert max(len(line) for line in text) > 76
    screen = io.StringIO()
    Console(file=screen, width=80, color_system=None).print(parts.panel())
    drawn = [line.strip("│| ").rstrip() for line in screen.getvalue().split("\n")]
    assert all(len(line) <= 80 for line in screen.getvalue().split("\n"))
    for line in [*parts.conditions(76), *parts.warning_lines(76)]:
        assert len(line) <= 76
        assert line.strip() in drawn


def test_a_caution_when_both_r2_values_are_below_the_row_without_features() -> None:
    """The card shows the row "no features", and one line of
    caution when both R^2 on the transformed test instances are below it."""
    table = make_table((0.6, 0.5), (0.55, 0.52))
    table.loc["no features"] = [0.58, 0.56, 0.02]
    lines = card(table).split("\n")
    assert lines[7] == (
        "no features              0.58                         0.56          0.02"
    )
    assert (
        'Both R^2 values are below the row "no features"; improve the features '
        "first." in lines
    )
    assert all(len(line) <= 80 for line in lines)
    table.loc["no features"] = [0.58, 0.50, 0.08]
    assert not any("no features" in line for line in card(table).split("\n")[8:])


def test_below_the_row_without_features_the_meaning_only_compares() -> None:
    """When both R^2 values on the transformed test instances are
    below the row "no features", the meaning compares the feature types instead of
    saying that one helps; when only the other feature type is below it, the meaning
    of the selected feature type stays."""
    second = "strictly increasing transformations of the objective."
    table = make_table((0.48, 0.412), (0.40, 0.405))
    table.loc["no features"] = [0.50, 0.50, 0.0]
    assert card(table).split("\n")[9:11] == [
        "Value-based features generalize better than rank-based features to",
        second,
    ]
    table = make_table((0.6, 0.3), (0.5, 0.4))
    table.loc["no features"] = [0.5, 0.45, 0.05]
    assert card(table).split("\n")[9:11] == [
        "Rank-based features generalize better than value-based features to",
        second,
    ]
    table = make_table((0.6, 0.4), (0.5, 0.4))
    table.loc["no features"] = [0.5, 0.45, 0.05]
    assert card(table).split("\n")[9:11] == [
        "Rank-based features generalize as well as value-based features to",
        second,
    ]
    # Only rank below the row: the meaning of value stays, without a caution.
    table = make_table((0.6, 0.5), (0.5, 0.4))
    table.loc["no features"] = [0.5, 0.45, 0.05]
    lines = card(table).split("\n")
    assert lines[9:12] == [
        "Value-based features still help the predictor after",
        second,
        "",
    ]


def test_few_test_instances_add_a_caution() -> None:
    """Below 10 test instances, one line says that the answer may
    change with other test instances; at 10 or more, the line is absent."""
    table = make_table((0.6, 0.5), (0.55, 0.52))
    sizes = {"n_algorithms": 3, "config": DEFAULT_CONFIG, "warnings": ()}
    few = render(table, select(table), set_sizes=(27, 9, 9), **sizes)  # pyright: ignore[reportArgumentType]
    one = render(table, select(table), set_sizes=(3, 1, 1), **sizes)  # pyright: ignore[reportArgumentType]
    enough = render(table, select(table), set_sizes=(30, 10, 10), **sizes)  # pyright: ignore[reportArgumentType]
    line = "Only 9 test instances: the answer may change with other test instances."
    assert few.split("\n")[10] == line
    assert "Only 1 test instance: the answer may change" in one
    assert "Only" not in enough
