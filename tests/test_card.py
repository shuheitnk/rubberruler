import pandas as pd
import pytest

from rubberruler._card import DEFAULT_CONFIG, RunConfig, render, select

EXAMPLE_CARD = """\
Rubber Ruler Test: use RANK-based features
  R^2 on transformed test: rank 0.63 > value 0.42

feature type   R^2 (test)   R^2 (transformed test)   degradation
value                0.71                     0.42          0.29
rank                 0.66                     0.63          0.03

With value-based features, the predictor does not generalize to
monotone transformations of the objective.

3 algorithms, 2400 train, 240 test, 240 transformed test
seed=42"""


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
    assert lines[1] == "  R^2 on transformed test: value 0.70 > rank 0.50"
    assert lines[7:9] == [
        "Value-based features still help the predictor after",
        "monotone transformations of the objective.",
    ]


def test_a_tie_selects_rank_with_an_equals_sign() -> None:
    """Equal R^2 on the transformed test chooses rank."""
    table = make_table((0.9, 0.5), (0.7, 0.5))
    assert select(table) == "rank"
    lines = card(table).split("\n")
    assert lines[1] == "  R^2 on transformed test: rank 0.500 = value 0.500"
    assert (
        lines[9]
        == "The difference is 0.0000 (below 0.01), so treat this answer as weak."
    )


def test_close_values_use_three_digits_and_a_weak_note() -> None:
    """Same at 2 digits gives 3 digits in the table and line 2."""
    lines = card(make_table((0.7123, 0.6312), (0.6655, 0.634))).split("\n")
    assert lines[1] == "  R^2 on transformed test: rank 0.634 > value 0.631"
    assert (
        lines[4] == "value               0.712                    0.631         0.081"
    )
    assert (
        lines[5] == "rank                0.665                    0.634         0.031"
    )
    assert (
        lines[9]
        == "The difference is 0.0028 (below 0.01), so treat this answer as weak."
    )


def test_comparison_uses_unrounded_values() -> None:
    """0.634 vs 0.6344 prints the same but the larger one wins with >."""
    table = make_table((0.5, 0.6344), (0.5, 0.634))
    assert select(table) == "value"
    assert card(table).split("\n")[1] == (
        "  R^2 on transformed test: value 0.634 > rank 0.634"
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
    assert lines[1] == "  R^2 on transformed test: rank 0.05 > value -0.04"
    assert lines[9:12] == [*LOW_R2_NOTE, ""]
    shown = card(make_table((0.3, 0.02), (0.2, 0.0999))).split("\n")  # shows 0.10
    assert shown[9:11] == LOW_R2_NOTE
    assert "weak" not in card(make_table((0.3, 0.02), (0.2, 0.1)))


def test_both_weak_notes_can_appear_together() -> None:
    """A small difference and a small R^2 give both notes, in this order."""
    lines = card(make_table((0.3, 0.051), (0.2, 0.05))).split("\n")
    assert lines[9:12] == [
        "The difference is 0.0010 (below 0.01), so treat this answer as weak.",
        *LOW_R2_NOTE,
    ]


def test_changed_settings_follow_the_seed_and_wrap_at_the_separator() -> None:
    """Seed, then changed settings in run_test's order, wrapped at ", "."""
    config = RunConfig(
        dimension=20,
        train_instance_numbers=(20, 11, 12, 13, 14, 15, 16, 17, 18, 19),
        test_instance_numbers=(1, 3, 5),
        budget_per_dimension=100,
        n_runs=31,
        n_samples_per_dimension=50,
        seed=7,
    )
    lines = card(make_table((0.7, 0.4), (0.6, 0.5)), config).split("\n")
    sizes = "3 algorithms, 2400 train, 240 test, 240 transformed test"
    conditions = lines[lines.index(sizes) + 1 :]
    assert conditions == [
        "seed=7 (default 42), dimension=20 (default 10),",
        "train_instance_numbers=11-20 (default 11-110),",
        "test_instance_numbers=3 numbers (default 1-10),",
        "budget_per_dimension=100 (default 1000)",
    ]
    assert all(len(line) <= 80 for line in conditions)


def test_default_instance_numbers_in_another_order_are_not_shown() -> None:
    """Instance numbers are compared as sorted tuples."""
    config = RunConfig(
        dimension=10,
        train_instance_numbers=tuple(reversed(range(11, 111))),
        test_instance_numbers=tuple(range(1, 11)),
        budget_per_dimension=1000,
        n_runs=31,
        n_samples_per_dimension=50,
        seed=42,
    )
    assert card(make_table((0.7, 0.4), (0.6, 0.5)), config).endswith("\nseed=42")


def test_a_single_instance_number_is_shown_as_itself() -> None:
    """One instance number is shown as "5", not as a range."""
    config = RunConfig(
        dimension=10,
        train_instance_numbers=tuple(range(11, 111)),
        test_instance_numbers=(5,),
        budget_per_dimension=1000,
        n_runs=31,
        n_samples_per_dimension=50,
        seed=42,
    )
    assert card(make_table((0.7, 0.4), (0.6, 0.5)), config).endswith(
        "\nseed=42, test_instance_numbers=5 (default 1-10)"
    )


def test_warnings_come_last() -> None:
    """Warnings are listed under the card."""
    failed = (
        "2 runs failed; their best value so far was used (+inf if none) (see the log)"
    )
    text = card(
        make_table((0.71, 0.42), (0.66, 0.63)),
        warnings=(failed, "12 instances have NaN features"),
    )
    assert text == EXAMPLE_CARD + (
        f"\n\nWarnings:\n  {failed}\n  12 instances have NaN features"
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
        lines[5] == "rank                 0.30                     0.30          0.00"
    )
