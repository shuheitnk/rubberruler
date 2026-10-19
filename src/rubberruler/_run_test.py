"""run_test: value-based or rank-based features?"""

import logging
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from rich.panel import Panel

from rubberruler._cache import check_cache_dir
from rubberruler._card import Card, RunConfig, card, select
from rubberruler._console import progress, say, stage
from rubberruler._errors import error
from rubberruler._features import (
    FEATURE_TYPES,
    MA_BBOB_PRECHECK_INSTANCES,
    FeatureExtractor,
    FeatureTable,
    check_n_samples_per_dimension,
    feature_table,
    feature_warnings,
    precheck_extractor,
)
from rubberruler._figure import FILE_NAME as FIGURE_FILE_NAME
from rubberruler._figure import save_figure
from rubberruler._instances import (
    DEFAULT_INSTANCE_NUMBERS,
    FUNCTION_IDS,
    InstanceSplit,
    Suite,
    check_dimension,
    check_instance_numbers,
    check_seed,
    check_suite,
    make_split,
    transformation_of,
)
from rubberruler._performance import (
    Measurement,
    check_budget_per_dimension,
    check_n_jobs,
    check_n_runs,
    check_portfolio,
    plan_line,
    plan_runs,
    precheck_algorithms,
    run_plan,
)
from rubberruler._prediction import (
    RANDOM_STATE_WARNING,
    feature_matrix,
    fit,
    has_unset_random_state,
    performance_matrix,
    precheck_predictor,
    score,
)
from rubberruler._problem import Algorithm

logger = logging.getLogger("rubberruler")

TABLE_COLUMNS = ["r2_test", "r2_transformed_test", "degradation"]


@dataclass(frozen=True, eq=False, repr=False)
class RubberRulerResult:
    """The answer of the Rubber Ruler Test and what it is based on.

    ``print(result)`` shows the card; its first line is the answer. With rich,
    ``rich.print(result)`` draws the card as a panel.

    Attributes:
        selected_feature_type: tau*, "value" or "rank".
        table: R^2 on the test and transformed test instances and the degradation,
            with the rows "value" and "rank".
        split: The instances that were used.
        performance: J with one row per instance_id and one column per algorithm.
        features: The features of each feature type ("value" and "rank").
        config: The eight settings; ``rr.run_test(..., **result.config)`` repeats
            the test with them.
    """

    selected_feature_type: Literal["value", "rank"]
    table: pd.DataFrame
    split: InstanceSplit
    performance: pd.DataFrame
    features: dict[str, pd.DataFrame]
    config: RunConfig
    _warnings: tuple[str, ...] = field(default=())

    def _card(self) -> Card:
        return card(
            self.table,
            self.selected_feature_type,
            n_algorithms=len(self.performance.columns),
            set_sizes=(
                len(self.split.train),
                len(self.split.test),
                len(self.split.transformed_test),
            ),
            config=self.config,
            warnings=self._warnings,
        )

    def __str__(self) -> str:
        return self._card().text()

    def __repr__(self) -> str:
        return str(self)

    def __rich__(self) -> Panel:
        return self._card().panel()


def run_test(
    portfolio: Mapping[str, Algorithm],
    feature_extractor: FeatureExtractor,
    predictor: object,
    *,
    suite: Literal["bbob", "ma-bbob"] = "bbob",
    dimension: int = 10,
    train_instance_numbers: Sequence[int] | None = None,
    test_instance_numbers: Sequence[int] | None = None,
    budget_per_dimension: int = 1000,
    n_runs: int = 31,
    n_samples_per_dimension: int = 50,
    seed: int = 42,
    n_jobs: int = -1,
    cache_dir: str | os.PathLike[str] | None = ".rubberruler",
) -> "RubberRulerResult":
    """Run the Rubber Ruler Test: should the features be value-based or rank-based?

    The predictor learns J from the features of the training instances. Its R^2 is
    measured on the test instances and on the transformed test instances, whose
    objective went through a strictly increasing transformation. The answer is the
    feature type with the larger R^2 on the transformed test instances.

    Before the long computation, a short pre-check tries the portfolio, the feature
    extractor, and the predictor. The defaults are the settings of the paper. At the
    end, the R^2 of the card is drawn as a bar chart, rubber_ruler_test.png and .svg
    in cache_dir, for slides and papers.

    While it runs, stderr shows the plan (how many runs there are, and how many of
    them are cached), the four stages with progress bars, and where the figure went.
    When only some runs are cached, a line under the plan names the algorithms that
    run again. Ctrl+C stops it within about a second and leaves no worker process
    behind, also in slim containers without ps or pgrep. The same call again
    resumes from the runs kept in cache_dir.

    Args:
        portfolio: Algorithm names mapped to algorithms: at least 2, with names that
            are non-empty strings. Each algorithm takes a ``rr.Problem`` and
            evaluates it with ``problem(x)``.
        feature_extractor: A function ``(X, y) -> dict[str, float]``.
        predictor: An unfitted regressor with ``fit(X, y)`` and ``predict(X)``,
            such as ``RandomForestRegressor(random_state=0)``.
        suite: The problem suite of all instances: "bbob" (the 24 BBOB functions,
            24 instances per instance number) or "ma-bbob" (MA-BBOB, one instance
            per instance number).
        dimension: d, an int of at least 2.
        train_instance_numbers: The instance numbers of the training instances:
            ints from 1 to 2**31 - 1, without duplicates, and at least one of them.
            None takes the default of the suite: range(11, 111) on BBOB and
            range(1, 2401) on MA-BBOB.
        test_instance_numbers: The instance numbers of the test instances: the
            same, and none of them among the training ones. None takes the default
            of the suite: range(1, 11) on BBOB and range(2401, 2641) on MA-BBOB.
        budget_per_dimension: The budget of one run divided by d, an int of at
            least 1.
        n_runs: The number of runs of each algorithm on each instance, an int of at
            least 1.
        n_samples_per_dimension: The number of points for the features divided by d,
            an int of at least 1.
        seed: The master seed of rubberruler, an int from 0 to 2**32 - 1. It is not
            the predictor's random_state; set that on the predictor yourself.
        n_jobs: The number of worker processes, an int other than 0; -1 uses every
            CPU.
        cache_dir: The folder that keeps finished runs and the figure, a path you can
            write to; None keeps nothing. Two calls that use the same folder at the
            same time do not share their work: each runs everything, with the same
            results.

    Returns:
        The result. ``print(result)`` shows the card, ``result.selected_feature_type``
        is the answer ("value" or "rank"), and ``result.table`` holds the R^2.
        ``result.split``, ``result.performance``, and ``result.features`` go into
        ``rr.fit_predictor`` and ``rr.r2_score`` to try another predictor.
    """
    return run(
        portfolio,
        feature_extractor,
        predictor,
        suite=suite,
        dimension=dimension,
        train_instance_numbers=train_instance_numbers,
        test_instance_numbers=test_instance_numbers,
        budget_per_dimension=budget_per_dimension,
        n_runs=n_runs,
        n_samples_per_dimension=n_samples_per_dimension,
        seed=seed,
        n_jobs=n_jobs,
        cache_dir=cache_dir,
        function_ids=FUNCTION_IDS,
    )


def run(
    portfolio: Mapping[str, Algorithm],
    feature_extractor: FeatureExtractor,
    predictor: object,
    *,
    suite: Suite = "bbob",
    dimension: int,
    train_instance_numbers: Sequence[int] | None,
    test_instance_numbers: Sequence[int] | None,
    budget_per_dimension: int,
    n_runs: int,
    n_samples_per_dimension: int,
    seed: int,
    n_jobs: int,
    cache_dir: str | os.PathLike[str] | None,
    function_ids: Sequence[int],
) -> RubberRulerResult:
    """run_test on the chosen BBOB functions (the tests use fewer than 24).

    On MA-BBOB, ``function_ids`` is not used.
    """
    algorithms = check_portfolio(portfolio)
    suite = check_suite(suite)
    dimension = check_dimension(dimension)
    default_train, default_test = DEFAULT_INSTANCE_NUMBERS[suite]
    train, test = check_instance_numbers(
        default_train if train_instance_numbers is None else train_instance_numbers,
        default_test if test_instance_numbers is None else test_instance_numbers,
        suite,
    )
    config = RunConfig(
        suite=suite,
        dimension=dimension,
        train_instance_numbers=train,
        test_instance_numbers=test,
        budget_per_dimension=check_budget_per_dimension(budget_per_dimension),
        n_runs=check_n_runs(n_runs),
        n_samples_per_dimension=check_n_samples_per_dimension(n_samples_per_dimension),
        seed=check_seed(seed),
    )
    n_jobs = check_n_jobs(n_jobs)
    folder = check_cache_dir(cache_dir)

    # The first training instance of each BBOB function, or the first 24 training
    # instances of MA-BBOB, which has no functions.
    firsts = make_split(
        dimension,
        train[:1] if suite == "bbob" else train[:MA_BBOB_PRECHECK_INSTANCES],
        (),
        config["seed"],
        function_ids,
        suite=suite,
    ).train
    precheck_predictor(predictor)
    precheck_extractor(feature_extractor, firsts, n_samples_per_dimension)
    precheck_algorithms(algorithms, firsts[0], n_jobs=n_jobs)

    split = make_split(
        dimension, train, test, config["seed"], function_ids, suite=suite
    )
    instances = list(split)
    plan = plan_runs(
        algorithms,
        instances,
        budget_per_dimension=config["budget_per_dimension"],
        n_runs=config["n_runs"],
        cache_dir=folder,
    )
    say(plan_line(plan))

    with stage("[1/4] Preparing instances"):
        for instance in split.transformed_test:
            transformation_of(instance)

    measurement = run_plan(
        plan, algorithms, n_jobs=n_jobs, description="[2/4] Measuring performance"
    )
    performance = measurement.performance

    tables: dict[str, FeatureTable] = {}
    raw_values: dict[tuple[str, int], NDArray[np.float64]] = {}
    with progress("[3/4] Computing features", 2 * len(instances)) as advance:
        for feature_type in FEATURE_TYPES:
            tables[feature_type] = feature_table(
                feature_extractor,
                instances,
                feature_type,
                config["n_samples_per_dimension"],
                advance=advance,
                raw_values=raw_values,
            )

    rows: dict[str, list[float]] = {}
    with stage("[4/4] Fitting predictors and scoring"):
        for feature_type in FEATURE_TYPES:
            features = tables[feature_type].features
            names, targets = performance_matrix(performance, split.train, "run_test")
            fitted = fit(
                predictor,
                feature_type,
                list(features.columns),
                names,
                feature_matrix(features, split.train, "run_test"),
                targets,
            )
            r2 = [
                score(
                    fitted,
                    feature_matrix(features, instance_set, "run_test"),
                    performance_matrix(performance, instance_set, "run_test")[1],
                )
                for instance_set in (split.test, split.transformed_test)
            ]
            rows[feature_type] = [r2[0], r2[1], r2[0] - r2[1]]
    table = pd.DataFrame.from_dict(rows, orient="index", columns=TABLE_COLUMNS)
    if any(math.isnan(value) for value in table["r2_transformed_test"]):
        raise error(
            "R^2 on the transformed test instances is nan, so there is no answer.",
            why="every algorithm has the same J on the transformed test instances, "
            "so there is nothing to predict.",
            fix="use algorithms that perform differently, or raise "
            "budget_per_dimension or n_runs.",
        )

    result = RubberRulerResult(
        selected_feature_type=select(table),
        table=table,
        split=split,
        performance=performance,
        features={name: tables[name].features for name in FEATURE_TYPES},
        config=config,
        _warnings=collect_warnings(measurement, tables, predictor),
    )
    if folder is not None:
        write_figure(table, folder)
    return result


def write_figure(table: pd.DataFrame, folder: Path) -> None:
    """Save the figure of the answer into cache_dir; if that fails, only warn."""
    location = os.path.join(str(folder), "")
    try:
        save_figure(table, folder)
    except Exception as error:
        logger.warning(
            "could not save the figure in %s: %s: %s",
            location,
            type(error).__name__,
            error,
        )
        return
    say(f"rubberruler: figure saved to {location}{FIGURE_FILE_NAME}.png and .svg")


def collect_warnings(
    measurement: Measurement,
    tables: Mapping[str, FeatureTable],
    predictor: object,
) -> tuple[str, ...]:
    """The warning lines for the card: failed runs, features, caching, random_state."""
    nan_instances = sorted(
        {name for table in tables.values() for name in table.nan_instances}
    )
    constant_instances = tables["value"].constant_instances
    lines = [
        *measurement.warnings,
        *feature_warnings(nan_instances, constant_instances),
        *measurement.identity_warnings,
    ]
    if has_unset_random_state(predictor):
        lines.append(RANDOM_STATE_WARNING)
    return tuple(lines)
