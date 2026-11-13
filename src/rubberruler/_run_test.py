"""run_test: value-based or rank-based features?"""

import logging
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import pandas as pd
from rich.panel import Panel

from rubberruler._cache import check_cache_dir
from rubberruler._card import BASELINE, Card, RunConfig, card, select
from rubberruler._console import progress, say, stage
from rubberruler._errors import error
from rubberruler._features import (
    FEATURE_TYPES,
    FeatureExtractor,
    FeatureTable,
    FeatureType,
    check_n_samples_per_dimension,
    feature_tables,
    feature_warnings,
    precheck_extractor,
)
from rubberruler._figure import FILE_NAME as FIGURE_FILE_NAME
from rubberruler._figure import save_figure
from rubberruler._instances import (
    DEFAULT_INSTANCE_NUMBERS,
    FUNCTION_IDS,
    OBJECTIVES,
    InstanceSplit,
    ProblemInstance,
    Suite,
    check_dimension,
    check_instance_numbers,
    check_seed,
    check_suite,
    make_split,
    reseeded,
    transformation_of,
)
from rubberruler._performance import (
    Measurement,
    check_budget_per_dimension,
    check_n_jobs,
    check_n_runs,
    check_portfolio,
    check_sendable,
    plan_line,
    plan_runs,
    precheck_algorithms,
    run_plan,
)
from rubberruler._prediction import (
    RANDOM_STATE_WARNING,
    baseline_score,
    feature_matrix,
    fit,
    has_unset_random_state,
    performance_matrix,
    precheck_predictor,
    score,
)
from rubberruler._problem import Algorithm
from rubberruler._tables import (
    check_given_columns,
    check_missing_inputs,
    given_features,
    given_items,
    given_performance,
    joined_features,
    joined_performance,
)

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
    # The items that state which rows came from tables given to run_test.
    _given: tuple[str, ...] = field(default=())

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
            given=self._given,
        )

    def __str__(self) -> str:
        return self._card().text()

    def __repr__(self) -> str:
        return str(self)

    def __rich__(self) -> Panel:
        return self._card().panel()


def run_test(
    portfolio: Mapping[str, Algorithm] | None,
    feature_extractor: FeatureExtractor | None,
    predictor: object,
    *,
    suite: Literal["bbob", "ma-bbob", "objectives"] = "bbob",
    dimension: int = 10,
    train_instance_numbers: Sequence[int] | None = None,
    test_instance_numbers: Sequence[int] | None = None,
    budget_per_dimension: int = 1000,
    n_runs: int = 31,
    n_samples_per_dimension: int = 50,
    seed: int = 42,
    n_jobs: int = -1,
    cache_dir: str | os.PathLike[str] | None = ".rubberruler",
    instances: InstanceSplit | None = None,
    performance: pd.DataFrame | None = None,
    features: Mapping[str, pd.DataFrame] | None = None,
) -> "RubberRulerResult":
    """Run the Rubber Ruler Test: should the features be value-based or rank-based?

    The predictor learns J from the features of the training instances. Its R^2 is
    measured on the test instances and on the transformed test instances, whose
    objective went through a strictly increasing transformation. The answer is the
    feature type with the larger R^2 on the transformed test instances.

    Before the long computation, a short pre-check tries the portfolio, the feature
    extractor, and the predictor. The defaults follow the paper (dimension 10,
    50 x d sample points, a budget of 1000 x d evaluations, 31 runs, training
    instances 11 to 110 and test instances 1 to 10 on BBOB, and the same
    distribution of rubber rulers), except that the sample points are uniform random
    rather than a Latin hypercube and the test uses one seed rather than the
    paper's 11. At the end, the R^2 of the card is drawn as a bar chart,
    rubber_ruler_test.png and .svg in cache_dir, for slides and papers.

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
            per instance number). For your own objectives, pass ``instances``
            instead; "objectives" is their suite in ``result.config``.
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
        instances: The instances of your own objective functions, from
            ``rr.objective_instances(...)`` (or any split of
            ``rr.bbob_instances`` or ``rr.ma_bbob_instances``). They decide the
            suite, the dimension, the instance numbers, and the seed, so leave
            those out.

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
        instances=instances,
        performance=performance,
        features=features,
    )


def run(
    portfolio: Mapping[str, Algorithm] | None,
    feature_extractor: FeatureExtractor | None,
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
    instances: InstanceSplit | None = None,
    performance: pd.DataFrame | None = None,
    features: Mapping[str, pd.DataFrame] | None = None,
) -> RubberRulerResult:
    """run_test on the chosen BBOB functions (the tests use fewer than 24).

    On MA-BBOB and with ``instances``, ``function_ids`` is not used.
    """
    algorithms = None if portfolio is None else check_portfolio(portfolio)
    suite = check_suite(suite)
    common = {
        "budget_per_dimension": check_budget_per_dimension(budget_per_dimension),
        "n_runs": check_n_runs(n_runs),
        "n_samples_per_dimension": check_n_samples_per_dimension(
            n_samples_per_dimension
        ),
    }
    if instances is None:
        if suite == OBJECTIVES:
            raise error(
                'suite="objectives" received no instances.',
                why="the instances of your objectives come from "
                "rr.objective_instances, which knows the functions.",
                fix="pass instances=rr.objective_instances({...}) instead of suite.",
            )
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
            budget_per_dimension=common["budget_per_dimension"],
            n_runs=common["n_runs"],
            n_samples_per_dimension=common["n_samples_per_dimension"],
            seed=check_seed(seed),
            instances=None,
        )
        split = make_split(
            dimension, train, test, config["seed"], function_ids, suite=suite
        )
    else:
        split = check_split(instances)
        given: dict[str, object] = {
            "suite": suite,
            "dimension": dimension,
            "train_instance_numbers": train_instance_numbers,
            "test_instance_numbers": test_instance_numbers,
            "seed": seed,
        }
        if split.train[0].suite == OBJECTIVES:
            # The objectives of the user take the master seed of run_test.
            split = reseeded(split, check_seed(given.pop("seed")))
        config = config_of(split, common, given=given)
    n_jobs = check_n_jobs(n_jobs)
    folder = check_cache_dir(cache_dir)

    instances_list = list(split)
    given_j = given_performance(
        performance, instances_list, None if algorithms is None else list(algorithms)
    )
    given_phi = given_features(features, instances_list)
    to_measure = [
        p
        for p in instances_list
        if given_j is None or p.instance_id not in given_j.index
    ]
    to_compute: dict[FeatureType, list[ProblemInstance]] = {
        kind: [
            p
            for p in instances_list
            if kind not in given_phi or p.instance_id not in given_phi[kind].index
        ]
        for kind in FEATURE_TYPES
    }
    check_missing_inputs(
        algorithms, feature_extractor, to_measure, to_compute, len(instances_list)
    )

    # The pre-checks can take a while with a slow extractor, so they are announced.
    say("rubberruler: pre-checks of the extractor, the predictor, and the portfolio")
    names: Sequence[str] = list(next(iter(given_phi.values()), pd.DataFrame()).columns)
    if feature_extractor is not None and any(to_compute.values()):
        # The extractor on the first training instance of each BBOB function, or on
        # the first 24 training instances of MA-BBOB or of your objectives.
        names = precheck_extractor(
            feature_extractor,
            split.train,
            config["n_samples_per_dimension"],
            n_jobs=n_jobs,
        )
        check_given_columns(names, given_phi, to_compute)
    # On dummy data with as many columns as the features (a model that reads the
    # features as a sample of points needs their number).
    precheck_predictor(predictor, max(1, len(names)))
    if algorithms is not None and to_measure:
        precheck_algorithms(algorithms, split.train[0], n_jobs=n_jobs)
    check_sendable(split, n_jobs)

    plan = None
    if algorithms is not None and to_measure:
        plan = plan_runs(
            algorithms,
            to_measure,
            budget_per_dimension=config["budget_per_dimension"],
            n_runs=config["n_runs"],
            cache_dir=folder,
        )
        say(plan_line(plan))
    else:
        say(
            f"rubberruler: the performance of all {len(instances_list):,} instances "
            "was given, so there are no runs"
        )

    with stage("[1/4] Preparing instances"):
        for instance in split.transformed_test:
            transformation_of(instance)

    if algorithms is not None and plan is not None:
        measurement = run_plan(
            plan, algorithms, n_jobs=n_jobs, description="[2/4] Measuring performance"
        )
    else:
        with stage("[2/4] Measuring performance"):
            measurement = Measurement(pd.DataFrame(), (), (), ())
    performance_of_all = joined_performance(
        given_j, measurement.performance if plan is not None else None, instances_list
    )

    total = sum(len(selected) for selected in to_compute.values())
    computed: dict[FeatureType, FeatureTable] = {}
    with progress("[3/4] Computing features", total) as advance:
        if feature_extractor is not None and to_compute["value"] == to_compute["rank"]:
            if to_compute["value"]:
                computed = feature_tables(
                    feature_extractor,
                    to_compute["value"],
                    FEATURE_TYPES,
                    config["n_samples_per_dimension"],
                    n_jobs=n_jobs,
                    advance=advance,
                )
        elif feature_extractor is not None:
            for kind in FEATURE_TYPES:
                if to_compute[kind]:
                    computed[kind] = feature_tables(
                        feature_extractor,
                        to_compute[kind],
                        (kind,),
                        config["n_samples_per_dimension"],
                        n_jobs=n_jobs,
                        advance=advance,
                    )[kind]
    tables: dict[FeatureType, FeatureTable] = {
        kind: joined_features(
            kind, given_phi.get(kind), computed.get(kind), instances_list
        )
        for kind in FEATURE_TYPES
    }
    performance = performance_of_all
    given_lines = given_items(given_j, given_phi, len(instances_list))

    rows: dict[str, list[float]] = {}
    fits = len(FEATURE_TYPES) * len(performance.columns)
    with progress("[4/4] Fitting predictors and scoring", fits) as advance:
        for feature_type in FEATURE_TYPES:
            phi = tables[feature_type].features
            names, targets = performance_matrix(performance, split.train, "run_test")
            fitted = fit(
                predictor,
                feature_type,
                list(phi.columns),
                names,
                feature_matrix(phi, split.train, "run_test"),
                targets,
                advance,
            )
            r2 = [
                score(
                    fitted,
                    feature_matrix(phi, instance_set, "run_test"),
                    performance_matrix(performance, instance_set, "run_test")[1],
                )
                for instance_set in (split.test, split.transformed_test)
            ]
            rows[feature_type] = [r2[0], r2[1], r2[0] - r2[1]]
        train_targets = performance_matrix(performance, split.train, "run_test")[1]
        baseline = [
            baseline_score(
                train_targets,
                performance_matrix(performance, instance_set, "run_test")[1],
            )
            for instance_set in (split.test, split.transformed_test)
        ]
        rows[BASELINE] = [baseline[0], baseline[1], baseline[0] - baseline[1]]
    table = pd.DataFrame.from_dict(rows, orient="index", columns=TABLE_COLUMNS)
    if any(math.isnan(rows[name][1]) for name in FEATURE_TYPES):
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
        _given=tuple(given_lines),
    )
    if folder is not None:
        write_figure(table, folder)
    return result


def check_split(instances: object) -> InstanceSplit:
    """Return the split given as instances; stop if it is not one, or a set is empty."""
    if not isinstance(instances, InstanceSplit):
        raise error(
            f"instances received a {type(instances).__name__}.",
            why="instances is a split of training, test, and transformed test "
            "instances.",
            fix="pass the result of rr.objective_instances(...), "
            "rr.bbob_instances(...), or rr.ma_bbob_instances(...).",
        )
    if not instances.train or not instances.test:
        raise error(
            f"instances has {len(instances.train)} training and "
            f"{len(instances.test)} test instances.",
            why="the predictor learns from the training instances and is scored on "
            "the test instances, so each set needs at least one.",
            fix="pass a split with training and test instances.",
        )
    return instances


def config_of(
    split: InstanceSplit, common: Mapping[str, int], *, given: Mapping[str, object]
) -> RunConfig:
    """The settings of a run on given instances; stop if a setting that the
    instances decide was given with another value than its default.

    The instances decide the suite, the dimension, the instance numbers, and, except
    on the suite "objectives", the master seed. Passing their own values again, as
    ``**result.config`` does, is allowed.
    """
    first = split.train[0]
    decided: dict[str, object] = {
        "suite": first.suite,
        "dimension": first.dimension,
        "train_instance_numbers": tuple(
            sorted({p.instance_number for p in split.train})
        ),
        "test_instance_numbers": tuple(sorted({p.instance_number for p in split.test})),
        "seed": first.seed,
    }
    for name, value in given.items():
        default = RUN_TEST_DEFAULTS[name]
        same = (
            value == decided[name]
            if not isinstance(value, Sequence) or isinstance(value, str)
            else tuple(sorted(cast(Sequence[int], value))) == decided[name]
        )
        if value != default and not same:
            raise error(
                f"{name}={value!r} was given together with instances, whose "
                f"{name} is {decided[name]!r}.",
                why="the instances already decide the suite, the dimension, the "
                "instance numbers, and, except for your own objectives, the seed.",
                fix=f"leave out {name}, or make the instances with it.",
            )
    return RunConfig(
        suite=first.suite,
        dimension=first.dimension,
        train_instance_numbers=cast(tuple[int, ...], decided["train_instance_numbers"]),
        test_instance_numbers=cast(tuple[int, ...], decided["test_instance_numbers"]),
        budget_per_dimension=common["budget_per_dimension"],
        n_runs=common["n_runs"],
        n_samples_per_dimension=common["n_samples_per_dimension"],
        seed=first.seed,
        instances=split,
    )


# The defaults of run_test for the settings that given instances decide.
RUN_TEST_DEFAULTS: dict[str, object] = {
    "suite": "bbob",
    "dimension": 10,
    "train_instance_numbers": None,
    "test_instance_numbers": None,
    "seed": 42,
}


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
    tables: Mapping[FeatureType, FeatureTable],
    predictor: object,
) -> tuple[str, ...]:
    """The warning lines for the card: failed runs, features, caching, random_state."""
    nan_instances = sorted(
        {name for table in tables.values() for name in table.nan_instances}
    )
    constant_instances = sorted(
        {name for table in tables.values() for name in table.constant_instances}
    )
    lines = [
        *measurement.warnings,
        *feature_warnings(nan_instances, constant_instances),
        *measurement.identity_warnings,
    ]
    if has_unset_random_state(predictor):
        lines.append(RANDOM_STATE_WARNING)
    return tuple(lines)
