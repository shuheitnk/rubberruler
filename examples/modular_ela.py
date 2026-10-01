"""ModularCMAES and ModDE in the portfolio, and pflacco's ELA features as the extractor.

pflacco pins NumPy 1.24 and SciPy 1.10, which have no wheels for Python 3.12, so this
example needs Python 3.11:

    pip install rubberruler pflacco==1.2.2 modcma==1.2.0 modde==0.0.4

It runs with the default settings, the settings of the paper, which take long: most of
the time goes to the ELA features, which take about half a second per instance at
d = 10 and are computed one after another, for 2,880 instances and both feature types.
To try it first, give run_test at the end fewer instances and runs:

    train_instance_numbers=range(11, 16), test_instance_numbers=range(1, 4),
    budget_per_dimension=100, n_runs=5

That took about 3.5 minutes on 16 cores, almost all of it for the features. Keep
d = 10: at d = 2, pflacco's ela_level cannot split the plateaus of some instances of
f7 and stops with an exception.
"""

from modcma import c_maes
from modde import ModularDE
from pflacco.classical_ela_features import (
    calculate_dispersion,
    calculate_ela_distribution,
    calculate_ela_level,
    calculate_ela_meta,
    calculate_information_content,
    calculate_nbc,
    calculate_pca,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

import rubberruler as rr


def modular_cmaes(**modules):
    """ModularCMAES, the C++ version, with the given modules."""

    def algorithm(problem):
        c_maes.utils.set_seed(problem.seed % 2**31)  # it takes seeds below 2**31
        settings = c_maes.settings_from_dict(
            problem.dimension,
            budget=problem.budget,
            lb=problem.lower_bounds,
            ub=problem.upper_bounds,
            **modules,
        )
        c_maes.ModularCMAES(settings)(problem)

    return algorithm


def modular_de(**modules):
    """ModDE with the given modules. It takes the bounds with shape (d, 1)."""

    def algorithm(problem):
        ModularDE(
            problem,
            budget=problem.budget,
            seed=problem.seed,
            lb=problem.lower_bounds.reshape(-1, 1),
            ub=problem.upper_bounds.reshape(-1, 1),
            **modules,
        ).run()

    return algorithm


portfolio = {
    "cma-es": modular_cmaes(),
    "ipop-cma-es": modular_cmaes(restart_strategy="IPOP"),
    "de": modular_de(),
    "shade": modular_de(
        mutation_base="target",
        mutation_reference="pbest",
        use_archive=True,
        adaptation_method_F="shade",
        adaptation_method_CR="shade",
    ),
}


def ela_features(X, y):
    """55 ELA features from seven pflacco feature sets.

    Every set also reports what it cost, such as the seconds it took
    (``*.costs_runtime``). Those change from run to run, so they are left out.
    ela_meta.quad_simple.cond divides by zero on the linear slope (f5) and gives
    infinity, which rubberruler counts as NaN.
    """
    features = (
        calculate_ela_meta(X, y)
        | calculate_ela_distribution(X, y)
        | calculate_ela_level(X, y)
        | calculate_pca(X, y)
        | calculate_nbc(X, y)
        | calculate_dispersion(X, y)
        | calculate_information_content(X, y, seed=0)
    )
    return {name: value for name, value in features.items() if ".costs_" not in name}


# The features can be NaN, so the pipeline fills them in before the random forest.
predictor = make_pipeline(SimpleImputer(), RandomForestRegressor(random_state=0))
result = rr.run_test(portfolio, ela_features, predictor)
print(result)
