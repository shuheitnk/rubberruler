# Examples

The examples are numbered in reading order, from the smallest to those that need more tools. Each runs with `python exampleN.py`; "the test" is the Rubber Ruler Test of `rr.run_test`.

| File | What it does | Replaces | Installs besides rubberruler |
|---|---|---|---|
| [example1.py](#example-1-three-scipy-optimizers-four-features-and-a-random-forest) | Runs the test with three SciPy optimizers, four features, and a random forest. | none (the other examples start from it) | scikit-learn |
| [example2.py](#example-2-your-own-objective-functions) | Runs the test on 40 objective functions of your own instead of BBOB. | objectives | scikit-learn |
| [example3.py](#example-3-augmentation-with-transformed-objectives) | Makes training data from transformed objectives with `rr.transform_objective`. | objectives | |
| [example4.py](#example-4-ga-pso-and-nelder-mead-from-pymoo) | Runs the test with GA, PSO, and Nelder-Mead from pymoo as the portfolio. | portfolio | scikit-learn, pymoo |
| [example5.py](#example-5-modularcmaes-modde-and-the-ela-features-of-pflacco) | Runs the test with ModularCMAES, ModDE, and the 55 ELA features of pflacco. | portfolio, feature extractor, predictor | pflacco, modcma, and modde, on Python 3.11 |
| [example6.py](#example-6-the-learned-features-of-deepela) | Runs the test with the 48 learned features of DeepELA as the extractor. | feature extractor | DeepELA from GitHub, scikit-learn, torch, lightning, einops, pyyaml |
| [example7.py](#example-7-a-feature-free-model-in-pytorch) | Runs the test with a feature-free model that learns J from the sample itself. | feature extractor, predictor | torch |

example4.py, example5.py, and example6.py use the default settings, which follow the paper except that the sample points are uniform random rather than a Latin hypercube and the test uses one seed rather than 11, and take from half an hour to several hours. The code below continues example1.py, whose names `portfolio`, `simple_features`, and `predictor` it uses.

## Example 1: Three SciPy optimizers, four features, and a random forest

```sh
pip install rubberruler scikit-learn
python example1.py
```

[example1.py](example1.py) is the [Quickstart](../README.md#quickstart) of the README, which shows the card that it prints and how to read it. With its small settings, it takes about half a minute on 24 CPUs. The other examples replace its portfolio, feature extractor, predictor, or objectives.

## Example 2: Your own objective functions

Map names to an objective and the bounds of its search range:

```python
objectives = {
    "sphere": (lambda x: float(np.sum(x**2)), np.full(5, -5.0), np.full(5, 5.0)),
    "rastrigin": (
        lambda x: float(10 * x.size + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))),
        np.full(5, -5.12),
        np.full(5, 5.12),
    ),
    # ... more objectives of the same dimension
}
for split_seed in (1, 2, 3):
    split = rr.objective_instances(objectives, split_seed=split_seed)
    result = rr.run_test(portfolio, simple_features, predictor, instances=split)
    print(split_seed, result.selected_feature_type)
```

Run [example2.py](example2.py) for 40 objectives; it takes about 8 seconds on 24 CPUs.

- One objective in four is a test instance, drawn by `split_seed`.
- With few objectives, the answer can change with the split, so try a few values of `split_seed`.
- Another `split_seed` reuses the cached runs of the same objectives.
- `test=["sphere"]` names the test instances instead.
- `optimal_values={"sphere": 0.0}` gives known optimal values to the rubber rulers.

## Example 3: Augmentation with transformed objectives

`rr.transform_objective` applies a random rubber ruler to an objective. Transformed objectives added to the training data teach the predictor value-based features under many rubber rulers:

```python
def rastrigin(x):
    return 10 * x.shape[-1] + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=-1)


lower, upper = np.full(5, -5.0), np.full(5, 5.0)
augmented = {
    f"rastrigin-{k}": (
        rr.transform_objective(rastrigin, lower, upper, seed=k),
        lower,
        upper,
    )
    for k in range(10)
}
split = rr.objective_instances({"rastrigin": (rastrigin, lower, upper), **augmented})
```

Give each transformed objective its own `seed`. Run [example3.py](example3.py); it prints the range of one feature over the transformed objectives:

```text
22 instances, J of 3 algorithms
rastrigin  value kurtosis from -1.033 to  4.090
rastrigin  rank  kurtosis from -1.200 to -1.200
rosenbrock value kurtosis from -1.397 to  6.428
rosenbrock rank  kurtosis from -1.200 to -1.200
```

Value-based features spread over the rubber rulers, and rank-based features stay the same.

## Example 4: GA, PSO, and Nelder-Mead from pymoo

```sh
pip install pymoo==0.6.2
```

```python
from pymoo.core.problem import Problem as PymooProblem
from pymoo.optimize import minimize as pymoo_minimize


class PymooView(PymooProblem):
    """The problem as pymoo sees it. pymoo passes the solutions together, (n, d)."""

    def __init__(self, problem):
        super().__init__(
            n_var=problem.dimension,
            n_obj=1,
            xl=problem.lower_bounds,
            xu=problem.upper_bounds,
        )
        self.problem = problem

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = self.problem(X)


def pymoo_algorithm(make_algorithm):
    """A pymoo algorithm, made new by make_algorithm() for each run."""

    def algorithm(problem):
        pymoo_minimize(
            PymooView(problem),
            make_algorithm(),
            ("n_eval", problem.budget),
            seed=problem.seed,
            verbose=False,
        )

    return algorithm
```

Run [example4.py](example4.py) for GA, PSO, and Nelder-Mead.

## Example 5: ModularCMAES, ModDE, and the ELA features of pflacco

[example5.py](example5.py) puts ModularCMAES and ModDE in the portfolio and pflacco's ELA features in the extractor. Because of pflacco, it needs Python 3.11.

### ModularCMAES

```sh
pip install modcma==1.2.0
```

```python
from modcma import c_maes


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
```

### ModDE

```sh
pip install modde==0.0.4
```

```python
from modde import ModularDE


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
```

This is the portfolio of example5.py.

### pflacco

pflacco installs only on Python 3.11:

```sh
pip install pflacco==1.2.2 "pyarrow<26" modcma==1.2.0 modde==0.0.4
```

```python
from pflacco.classical_ela_features import (
    calculate_dispersion,
    calculate_ela_distribution,
    calculate_ela_level,
    calculate_ela_meta,
    calculate_information_content,
    calculate_nbc,
    calculate_pca,
)


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
```

Some features can be NaN, so example5.py uses [a pipeline that fills NaN in](#a-pipeline-that-fills-nan-in). Keep d = 10: at d = 2, `calculate_ela_level` fails on some instances.

### A pipeline that fills NaN in

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

predictor = make_pipeline(SimpleImputer(), RandomForestRegressor(random_state=0))
```

## Example 6: The learned features of DeepELA

DeepELA is not on PyPI. Fetch it at a fixed commit and put it on `PYTHONPATH`:

```sh
commit=c0368677461f96a17536111ee947bf111e8b8425
git init deep_ela
git -C deep_ela fetch --depth 1 https://github.com/mvseiler/deep_ela $commit
git -C deep_ela checkout FETCH_HEAD
pip install torch lightning einops pyyaml
export PYTHONPATH="$PWD/deep_ela"
```

```python
import deep_ela  # github.com/mvseiler/deep_ela at c036867, on PYTHONPATH
import torch
from sklearn.ensemble import RandomForestRegressor

model = deep_ela.load_large_50d_v1(stride=1)


def deep_ela_features(X, y):
    if X.shape[1] > 11:
        raise ValueError(
            "DeepELA's large models take at most 11 dimensions, "
            f"but received {X.shape[1]}"
        )
    torch.manual_seed(0)  # predict() permutes the input slots with torch's global RNG
    return model(X, y)


predictor = RandomForestRegressor(random_state=0)
# Each worker loads torch and its own copy of the model (about 0.5 GB); 4 fit in 16 GB.
result = rr.run_test(portfolio, deep_ela_features, predictor, n_jobs=4)
```

Run [example6.py](example6.py). The features of the default settings took 41 minutes on 24 CPUs.

## Example 7: A feature-free model in PyTorch

A feature-free model receives the sample itself. The extractor returns the sample, and the model reads it back as n points of size d + 1:

```python
def sample(X, y):
    """The sample itself: the coordinates and the value of every point, in order."""
    points = np.column_stack([X, y])
    return {f"p{k}": value for k, value in enumerate(points.ravel())}
```

Run [example7.py](example7.py) for a small DeepSets network in PyTorch; it takes about 80 seconds on 24 CPUs.

## Algorithms

An algorithm minimizes `problem` by calling `problem(x)`. rubberruler ends the run when the budget is spent. Example 4 connects pymoo, and Example 5 connects ModularCMAES and ModDE.

### pycma and Nevergrad

```sh
pip install cma nevergrad
```

```python
import cma
import numpy as np


def run_cma_es(problem):
    x0 = np.random.default_rng(problem.seed).uniform(
        problem.lower_bounds, problem.upper_bounds
    )
    options = {
        "bounds": [problem.lower_bounds, problem.upper_bounds],
        "maxfevals": problem.budget,
        "seed": problem.seed,
        "verbose": -9,
    }
    cma.fmin2(problem, x0, 2.0, options)
```

```python
import nevergrad as ng
import numpy as np


def run_nevergrad_cma(problem):
    parametrization = ng.p.Array(shape=(problem.dimension,)).set_bounds(
        problem.lower_bounds, problem.upper_bounds
    )
    parametrization.random_state = np.random.RandomState(problem.seed)
    ng.optimizers.CMA(parametrization, budget=problem.budget).minimize(problem)
```

pycma can stop before it spends its budget. An algorithm that stops early is evaluated with that stop, because J measures the algorithms as they are.

### Functions of another shape

A function with other arguments takes one line: `"my-es": lambda p: my_es(p, p.dimension, p.budget, p.seed)`.

## Feature extractors

A feature extractor receives a sample `X` of shape (n, d) and its values `y` of shape (n,) in [0, 1], and returns the same keys for every instance. Example 5 connects pflacco, and Example 6 connects DeepELA.

### An extractor of another shape

A function of another shape takes one line:

```python
import numpy as np
import pandas as pd


def my_features(y, X):
    """Features in another shape: y comes first, and a pandas Series comes back."""
    return pd.Series(
        {"y_mean": y.mean(), "y_std": y.std(), "best_x0": X[np.argmin(y), 0]}
    )


feature_extractor = lambda X, y: dict(my_features(y, X))
```

## Predictors

A predictor is any object with `fit(X, y)` and `predict(X)`. Set its own seed, such as `random_state=0`. Example 5 uses [a pipeline that fills NaN in](#a-pipeline-that-fills-nan-in), and Example 7 uses a network in PyTorch that reads the sample.

### PyTorch

```sh
pip install torch
```

```python
import torch


class TorchRegressor:
    def fit(self, X, y):
        torch.manual_seed(0)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        inputs = torch.as_tensor(X, dtype=torch.float32)
        targets = torch.as_tensor(y, dtype=torch.float32)
        for _ in range(500):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(self.model(inputs).squeeze(1), targets)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X):
        with torch.no_grad():
            inputs = torch.as_tensor(X, dtype=torch.float32)
            return self.model(inputs).squeeze(1).numpy()
```

## MA-BBOB

`rr.run_test(portfolio, simple_features, predictor, suite="ma-bbob")` uses MA-BBOB, the affine combinations of the BBOB functions. Its training instances are 1 to 2400, and its test instances are 2401 to 2640. With the default settings, the test of example1.py on MA-BBOB takes about two hours on 16 cores.

## Precomputed tables

`run_test` takes the performance and the features that you already have, and computes only the missing rows. This scores another predictor without running the algorithms again:

```python
from sklearn.ensemble import GradientBoostingRegressor

with_boosting = rr.run_test(
    None,
    None,
    GradientBoostingRegressor(random_state=0),
    performance=result.performance,
    features=result.features,
    **result.config,
)
print(with_boosting)
```

The stages are also available one by one:

```python
fitted = rr.fit_predictor(
    GradientBoostingRegressor(random_state=0),
    result.features["rank"],
    result.performance,
    instances=result.split.train,
)
r2 = rr.r2_score(
    fitted,
    result.features["rank"],
    result.performance,
    instances=result.split.transformed_test,
)
```

`r2` is the R² on the transformed test instances that the card compares.

## Parallel computation and memory

`n_jobs` worker processes run the algorithms and compute the features; the default `-1` uses every CPU, and the result does not depend on `n_jobs`. Each worker holds its own copy of the extractor. With the default settings and 24 workers, the features took at most 2.6 GB for the extractor of example1.py, 2.8 GB for pflacco, and 13 GB for DeepELA. With `n_jobs=4`, DeepELA took 2.9 GB.
