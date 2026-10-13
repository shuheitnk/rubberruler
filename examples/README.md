# Examples

This folder has runnable scripts, and this page shows how to use rubberruler with other tools.

## Example scripts

| Script | What it shows | Install |
|---|---|---|
| [quickstart.py](quickstart.py) | The quickstart of the main README: random search, DE, and Nelder-Mead from SciPy, four ELA-style features, and a random forest | `pip install rubberruler scikit-learn` |
| [pymoo_portfolio.py](pymoo_portfolio.py) | GA, PSO, and Nelder-Mead from pymoo, with the quickstart's extractor and predictor | `pip install rubberruler scikit-learn pymoo==0.6.2` |
| [modular_ela.py](modular_ela.py) | CMA-ES and IPOP-CMA-ES from ModularCMAES, DE and SHADE from ModDE, and pflacco's 55 ELA features (Python 3.11) | `pip install rubberruler pflacco==1.2.2 modcma==1.2.0 modde==0.0.4` |
| [deepela.py](deepela.py) | DeepELA's 48 learned features, with the quickstart's portfolio and predictor | See [DeepELA](#deepela) |
| [augmentation.py](augmentation.py) | Ten rubber rulers on your own objective, and what they do to its features; see [Augmenting training data](#augmenting-training-data) | `pip install rubberruler` |

Run a script with Python, such as `python quickstart.py`. Apart from the quickstart, the scripts use the default settings, the settings of the paper, which take from about half an hour to several hours. The docstring of each script gives small settings to try it first.

## Integrations

Each section below shows how to install a tool and the code that connects it to rubberruler.

| Tool | What it provides | Note |
|---|---|---|
| [ModularCMAES](#modularcmaes) | CMA-ES variants | |
| [ModDE](#modde) | DE variants | |
| [pymoo](#pymoo) | GA, PSO, and more | |
| [pycma, Nevergrad](#pycma-and-nevergrad) | CMA-ES and more | |
| [pflacco](#pflacco) | 55 ELA features | Python 3.11 only |
| [DeepELA](#deepela) | 48 learned features | Installed from GitHub; d ≤ 11 |
| [PyTorch](#pytorch) | A neural network as the predictor | |

## Algorithms

An algorithm minimizes `problem` by calling `problem(x)`. When the budget is spent, the next call ends the run with an exception that `except Exception` does not catch, and the run keeps the smallest value it saw. Points outside the search range can be evaluated too, and they count toward the budget like the others.

SciPy's algorithms are in the quickstart: `de` and `nelder_mead` in [quickstart.py](quickstart.py).

### ModularCMAES

Tested on Python 3.11 and 3.12:

```sh
pip install modcma==1.2.0
```

ModularCMAES, the C++ version, takes seeds below 2³¹:

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

[modular_ela.py](modular_ela.py) puts CMA-ES and IPOP-CMA-ES from `modular_cmaes` into its portfolio.

### ModDE

Tested on Python 3.11 and 3.12:

```sh
pip install modde==0.0.4
```

ModDE takes the bounds with shape (d, 1), and reads `problem.state.optimum_found`, which stays `False`:

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
```

[modular_ela.py](modular_ela.py) puts DE and SHADE from `modular_de` into its portfolio.

### pymoo

Tested on Python 3.11 and 3.12, also alongside pflacco:

```sh
pip install pymoo==0.6.2
```

pymoo passes the solutions together, with shape (n, d), which `problem` takes as they are:

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

To give an algorithm options, pass a lambda, such as `pymoo_algorithm(lambda: GA(pop_size=50))`.

[pymoo_portfolio.py](pymoo_portfolio.py) runs the test with GA, PSO, and Nelder-Mead, and the quickstart's extractor and predictor.

### pycma and Nevergrad

Tested with cma 4.5.0 and nevergrad 1.0.12 on Python 3.12:

```sh
pip install cma nevergrad
```

pycma:

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

Nevergrad:

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

Nevergrad's CMA can print pycma's `InjectionWarning` ("orphanated injected solution") on some BBOB functions. It does so without rubberruler too, and the runs end as usual.

To run them, add them to the portfolio in [quickstart.py](quickstart.py), such as `"cma-es": run_cma_es`.

### IOHexperimenter, and functions of another shape

An algorithm written for IOHexperimenter runs as it is: `problem.bounds.lb`, `problem.bounds.ub`, `problem.meta_data.n_variables`, `problem.state.evaluations`, and `problem.state.optimum_found` (always `False`) are there too.

A function of another shape takes one line in the portfolio: `"my-es": lambda p: my_es(p, p.dimension, p.budget, p.seed)`.

## Feature extractors

### pflacco

pflacco installs only on Python 3.11:

```sh
pip install pflacco==1.2.2
```

pflacco pins NumPy 1.24 and SciPy 1.10, which have no wheels for Python 3.12. On 3.12, pip tries to build NumPy 1.24 and stops with errors such as `Cannot import 'setuptools.build_meta'` or `module 'pkgutil' has no attribute 'ImpImporter'`. Name the version of pflacco: without it, pip can fall back to pflacco 0.4, an old version that needs R through rpy2.

Each feature set of pflacco also reports what it cost, such as the seconds it took (`*.costs_runtime`). Those change from run to run, so leave them out:

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

Some values are infinite, and rubberruler counts them as NaN, so use a predictor that fills NaN in, such as [the pipeline below](#a-pipeline-that-fills-nan-in).

Keep d = 10, unlike the quickstart's d = 2: at d = 2, `calculate_ela_level` cannot split the plateaus of some instances of f7 and raises an exception, which stops `run_test`.

pflacco takes about half a second per instance at d = 10, and the instances are computed one after another, so with the default settings the features alone take most of an hour. To try it first, give `run_test` fewer instances and runs: `train_instance_numbers=range(11, 16), test_instance_numbers=range(1, 4), budget_per_dimension=100, n_runs=5`. That took about 3.5 minutes on a 16-core PC, almost all of it for the features.

[modular_ela.py](modular_ela.py) runs the test with these 55 features and the algorithms of ModularCMAES and ModDE above. It needs pflacco, ModularCMAES, and ModDE: `pip install rubberruler pflacco==1.2.2 modcma==1.2.0 modde==0.0.4`.

### DeepELA

DeepELA is not on PyPI. Fetch its repository at a fixed commit, install what it imports, and put the folder on `PYTHONPATH`:

```sh
commit=c0368677461f96a17536111ee947bf111e8b8425
git init deep_ela
git -C deep_ela fetch --depth 1 https://github.com/mvseiler/deep_ela $commit
git -C deep_ela checkout FETCH_HEAD
pip install torch lightning einops pyyaml
export PYTHONPATH="$PWD/deep_ela"
```

- Tested with torch 2.14.0 and lightning 2.6.6 on Python 3.11 and 3.12, so it can share an environment with pflacco.
- Fix the commit: the features change from commit to commit. The last four commits changed how the coordinates and objective values reach the model, and the stride.
- Do not use the repository's `install.sh`: it uninstalls pflacco and builds DeepELA into the latest pflacco from GitHub.
- On Linux, torch from PyPI comes with the CUDA libraries, several GB. If you use only the CPU, install torch as [PyTorch's page](https://pytorch.org/get-started/locally/) shows for the CPU.

```python
import deep_ela  # github.com/mvseiler/deep_ela at c036867, on PYTHONPATH
import torch

model = deep_ela.load_large_50d_v1(stride=1)


def deep_ela_features(X, y):
    if X.shape[1] > 11:
        raise ValueError(
            f"DeepELA's large models take at most 11 dimensions, got {X.shape[1]}"
        )
    torch.manual_seed(0)  # predict() permutes the input slots with torch's global RNG
    return model(X, y)
```

Set torch's seed before each call. DeepELA averages its features over 10 random orders of its input slots, drawn from torch's global random numbers, so without the seed they change from call to call. With the seed, they repeat bit for bit on the same machine, whatever the number of threads or the Python version; on another CPU they can differ in the last bit of a float32.

The large models put the coordinates and the objective value into 12 slots, so they take d ≤ 11. At d = 12 they run without an error, but the objective value shares a slot with the last coordinate. So the extractor raises an error above d = 11, and `run_test` stops on it before the long work.

DeepELA's README asks for input without preprocessing. Value-based features come out almost the same as for the raw input, as DeepELA standardizes each column itself. How far rank-based features differ from them is what rubberruler measures.

[deepela.py](deepela.py) runs the test with these 48 features and the quickstart's portfolio and predictor. With an NVIDIA GPU, pass `device="cuda"` to `load_large_50d_v1`: on an RTX 4070 Ti the features of the default settings then took about a minute instead of about 30, with values up to about 10⁻⁵ away from the CPU's.

### An extractor of another shape

A function of another shape joins in one line. Here yours takes `(y, X)` and returns a pandas Series:

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

To run it, pass `feature_extractor` in place of `simple_features` in [quickstart.py](quickstart.py).

## Predictors

Feature values that differ only by rounding are made equal before the predictor fits and predicts. Rounding can differ from one machine to another, and this keeps it from changing what the predictor learns.

Set the predictor's own seed, such as `random_state=0`. rubberruler does not touch it, and warns under the card if `random_state` is None.

### A pipeline that fills NaN in

If your features can be NaN, fill them in a scikit-learn pipeline:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

predictor = make_pipeline(SimpleImputer(), RandomForestRegressor(random_state=0))
```

[modular_ela.py](modular_ela.py) uses this pipeline for pflacco's features.

### PyTorch

Tested with torch 2.14.0 on Python 3.12:

```sh
pip install torch
```

Any object with `fit` and `predict` works, such as a small network:

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

To run it, pass `TorchRegressor()` as the predictor in [quickstart.py](quickstart.py).

## Another predictor on the same data

The result keeps the instances, the performance, and the features, so you can score another predictor without running the algorithms again:

```python
from sklearn.ensemble import GradientBoostingRegressor

features = result.features["rank"]
fitted = rr.fit_predictor(
    GradientBoostingRegressor(random_state=0),
    features,
    result.performance,
    instances=result.split.train,
)
r2 = rr.r2_score(
    fitted, features, result.performance, instances=result.split.transformed_test
)
```

Here `result` is the one from [quickstart.py](quickstart.py), and `r2` is the R² on the transformed test instances, the number that the card compares.

## Augmenting training data

If the card says rank but you want to keep value-based features, train the predictor on transformed copies of your own problems too, so that it learns features under many rubber rulers. `rr.transform_objective` applies a random rubber ruler to any objective: it returns g(f(x)), where g is built as for BBOB in [How it works](../README.md#how-it-works), but rescales f by the smallest and the largest of 10,000 random values of f. Pass a different `seed` for each copy.

This example makes ten copies of the Rastrigin function and compares two features of the quickstart on the original and on the copies:

```python
import numpy as np
from scipy.stats import kurtosis, rankdata

import rubberruler as rr


def rastrigin(x):
    return 10 * x.shape[-1] + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=-1)


d = 5
lower_bounds = np.full(d, -5.12)
upper_bounds = np.full(d, 5.12)
copies = [
    rr.transform_objective(rastrigin, lower_bounds, upper_bounds, seed=k)
    for k in range(10)
]


# Two features of the quickstart, with y value-based or rank-based as in rubberruler.
def features(X, y, feature_type):
    if feature_type == "value":
        y = (y - y.min()) / (y.max() - y.min())
    else:
        y = rankdata(y) / len(y)
    design = np.hstack([np.ones((len(y), 1)), X, X**2])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    quadratic_r2 = 1 - np.var(y - design @ coefficients) / np.var(y)
    return {"kurtosis": kurtosis(y), "quadratic_r2": quadratic_r2}


X = np.random.default_rng(0).uniform(lower_bounds, upper_bounds, size=(50 * d, d))
for feature_type in ("value", "rank"):
    original = features(X, rastrigin(X), feature_type)
    transformed = [features(X, copy(X), feature_type) for copy in copies]
    for name, value in original.items():
        values = [copy_features[name] for copy_features in transformed]
        print(
            f"{feature_type:5} {name:12}  original {value:6.3f}"
            f"  copies {min(values):6.3f} to {max(values):6.3f}"
        )
```

It prints:

```text
value kurtosis      original -0.233  copies -0.920 to  4.374
value quadratic_r2  original  0.562  copies  0.422 to  0.566
rank  kurtosis      original -1.200  copies -1.200 to -1.200
rank  quadratic_r2  original  0.526  copies  0.526 to  0.526
```

The value-based features spread over the copies, while the rank-based ones stay exactly the same, because g keeps the order of the values. Measure your algorithms on the copies as on the originals and add them to the training data.

Values better than all of the random ones, near the optimum, are only rescaled, not stretched; the random sample points that features are computed from mostly lie in the stretched range.
