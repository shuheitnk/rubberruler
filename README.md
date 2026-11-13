# rubberruler

rubberruler tells you whether your algorithm-selection features should be computed value-based (from the objective values) or rank-based (from their ranks).

<img alt="R² of value-based and rank-based features: on the test instances value 0.41 and rank 0.30; on the transformed test instances value 0.26 and rank 0.31; a dashed line marks the R² without features, 0.13 and 0.14" src="https://raw.githubusercontent.com/shuheitnk/rubberruler/main/assets/rubber_ruler_test.png" width="560">

## Installation

```sh
pip install rubberruler
```

rubberruler needs Python 3.11 or later.

## Quickstart

```sh
pip install rubberruler scikit-learn
curl -O https://raw.githubusercontent.com/shuheitnk/rubberruler/main/examples/example1.py
python example1.py
```

The Quickstart is [example1.py](https://github.com/shuheitnk/rubberruler/blob/main/examples/example1.py), the first of the [examples](https://github.com/shuheitnk/rubberruler/blob/main/examples/README.md). It defines a portfolio of three SciPy optimizers, a feature extractor of four features, and a random forest, and ends with the test:

```python
predictor = RandomForestRegressor(random_state=0)

result = rr.run_test(
    portfolio,
    simple_features,
    predictor,
    # Small settings for a first try.
    # Without them, run_test takes about an hour on a 16-core PC.
    dimension=2,
    budget_per_dimension=100,
    n_runs=5,
)
print(result)
```

It takes about half a minute on 24 CPUs and prints the card:

```text
Rubber Ruler Test: use RANK-based features
  R^2 on the transformed test instances: rank 0.31 > value 0.26

                       R^2 on                       R^2 on
feature type   test instances   transformed test instances   degradation
value                    0.41                         0.26          0.14
rank                     0.30                         0.31         -0.01
no features              0.13                         0.14         -0.01

With value-based features, the predictor does not generalize to
strictly increasing transformations of the objective.

3 algorithms, 2400 training instances, 240 test instances,
240 transformed test instances
seed=42, suite=bbob, dimension=2 (default 10),
budget_per_dimension=100 (default 1000), n_runs=5 (default 31)
```

The first line is the answer: the feature type with the larger R² on the transformed test instances. The degradation is the drop in R² from the test to the transformed test instances. The row "no features" is a reference: features that help the predictor score above it.

## Use your own portfolio, feature extractor, and predictor

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor

import rubberruler as rr


# An algorithm minimizes problem(x) until rubberruler ends the run at the budget.
def random_search(problem):
    rng = np.random.default_rng(problem.seed)
    while True:
        problem(rng.uniform(problem.lower_bounds, problem.upper_bounds))


def local_search(problem):
    rng = np.random.default_rng(problem.seed)
    x = rng.uniform(problem.lower_bounds, problem.upper_bounds)
    fx = problem(x)
    while True:
        step = rng.normal(0, 0.5, problem.dimension)
        candidate = np.clip(x + step, problem.lower_bounds, problem.upper_bounds)
        if (value := problem(candidate)) <= fx:
            x, fx = candidate, value


# A feature extractor maps a sample X of shape (n, d) and its values y to features.
def my_features(X, y):
    return {"mean": float(np.mean(y)), "spread": float(np.std(y))}


portfolio = {"random-search": random_search, "local-search": local_search}
predictor = RandomForestRegressor(random_state=0)  # any regressor with fit and predict
result = rr.run_test(portfolio, my_features, predictor)
print(result)
```

[examples/README.md](https://github.com/shuheitnk/rubberruler/blob/main/examples/README.md) connects ModularCMAES, ModDE, pymoo, pycma, Nevergrad, pflacco, DeepELA, and PyTorch.

## Options

- [MA-BBOB](https://github.com/shuheitnk/rubberruler/blob/main/examples/README.md#ma-bbob): `rr.run_test(..., suite="ma-bbob")`.
- [example2.py](https://github.com/shuheitnk/rubberruler/blob/main/examples/README.md#example-2-your-own-objective-functions): your own objective functions, with `rr.run_test(..., instances=rr.objective_instances(objectives, split_seed=1))`.
- [Precomputed tables](https://github.com/shuheitnk/rubberruler/blob/main/examples/README.md#precomputed-tables): `rr.run_test(..., performance=result.performance, features=result.features)`.
- [example3.py](https://github.com/shuheitnk/rubberruler/blob/main/examples/README.md#example-3-augmentation-with-transformed-objectives): augmentation, with `rr.transform_objective`, which applies a random rubber ruler to any objective function.
- [example7.py](https://github.com/shuheitnk/rubberruler/blob/main/examples/README.md#example-7-a-feature-free-model-in-pytorch): a feature-free model, with an extractor that returns the sample itself.
- [Parallel computation and memory](https://github.com/shuheitnk/rubberruler/blob/main/examples/README.md#parallel-computation-and-memory): `rr.run_test(..., n_jobs=4)`.
- Settings: the defaults follow the paper (dimension 10, 50 x dimension sample points, a budget of 1000 x dimension evaluations, 31 runs, training instances 11-110 and test instances 1-10 on BBOB, and the same distribution of rubber rulers), except that the sample points are uniform random rather than a Latin hypercube and the test uses one seed rather than the paper's 11. Finished runs are cached in `cache_dir`, and `help(rr.run_test)` lists every setting.

## How it works

rubberruler takes training and test instances from BBOB, from MA-BBOB (`suite="ma-bbob"`), or from your own objective functions (the suite `"objectives"`, given with `instances=rr.objective_instances(...)`), and makes a transformed test instance from each test instance by passing its objective through a random strictly increasing function, the rubber ruler. It runs every algorithm on every instance and measures its performance J, the probability that one of its runs beats a run of another algorithm, with a tie counting as half. For each feature type, the extractor receives uniform random points and their values (value) or the ranks of their values (rank), both min-max normalized to [0, 1], and the predictor fitted on the training instances is scored by R² on the test and the transformed test instances. The answer is the feature type with the larger R² on the transformed test instances; a tie selects rank. The row "no features" predicts the mean J of each algorithm on the training instances, so it shows how much of R² needs no features.

## Citing

Please cite rubberruler with [CITATION.cff](https://github.com/shuheitnk/rubberruler/blob/main/CITATION.cff); "Cite this repository" on GitHub gives it in APA and BibTeX.

## License

[MIT](https://github.com/shuheitnk/rubberruler/blob/main/LICENSE)
