# rubberruler Specification

- Version: 1.1 (2026-10-14; version 1.0 of 2026-09-29, translated into English).
- This document is the single source of truth for the specification. The code, the README, and the tests follow it. A change to the specification is made in this document first.
- Each requirement has an ID (for example, `R-API-3`). The docstring of a test names the ID of the requirement that it checks.
- The reasons are in `decisions.md`, the construction and its pitfalls in `implementation.md`, and the grounds of the user experience in `ux-design.md`.

---

## 1. Purpose and Scope

### 1.1 Purpose

For any algorithm portfolio, any feature extractor, and any performance predictor, rubberruler answers, with the least effort for the user, the question "should the features be computed value-based or rank-based?"

### 1.2 Promise

The user only matches three input and output types (Chapter 4). Once the types match, rubberruler reports the coefficient of determination R² of performance prediction on test instances transformed by strictly increasing functions, for both value and rank, and the higher one is the answer.

### 1.3 Out of Scope

The following are not built. They are not added during implementation unless they are requested.

- Comparison of several extractors, diagnostics per feature, breakdowns by function class, output in paper formats (such as LaTeX), and figures (only the figure of the answer in 3.4 is built).
- A selector for automated algorithm selection (a function that selects the algorithm with the largest predicted value).
- The answer on problems other than BBOB (`transform_objective` transforms any objective function, but the answer is decided on BBOB), preprocessing other than value and rank, transformation families other than the spline family, and sampling other than uniform random sampling.
- A procedure that enlarges a training set as a whole (such as a function that multiplies instances). `transform_objective` transforms one objective function (D43).
- Presets of settings (such as `scale`), setting objects, and arguments that switch behavior (`verbose`, `dry_run`, `multi_output`, `on_*`, `run_timeout`, and so on).
- Precomputed data, demonstration functions, functions that inspect or delete the cache, and a function that runs only the pre-checks.
- Other forms of input (injection of arguments by name, lists of functions, factories of predictors, and so on).
- Input and output with extractors written in languages other than Python, and a documentation site (the first release has only the README, `examples/README.md`, and docstrings).

---

## 2. Terminology

The code, the docstrings, the error messages, the card, and the README use only the names in the right column of this table.

| Symbol in the paper | Term | Name in the code |
|---|---|---|
| p | problem instance | `ProblemInstance` (an element of an instance set) |
| (p as one run sees it) | problem | `Problem` (what an algorithm receives; the parameter is named `problem`) |
| P_train / P_test / P̃_test | training / test / transformed test instances | `split.train` / `split.test` / `split.transformed_test` |
| 𝒜 | algorithm portfolio | `portfolio` |
| J(p, a) | performance (mean probability of superiority) | `performance` |
| τ ∈ {value, rank} | feature type | `feature_type` |
| φ_p | features | `features` |
| predictor_τ | performance predictor | `predictor` |
| g_{ψ,p} | monotone transformation | `transform` |
| f (the user's objective function) | objective | `objective` (the parameter of `transform_objective`) |
| R²_τ(Q) | coefficient of determination | `r2_score(..., instances=Q)` |
| R²_τ(P_test) − R²_τ(P̃_test) | degradation | `degradation` |
| τ* | selected feature type | `selected_feature_type` |

- Names are not abbreviated (`dimension`, not `dim`).
- No public function starts with `test` and no public class starts with `Test` (pytest would collect them when a user imports them into a test).

---

## 3. Overview from the User's Side

### 3.1 Minimal Example (R-UX-1)

```python
import rubberruler as rr
from sklearn.ensemble import RandomForestRegressor

portfolio = {"random-search": random_search, "de": de, "nelder-mead": nelder_mead}
predictor = RandomForestRegressor(random_state=0)
result = rr.run_test(portfolio, simple_features, predictor)  # simple_features(X, y) -> dict[str, float]
print(result)
```

The three algorithms and `simple_features` are defined in the quickstart of the README (`examples/quickstart.py`). The user needs to remember three things.

1. The objects to pass, with matching types, are `portfolio`, `feature_extractor`, and `predictor` (Chapter 4).
2. The function to call is `rr.run_test`.
3. The line to read is the first line of the card.

### 3.2 Card (R-UX-2)

`print(result)` writes the following form to stdout.

```
Rubber Ruler Test: use RANK-based features
  R^2 on transformed test: rank 0.63 > value 0.42

feature type   R^2 (test)   R^2 (transformed test)   degradation
value                0.71                     0.42          0.29
rank                 0.66                     0.63          0.03

With value-based features, the predictor does not generalize to
monotone transformations of the objective.

3 algorithms, 2400 train, 240 test, 240 transformed test
seed=42
```

- The order is answer, evidence, meaning, and conditions. Every line has at most 80 characters. The text of `print(result)` has no color.
- The characters are ASCII only. R² is written `R^2` (D35; the same holds for the plan line, the warnings, the error messages, and the docstrings of the public functions).
- Numbers have two decimal places and are right-aligned. When the R² values of the two feature types on the transformed test instances are equal after rounding to two decimal places, all three columns of the table and the second line use three decimal places.
- Comparisons always use the values before rounding. The number of displayed digits is never used for a comparison.
- `RANK` or `VALUE` in the first line is `selected_feature_type` in upper case. The second line names the selected feature type first and joins the two values with `>` if the values before rounding differ, and with `=` if they are equal.
- The two lines of meaning depend on the selected feature type.
  - For rank: `With value-based features, the predictor does not generalize to` / `monotone transformations of the objective.`
  - For value: `Value-based features still help the predictor after` / `monotone transformations of the objective.`
- If the absolute difference δ of the two R² values on the transformed test instances (before rounding) is below 0.01, the line `The difference is {δ:.4f} (below 0.01), so treat this answer as weak.` follows the meaning.
- If the larger R² on the transformed test instances (before rounding) is below 0.1, the two lines `Both R^2 values are below 0.1, so this answer is weak.` / `Check the extractor and the predictor first.` follow the meaning (after the line on the difference, if present). With either feature type, the predictor then explains almost none of the performance, so the answer is weak, and the extractor and the predictor should be revised first (D36).
- The last two lines are the conditions. The first is the number of algorithms and the size of each instance set, joined by `, `. The second is `seed` and the settings that differ from their defaults, in the order of the table in 5.2, joined by `, `. A line longer than 80 characters is broken at a `, `, and a broken line keeps the `,` at its end (at most 80 characters including the `,`).
  - `seed` is always shown: `seed=42` for the default, and `seed=7 (default 42)` otherwise.
  - An integer setting: `dimension=20 (default 10)`.
  - A sequence of instance numbers is converted to a sorted tuple of integers before it is compared with the default. A contiguous sequence is shown as `train_instance_numbers=11-20 (default 11-110)`, a non-contiguous one as `train_instance_numbers=12 numbers (default 11-110)`, and a single number as `test_instance_numbers=5 (default 1-10)`.
  - `n_jobs` and `cache_dir` do not change the result and are not shown.
- If there are warnings, their counts come last, together (5.6).
- `rich.print(result)` draws the same parts in a panel titled `Rubber Ruler Test` (the `__rich__` method of the result): the answer in bold green, the table with the row of the selected feature type in bold, the cautions (the lines on a weak answer) and the warnings in yellow, and the conditions dimmed. The text of `print(result)` does not change.

### 3.3 Decision Rule (R-DEC-1)

- `selected_feature_type` is the feature type with the larger R² on the transformed test instances.
- A tie selects `"rank"` (rank-based features do not change under the transformation, so nothing is lost in a tie).
- The degradation is only shown in the table and is not used for the decision.
- If R² becomes `nan` (6.6; all values of J on the transformed test instances are equal), there is no answer, and `run_test` stops with a `RubberRulerError`.

### 3.4 Figure of the Answer (R-FIG-1)

At the end, `run_test` writes the R² values of the card as a bar chart in PNG and SVG. The figure can be pasted into slides and papers as it is. It adds no public name, no argument of `run_test`, and no attribute of `RubberRulerResult` (D26).

- The files are `rubber_ruler_test.png` and `rubber_ruler_test.svg` in `cache_dir` (overwritten every time). Nothing is written if `cache_dir=None`.
- After writing, one line in the form of the plan line goes to stderr: `rubberruler: figure saved to .rubberruler/rubber_ruler_test.png and .svg` (with the actual `cache_dir`).
- The card (`str(result)`) does not change.
- If the figure cannot be written (for example, the folder is not writable), the result is kept: the logger `rubberruler` receives the WARNING `could not save the figure in .rubberruler/: {type of the exception}: {message}` (with the actual `cache_dir`), and the computation continues. The warning is not added to the Warnings of the card.

Form of the figure:

- The data are `r2_test` and `r2_transformed_test` of `result.table` (value and rank). The values before rounding are used; they are rounded to two decimal places only in the labels. A value of `nan` (all values of J on the test instances are equal) gets neither a bar nor a label.
- The blocks are, from left to right, `test` and `transformed test`. Each block has the bars `value` and `rank`, from left to right. A bar is at `block * 3 + slot` (slot 0 or 1) with width 0.74.
- Colors (Okabe-Ito): value `#D55E00`, rank `#0072B2`. The fill is the color moved 35% toward white (value `#E49659`, rank `#59A3CD`); the edge is the original color with line width 1.8.
- The value is written at the end of each bar (above a positive bar, below a negative one): two decimal places, bold, in the original color of the bar, 24 pt. A negative number uses the minus sign U+2212, not a hyphen (a negative number that rounds to 0.00 is written `0.00`, as on the card). The gap between a bar and its value is 2% of the range of the y axis.
- The value labels have a white background (`bbox={"facecolor": "white", "edgecolor": "none", "boxstyle": "square,pad=0.1"}`), so that the reference line at R² = 1 and the grid lines break behind a label. The labels are drawn above the lines.
- Reference lines: horizontal lines at R² = 0 (a model that predicts the training mean) and R² = 1 (perfect prediction), color `#1B1B1B`, line width 1.4, above the bars.
- The horizontal grid lines are `#E0E0E0` with line width 1.1, behind the bars. All spines and tick marks (length=0) are removed.
- Range of the y axis: the top is `max(1, largest value + 0.08) + 0.06`. The bottom is 0 if no value is negative. Otherwise, it is the smallest displayed value low (clipped to −1 if it is below −1), lowered by 14% of `(top − low)` (so that the label of a negative bar does not overlap the bar names). A bar below −1 stops at −1, and its label shows the true value (for example, `−3.42`).
- Text: the y-axis label `$R^2$` at 24 pt, the y tick labels at 19.2 pt, the bar names (minor x ticks) at 20 pt, and the block names (major x ticks) at 26 pt below the bar names (pad 38). There is no title.
- The size is 8.4 × 5.6 inches, the background is white, the PNG has 200 dpi, and `tight_layout` is used. The fonts are Arial, Liberation Sans, Helvetica, and DejaVu Sans, in this order. The SVG keeps the text as text (`svg.fonttype = "none"`).

---

## 4. The Three Types

The user matches only this table.

| Object | Input | Output |
|---|---|---|
| each algorithm in `portfolio` | `problem: rr.Problem` | none (the return value is not used) |
| `feature_extractor` | `X`: a float64 array of shape (n, d), `y`: a float64 array of shape (n,) | `dict[str, float]` (any `Mapping`) |
| `predictor` | `fit(X, y)`: `X` is a float64 array of shape (m, k), `y` is a float64 array of shape (m,) | `predict(X)`: a float64 array of shape (m,) |

d is the dimension, n the number of sample points for the features, m the number of instances, and k the number of features. `portfolio` is a `Mapping[str, algorithm]`, whose keys are the algorithm names.

### 4.1 Algorithms and `Problem` (R-TYPE-1)

An algorithm is a function that receives one `Problem`. It evaluates the objective function by calling `problem`. Its return value is not used.

| Attribute | Type | Meaning |
|---|---|---|
| `problem(x)` | `float` for x of shape (d,); an `NDArray[float64]` of shape (n,) for x of shape (n, d) | The value of the objective function; on a transformed instance, the transformed value |
| `problem.dimension` | `int` | d |
| `problem.lower_bounds`, `problem.upper_bounds` | `NDArray[float64]` of shape (d,) | The search range (−5 and 5 on BBOB) |
| `problem.budget` | `int` | The maximum number of evaluations of this run |
| `problem.seed` | `int` | The random seed of this run (from 0 to 2³²−1) |
| `problem.evaluations` | `int` | The number of evaluations so far |
| `problem.bounds.lb`, `problem.bounds.ub` | as above | The names of IOHexperimenter |
| `problem.meta_data.n_variables` | `int` | The name of IOHexperimenter |
| `problem.state.evaluations` | `int` | The name of IOHexperimenter |
| `problem.state.optimum_found` | `bool` | The name of IOHexperimenter. Always `False` (whether the optimum was reached is not revealed either) |

- The performance uses the smallest value that `problem` returned within the budget. A run that returns without any evaluation gets +inf and counts as a failure.
- A call to `problem` after the budget is spent raises an internal exception. The exception derives from `BaseException` and is not public. rubberruler catches it and ends the run normally.
- The optimal value f_min is not revealed.
- A function of another form is wrapped by the user in one line: `"my-es": lambda p: my_es(p, p.dimension, p.budget, p.seed)`.
- ModularCMAES (the C++ version `c_maes` of modcma 1.2.0) is called as `c_maes.ModularCMAES(settings)(problem)` with the settings `c_maes.settings_from_dict(problem.dimension, budget=problem.budget, lb=problem.lower_bounds, ub=problem.upper_bounds)`, after `c_maes.utils.set_seed(problem.seed % 2**31)` (it accepts only seeds below 2³¹) (`examples/README.md` and `examples/modular_ela.py`).
- ModDE (modde 0.0.4) is called as `ModularDE(problem, budget=problem.budget, seed=problem.seed, lb=problem.lower_bounds.reshape(-1, 1), ub=problem.upper_bounds.reshape(-1, 1)).run()`. The bounds are passed with shape (d, 1). It reads `problem.state.optimum_found` in every generation (D28).
- An algorithm of pymoo 0.6.2 runs with `problem` wrapped in a subclass of `pymoo.core.problem.Problem` (`n_var=problem.dimension`, the bounds as `xl` and `xu`, and `out["F"] = problem(X)` in `_evaluate`; pymoo passes the solutions together with shape (n, d)), through `pymoo.optimize.minimize(view, algorithm, ("n_eval", problem.budget), seed=problem.seed, verbose=False)` (`examples/README.md` and `examples/pymoo_portfolio.py` import it as `pymoo_minimize`, so that it does not collide with SciPy's `minimize` in the quickstart; D31).

### 4.2 Feature Extractors (R-TYPE-2)

`feature_extractor(X, y) -> dict[str, float]`

- X is the sample, passed in the search range (on BBOB, [−5, 5]). y holds the objective values preprocessed for the feature type and lies in [0, 1] (6.4).
- One extractor serves both feature types.
- The keys of the dict are the same set on every instance. The order of the keys is that of the first instance computed. A value that cannot be computed is `float("nan")`.
- A value may be anything that `float()` converts (`int`, `numpy.float64`, `numpy.int64`, and so on). Otherwise, it is an error.
- A value of ±inf is converted to NaN by rubberruler on receipt and counted in the same warning line as NaN (5.6; D30).
- A function of another form is connected by a one-line lambda. For example, a function `my_features` that takes `(y, X)` and returns a `pandas.Series` is connected by `lambda X, y: dict(my_features(y, X))` (the same lambda appears in `examples/README.md` and in the tests).
- The ELA features of pflacco 1.2.2 are returned by merging the dicts of the feature sets with `|` and leaving out the names that contain `.costs_` (such as the seconds of computation, which change from run to run) (`examples/README.md` and `examples/modular_ela.py`). pflacco is not a dependency (D29).
- DeepELA (commit c0368677461f96a17536111ee947bf111e8b8425 of github.com/mvseiler/deep_ela, placed on PYTHONPATH) is used by loading the model with `deep_ela.load_large_50d_v1(stride=1)` and calling `model(X, y)` after `torch.manual_seed(0)` on every call; it returns 48 features. The large model handles only d ≤ 11, so for d > 11 the extractor raises a ValueError, and the pre-check stops (`examples/README.md` and `examples/deepela.py`). DeepELA is neither a dependency nor an extra (D33).

### 4.3 Performance Predictors (R-TYPE-3)

An unfitted object with `fit(X, y)` and `predict(X)` (the scikit-learn form).

- rubberruler copies it for each feature type and each algorithm and fits the copies. The copy is made with `sklearn.base.clone` if the object has `get_params`, and with `copy.deepcopy` otherwise. The object passed in is not modified.
- X is a two-dimensional float64 array (rows are instances, columns are features, in the order of the keys of 4.2). Values that differ only by rounding are unified before X is passed (6.6). y is the performance of one algorithm. No DataFrame is passed.
- The X and y given to the predictor are writable copies on every call of `fit` and `predict` (a predictor that modifies them in place affects neither the fitting for other algorithms nor the original DataFrame).
- The return value of `predict` only needs to convert to an array of shape (m,) (shape (m, 1) is flattened). The predicted values are clipped to [0, 1] before scoring.

### 4.4 Type Checks (R-TYPE-4)

- **Static**: the parameters of `run_test` and of the component functions have type annotations. Without any import by the user, pyright or mypy points out a mismatch at the call.

  ```python
  Algorithm = Callable[[Problem], object]
  FeatureExtractor = Callable[[NDArray[np.float64], NDArray[np.float64]], Mapping[str, float]]
  ```

  The type of the predictor parameter is `object`. scikit-learn's `Pipeline` defines `predict` with the `available_if` descriptor, so pyright judges that it does not match a `Protocol` with `fit` and `predict` (confirmed with pyright and scikit-learn 1.9.1 on 2026-09-29). The type of the predictor is checked at run time by the pre-check.

- **At run time**: before the heavy computation, the pre-checks (5.5) check the same types and stop with a `RubberRulerError` if they do not match.

---

## 5. Public API

### 5.1 Public Names (R-API-1)

`rubberruler.__all__` contains exactly these 10 names.

```
run_test
bbob_instances, transform, measure_performance, compute_features, fit_predictor, r2_score
transform_objective
Problem, RubberRulerError
```

- `transform_objective` is not used for the answer. It is a tool with which the user, after reading the answer of the card, augments their own training data (6.7; D43).
- `ProblemInstance`, `InstanceSplit`, `FittedPredictor`, and `RubberRulerResult` appear only as return values and are not public (the user does not need to import them). Every module name starts with `_`.
- Type aliases such as `Algorithm` and `FeatureExtractor` are not public either.

### 5.2 `rr.run_test` (R-API-2)

```python
def run_test(
    portfolio: Mapping[str, Algorithm],
    feature_extractor: FeatureExtractor,
    predictor: object,
    *,
    dimension: int = 10,
    train_instance_numbers: Sequence[int] = range(11, 111),
    test_instance_numbers: Sequence[int] = range(1, 11),
    budget_per_dimension: int = 1000,
    n_runs: int = 31,
    n_samples_per_dimension: int = 50,
    seed: int = 42,
    n_jobs: int = -1,
    cache_dir: str | os.PathLike[str] | None = ".rubberruler",
) -> RubberRulerResult
```

| Argument | Meaning | Accepted values (otherwise `RubberRulerError`) |
|---|---|---|
| `portfolio` | Algorithm names mapped to algorithms | At least 2; the names are non-empty `str` |
| `dimension` | d | An `int` of at least 2 |
| `train_instance_numbers` | BBOB instance numbers of the training instances | `int` values of at least 1, without repetition, not empty |
| `test_instance_numbers` | Instance numbers of the test instances | As above, and disjoint from the training ones |
| `budget_per_dimension` | The maximum number of evaluations of one run divided by d | At least 1 |
| `n_runs` | Runs per pair of instance and algorithm | At least 1 |
| `n_samples_per_dimension` | Sample points for the features divided by d | At least 1 (and n = d × this value is at least 2) |
| `seed` | The master seed | From 0 to 2³²−1 |
| `n_jobs` | The number of parallel workers; −1 uses every CPU | A nonzero `int` |
| `cache_dir` | The folder that stores the results of runs; `None` caches nothing | A writable path |

- All 24 BBOB functions are used. The number of instances is (training + test + transformed test) × 24.
- The defaults are the experimental settings of the paper (dimension, instances, budget, number of runs, and number of sample points for the features). rubberruler is independent of the experimental code of the paper and does not aim to match its random values or its numbers (D24).
- Order of processing: checks of the settings → pre-checks (5.5) → preparing instances → measuring performance → computing features → fitting predictors and scoring → result → writing the figure of the answer (3.4; when `cache_dir` is set).
- Output on stderr (R-UX-3): the plan line, the four stages with their progress, and the line on the written figure (3.4). The progress is drawn with rich.

  - In a terminal or a notebook (rich's `Console.is_interactive` or `is_jupyter`), each stage is drawn: a stage with a count shows a spinner, a bar of 20 characters, the percentage, the count, and the time left while it runs, and the time it took when it ends; a stage without a count shows a spinner while it runs. Each stage ends with a check mark. The prefix `rubberruler:` of a message is bold. With a console encoding that does not start with `utf`, rich draws an ASCII spinner and an ASCII bar, and the check mark is a space.

    ```
    rubberruler: 43,200 runs (0 cached, 43,200 to run) in .rubberruler/
    ✔ [1/4] Preparing instances
    ✔ [2/4] Measuring performance ━━━━━━━━━━━━━━━━━━━━ 100% 43,200/43,200 0:00:21
    ✔ [3/4] Computing features ━━━━━━━━━━━━━━━━━━━━ 100% 5,760/5,760 0:00:04
    ✔ [4/4] Fitting predictors and scoring
    rubberruler: figure saved to .rubberruler/rubber_ruler_test.png and .svg
    ```

  - Elsewhere (for example, when stderr is redirected to a file), each stage writes plain ASCII lines: a stage without a count writes its name when it starts; a stage with a count writes its progress when it starts, at most every 30 seconds, and when it ends (also on an exception, unless the last line shows the same count).

    ```
    rubberruler: 43,200 runs (0 cached, 43,200 to run) in .rubberruler/
    [1/4] Preparing instances
    [2/4] Measuring performance   0% 0/43,200 [00:00<?]
    [2/4] Measuring performance 100% 43,200/43,200 [00:19<00:00]
    [3/4] Computing features   0% 0/5,760 [00:00<?]
    [3/4] Computing features 100% 5,760/5,760 [00:04<00:00]
    [4/4] Fitting predictors and scoring
    rubberruler: figure saved to .rubberruler/rubber_ruler_test.png and .svg
    ```

  - The environment variable `RUBBERRULER_NO_PROGRESS` with any value except `0` hides the bars; the names of the stages are still written (there is no argument for this).

  The plan line with `cache_dir=None`: `rubberruler: 267,840 runs (0 cached, 267,840 to run) without a cache`. `measure_performance` writes a plan line of the same form.

  When some runs are cached and others are not, the line after the plan line names the algorithms with runs left and their counts (in the order of the portfolio, joined by `, `): `rubberruler: to run: de 89,280`. The line shows which algorithms are considered modified (a change of a comment changes the source; 7.2).

- On Ctrl-C, one line states the number of saved runs and how to resume, and `KeyboardInterrupt` is raised again (R-UX-4): `rubberruler: stopped. 240,000 of 267,840 runs are saved. Run the same call again to resume.` With `cache_dir=None`: `rubberruler: stopped. No runs are saved because cache_dir is None.` In a terminal, these lines are yellow.

### 5.3 Result `RubberRulerResult` (R-API-3)

An immutable dataclass.

| Attribute | Type | Content |
|---|---|---|
| `selected_feature_type` | `Literal["value", "rank"]` | The answer (3.3) |
| `table` | `pandas.DataFrame` | Rows `"value"` and `"rank"`; columns `r2_test`, `r2_transformed_test`, and `degradation` |
| `split` | `InstanceSplit` | The instance sets used |
| `performance` | `pandas.DataFrame` | Rows are `instance_id`, columns are algorithm names, values are J |
| `features` | `dict[str, pandas.DataFrame]` | Keys are feature types; rows are `instance_id`, columns are feature names |
| `config` | `RunConfig` (a `TypedDict`, not public) | The seven settings actually used (`dimension: int`, `train_instance_numbers: tuple[int, ...]`, `test_instance_numbers: tuple[int, ...]`, `budget_per_dimension: int`, `n_runs: int`, `n_samples_per_dimension: int`, `seed: int`). `rr.run_test(..., **result.config)` repeats the same settings, and pyright accepts it |

- `str(result)` and `repr(result)` are the card (3.2). Jupyter shows the same text. `rich.print(result)` draws the panel of 3.2 (`__rich__`).
- `split`, `performance`, and `features` can be passed to the component functions as they are.

### 5.4 Component Functions (R-API-4)

`run_test` calls these in order. They serve to try another extractor or predictor on the same performance data.

```python
def bbob_instances(
    *, dimension: int = 10,
    train_instance_numbers: Sequence[int] = range(11, 111),
    test_instance_numbers: Sequence[int] = range(1, 11),
    seed: int = 42,
) -> InstanceSplit

def transform(instance: ProblemInstance) -> ProblemInstance

def measure_performance(
    portfolio: Mapping[str, Algorithm], instances: Iterable[ProblemInstance], *,
    budget_per_dimension: int = 1000, n_runs: int = 31,
    n_jobs: int = -1, cache_dir: str | os.PathLike[str] | None = ".rubberruler",
) -> pandas.DataFrame

def compute_features(
    feature_extractor: FeatureExtractor, instances: Iterable[ProblemInstance],
    feature_type: Literal["value", "rank"], *,
    n_samples_per_dimension: int = 50,
) -> pandas.DataFrame

def fit_predictor(
    predictor: object, features: pandas.DataFrame, performance: pandas.DataFrame, *,
    instances: Iterable[ProblemInstance],
) -> FittedPredictor

def r2_score(
    fitted: FittedPredictor, features: pandas.DataFrame, performance: pandas.DataFrame, *,
    instances: Iterable[ProblemInstance],
) -> float

def transform_objective(
    objective: Callable[[NDArray[np.float64]], float],
    lower_bounds: ArrayLike, upper_bounds: ArrayLike, *,
    seed: int, n_samples: int = 10_000,
) -> Callable[[NDArray[np.float64]], float | NDArray[np.float64]]
```

- `InstanceSplit` has `train`, `test`, and `transformed_test` (each a `tuple[ProblemInstance, ...]`). Iterating over it yields all instances of the three in order, so `measure_performance(portfolio, split)` is valid.
- `ProblemInstance` is read-only and has `instance_id`, `function_id`, `instance_number`, `dimension`, `lower_bounds`, `upper_bounds`, `seed` (the master seed), `is_transformed`, and `original_instance_id`. It is not callable (only the `Problem` of an algorithm evaluates the objective function).
- `instance_id` is `f"f{function_id:02d}_i{instance_number:03d}_d{dimension}"` (for example, `f05_i003_d10` and `f05_i1000_d2`). A transformed instance appends `~transformed` (for example, `f05_i003_d10~transformed`). `original_instance_id` is the id before the transformation (the instance's own id for an original instance).
- `bbob_instances(seed=...)` stores the master seed in the `seed` of every `ProblemInstance`. The other components use the `seed` of the instance.
- `transform` applies a transformation determined only by (master seed, function, instance number, dimension). The same instance always receives the same transformation. Transforming a transformed instance again is an error.
- `compute_features` stores the feature type in `DataFrame.attrs["feature_type"]` of its return value. `fit_predictor` records it in the `FittedPredictor`, and `r2_score` raises an error if the feature type or the columns of the features differ from those at fitting.
- `fit_predictor` fits one predictor per algorithm (4.3). It uses only the rows of `instances`.
- `transform_objective` calls neither `run_test` nor the other components. It receives the user's objective function, not a BBOB instance, and returns a function that returns g(objective(x)) (6.7). `seed` is a required argument without a default (with a default, `[rr.transform_objective(f, lb, ub) for _ in range(10)]` would return the same transformation 10 times).

### 5.5 Pre-checks (R-CHK-1)

The following is checked automatically before the heavy computation. `run_test` checks all three inputs, `measure_performance` the algorithms, `compute_features` the extractor, and `fit_predictor` the predictor. There is no function that runs only the checks and no argument that disables them.

| Object | Checked |
|---|---|
| Each algorithm | On the first of the given instances (the first training instance in `run_test`), a run with budget 10·d and seed 0 ends within 60 seconds, evaluates at least once, and raises no exception. If `n_jobs` is not 1, the algorithm can be sent to another process (cloudpickle can serialize it) |
| Feature extractor | On the first given instance of each function (in `run_test`, the first training instance of each of the 24 functions), it can be called for both value and rank, returns a `Mapping` with `str` keys and values that `float()` converts, and returns the same keys every time (D40) |
| Predictor | It has `fit` and `predict` and is not the class itself. `fit` and `predict` work on small dummy data (10 × 3) and return values that convert to shape (10,) |

- The time limit of 60 seconds applies on every OS and from every thread. The run of the pre-check runs in another thread of this process, and the calling thread waits and checks the time every 0.1 seconds (so Ctrl-C also arrives at once). Nothing is sent to another process, so the limit applies with `n_jobs=1` and to algorithms that cloudpickle cannot serialize. After 60 seconds, the `problem` of that run raises an exception on every further evaluation, an exception is also sent to the thread of the run, and a `RubberRulerError` is raised without waiting for the run to stop (D37). The thread of the run is a daemon, so a script ends right after the error. The exception sent to the thread does not reach a call inside C, such as `time.sleep`, so in a process that continues, such as Jupyter, the thread keeps running in the background until the C call returns.

### 5.6 Errors and Warnings (R-ERR-1)

- There is one exception, `RubberRulerError` (a subclass of `Exception`). `RubberRulerError.__module__ = "rubberruler"`, so a traceback shows `rubberruler.RubberRulerError`.
- The message has three lines: what happened (and where) / `Why:` why it is a problem / `Fix:` how to fix it. "Where" contains whatever is known among the algorithm name, the `instance_id`, and the feature type. The length of a line is not limited. The text that rubberruler writes is ASCII only (D35; names and values given by the user are written as they are).

  ```
  rubberruler.RubberRulerError: feature_extractor returned 14 features on f05_i003_d10 (rank) but 16 on f01_i011_d10 (rank).
  Why: every instance must give the same features.
  Fix: return a dict with fixed keys; use float("nan") for values you cannot compute.
  ```

- **Stop** (`RubberRulerError`): a setting does not match the table in 5.2; a pre-check fails; the keys of the features differ between instances; the extractor raised an exception (the message names the instance, and the original exception is kept in `__cause__`); the `fit` or `predict` of the predictor raised an exception (if there are NaN values, the message gives their count and suggests a `Pipeline` with `SimpleImputer`); R² on the transformed test instances is nan (3.3; the first line is `R^2 on the transformed test instances is nan, so there is no answer.`).
- **Failure of the import**: if `import rubberruler` cannot load the C++ part of ioh (on Windows without the Visual C++ Redistributable; ioh's wheel does not bundle MSVCP140.dll; the message contains `DLL load failed`), the result is an `ImportError` whose three lines say to install the X64 package from Microsoft's official page (https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist ). It is a failure of the import, so it is not a `RubberRulerError` (code written as `except ImportError` keeps working). Other ImportErrors pass unchanged.
- **Continue** (failure of a run): a run in which the algorithm raised an exception takes the smallest value so far. Without any evaluation, it takes +inf.
- **Warnings**: as in the following table. The component functions log them as WARNING to the logger `rubberruler`. `run_test` collects them during the run and writes them together below the card under `Warnings:`. The details of failed runs (algorithm name, `instance_id`, run number, type and message of the exception) go to the logger for the first 10 failures.

| Condition | Line below the card |
|---|---|
| k runs failed | `{k} runs failed; their best value so far was used (+inf if none) (see the log)`; for k = 1, `1 run failed; its best value so far was used (+inf if none) (see the log)` |
| k instances have features with NaN (±inf counts as NaN; 4.2) | `{k} instances have NaN features`; for k = 1, `1 instance has NaN features` |
| k instances have a constant y on the feature sample (value becomes a zero vector) | `{k} instances have a constant objective on the feature sample`; for k = 1, `1 instance has a constant objective on the feature sample` |
| A function identified by its name only because its source is not available | `{name}: cached by name only (source not available)` |
| A function identified by its name and source only because some values (such as closure values) could not be identified | `{name}: cached by name and source only (some values could not be hashed)` |
| A predictor whose `get_params()` has `random_state` equal to `None` | `predictor has random_state=None, so R^2 may change between calls` |
| In the component `r2_score`, all values of J are equal (R² is undefined; 6.6) | Not on the card. The logger receives `R^2 is undefined because all performance values are equal; returned nan` (in `run_test`, this is the "Stop" case above) |

Example below the card:

```
Warnings:
  2 runs failed; their best value so far was used (+inf if none) (see the log)
  12 instances have NaN features
```

---

## 6. Definitions of the Computation

### 6.1 Random Numbers (R-RNG-1)

- The only source of randomness is the master seed `seed`.
- The seed of each use is derived directly with `derive_seed(domain, seed, *keys)`. It depends neither on the number of workers nor on the order of execution.

  ```python
  def derive_seed(domain: str, seed: int, *keys: str | int) -> int:
      payload = json.dumps([domain, seed, *keys], ensure_ascii=False, separators=(",", ":"))
      digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=4).digest()
      return int.from_bytes(digest, "little")
  ```

  - The value lies from 0 to 2³²−1. It has 32 bits so that many libraries (`numpy.random.seed`, pycma, the `random_state` of scikit-learn) accept it as it is.
  - Integers in keys are converted to Python `int` before they are passed (convert NumPy integers with `int()`; `json.dumps` cannot handle them).
  - Python's `hash()` is not used (its value changes from process to process).

| Use | domain | keys | How the random numbers are drawn |
|---|---|---|---|
| Monotone transformation ψ | `"transform"` | `"bbob"`, function_id, instance_number, dimension | 6.3 |
| Uniform sample for f̂_max | `"fmax"` | `"bbob"`, function_id, instance_number, dimension | `rng.uniform(lower_bounds, upper_bounds, size=(10_000, d))` |
| Sample X for the features | `"features"` | `"bbob"`, function_id, instance_number, dimension | `rng.uniform(lower_bounds, upper_bounds, size=(n, d))` |
| ψ of `transform_objective` | `"transform"` | `"objective"` | 6.7 |
| Uniform sample of `transform_objective` | `"fmax"` | `"objective"` | `rng.uniform(lower_bounds, upper_bounds, size=(n_samples, d))` |
| Run of an algorithm | `"run"` | `"bbob"`, function_id, instance_number, dimension, algorithm name (`str`), run number | The value of `derive_seed` becomes `problem.seed` as it is |

- `rng = numpy.random.default_rng(derive_seed(domain, seed, *keys))`.
- Test values (fixed in `tests/`):

| Call | Value |
|---|---|
| `derive_seed("transform", 42, "bbob", 1, 1, 10)` | `2715082314` |
| `derive_seed("fmax", 42, "bbob", 1, 1, 10)` | `1678112591` |
| `derive_seed("features", 42, "bbob", 1, 1, 10)` | `1617016386` |
| `derive_seed("run", 42, "bbob", 1, 1, 10, "cma-es", 0)` | `374886278` |
| `derive_seed("transform", 42, "objective")` | `1546156558` |
| `derive_seed("fmax", 42, "objective")` | `3206986475` |

- The seed of a run of an algorithm is determined by the original instance and does not change with the transformation.
- Right before each run, `random.seed` and `numpy.random.seed` (and `torch.manual_seed` if torch has been imported) are called with the seed of the run (so that a user's algorithm that uses the global random generators is reproducible too).

### 6.2 Instances (R-INS-1)

- BBOB comes from `ioh`: `ioh.get_problem(function_id, instance_number, dimension, ioh.ProblemClass.BBOB)`.
- f_min is `problem.optimum.y`. f̂_max is the maximum over 10⁴ points drawn uniformly from the search range (with the `"fmax"` seed).

### 6.3 Monotone Transformation (R-TRF-1)

The transformation is built by the following steps, with random numbers drawn in this order. There is one transformation per instance, determined only by (master seed, function, instance number, dimension).

1. `rng = numpy.random.default_rng(derive_seed("transform", seed, "bbob", fid, iid, d))`
2. `m = rng.integers(5, 9)` (5 to 8)
3. Knot positions `t = numpy.linspace(0, 1, m)` and heights `s_j = rng.normal(0.0, 1.0, m)`
4. `s = scipy.interpolate.CubicSpline(t, s_j)` (not-a-knot)
5. On the equally spaced grid of 2001 points `z = numpy.linspace(0, 1, 2001)`, compute `g = scipy.integrate.cumulative_trapezoid(numpy.exp(s(z)), z, initial=0)` and set `g = g / g[-1]`. Assign `g[0] = 0` and `g[-1] = 1`
6. ψ is the `scipy.interpolate.PchipInterpolator` of (z_k, g_k)
7. g(y): `z = (y − f_min) / (f̂_max − f_min)`, `z = max(z, 0)`. ψ(z) if z ≤ 1, and z if z > 1

The objective function of a transformed instance is g(f(x)). The search space does not change.

Test values (seed=42, function 1, instance number 1, dimension 10; computed with NumPy 2.4, SciPy 1.17, and ioh 0.3.22):

| Quantity | Value | Tolerance |
|---|---|---|
| m | `5` | exact |
| ψ(0.25), ψ(0.5), ψ(0.75) | `0.14828703445665872`, `0.21005289625892246`, `0.35689515337847727` | absolute 1e-9 |
| f_min, f̂_max | `79.48`, `332.54400819810616` | absolute 1e-6 |

The properties ψ(0) = 0 and ψ(1) = 1 are checked with an absolute tolerance of 1e-12, because PCHIP evaluated at the last knot can return a value 1 ulp below 1 by rounding (the steps do not change).

### 6.4 Features (R-FEA-1)

- Sample X: n = n_samples_per_dimension × d points drawn uniformly from the search range. It is drawn once per instance, and the original and the transformed instance share the same X.
- y_raw: f(X) on an original instance and g(f(X)) on a transformed instance. f(X) is evaluated once.
- Preprocessing:
  - value: `(y_raw − min) / (max − min)`. If max = min, the result is a zero vector, and the instance is counted in a warning.
  - rank: `scipy.stats.rankdata(y_raw, method="average") / n`.
- `feature_extractor(X, y)` is called, and its result becomes one row.

### 6.5 Performance (R-PERF-1)

- Run: budget B = budget_per_dimension × d. The value is the smallest value seen within the budget (on a transformed instance, the transformed value). Failures follow 5.6.
- Probability of superiority: `PS(a, a′) = mean_{i,j}( [r_a,i < r_a′,j] + 0.5 [r_a,i = r_a′,j] )` (over all pairs of runs; ties have no tolerance; two values of +inf are tied).
- Mean probability of superiority: `J(p, a) = (1 / (|𝒜| − 1)) Σ_{a′ ≠ a} PS(a, a′)`. The value lies in [0, 1], and larger is better.

### 6.6 Fitting and Coefficient of Determination (R-R2-1)

- There is one predictor per algorithm, fitted on (features, J(·, a)) of the training instances. The same predictor is copied for the two feature types.
- Before the features are passed to the predictor, values that differ only by rounding are unified (D38).
  - Two values a and b are unified only if `math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)` (below, "close"). Closeness depends only on the two values and not on the other values of the column (such as outliers).
  - Fitting: in each column, the values of the training instances are visited in increasing order, and a value close to the last kept value is not kept. Then each value is replaced by the nearest kept value if the two are close. If all values of a column are close to the smallest value, the column becomes exactly constant.
  - Prediction: each value is replaced by the nearest kept value if the two are close.
  - NaN and ±inf stay as they are. `fit_predictor` and `r2_score` do the same, so the component functions give the same R² as `run_test`. `result.features` holds the values as the extractor returned them.
- The predicted values Ĵ are clipped to [0, 1].
- `R²(Q) = 1 − Σ (J − Ĵ)² / Σ (J − J̄)²`. The sums and the mean J̄ are taken over all pairs q ∈ Q and a ∈ 𝒜 arranged in one column (not the mean of R² per algorithm). The value equals `sklearn.metrics.r2_score(J.ravel(), clip(Ĵ).ravel())`.
- If all values of J are equal (the denominator is 0), `r2_score` returns `nan` and logs `R^2 is undefined because all performance values are equal; returned nan`. Equality is decided by comparing the values of J directly (not by relying on the rounding of the mean or of the denominator: the mean of equal values can differ from that value by rounding, in which case the denominator is not 0 but about 1e-33, and R² becomes a value such as −6×10³¹).
- The values of the table: `r2_test = R²(P_test)`, `r2_transformed_test = R²(P̃_test)`, `degradation = r2_test − r2_transformed_test`.

### 6.7 Monotone Transformation of Any Objective Function (R-AUG-1)

`rr.transform_objective(objective, lower_bounds, upper_bounds, *, seed, n_samples=10_000)` returns h(x) = g(objective(x)). g follows the formula of the paper and is built by the following steps. The key `"objective"` does not overlap with the keys of BBOB (which start with `"bbob"`).

1. `lower_bounds` and `upper_bounds` are converted to float64 arrays. Both are one-dimensional with at least one element, have the same shape, are finite, and satisfy lower < upper in every dimension. d is their length.
2. `rng = numpy.random.default_rng(derive_seed("fmax", seed, "objective"))` and `X = rng.uniform(lower_bounds, upper_bounds, size=(n_samples, d))`. `objective` is called on each row of X, one point at a time, and each value is converted with `float()`.
3. f̂_min and f̂_max are the smallest and the largest finite value (NaN and ±inf excluded).
4. ψ is built by steps 2 to 6 of 6.3, with the rng of step 1 replaced by `numpy.random.default_rng(derive_seed("transform", seed, "objective"))`. ψ depends only on `seed`; the same `seed` gives the same shape for every objective function.
5. g(y): `z = (y − f̂_min) / (f̂_max − f̂_min)`. ψ(z) if 0 < z < 1, and z otherwise (z ≤ 0 or z ≥ 1; the identity). NaN stays NaN, +inf stays +inf, and −inf stays −inf. Since ψ(0) = 0 and ψ(1) = 1, g is continuous at z = 0 and z = 1 and strictly increasing on the whole real line.
6. h(x): the value of `objective(x)` is converted to a float64 array, g is applied elementwise, and the result is returned as a `float` if it has zero dimensions and as an array otherwise. If `objective` takes (n, d) and returns (n,), h also takes (n, d). h is an instance of a private class and can be pickled if `objective` can.

The values are returned on the scale rescaled to [0, 1] by f̂_min and f̂_max (not on the original scale). Values better than the sample (near the optimum) fall on the side of the identity and are not stretched. The features are computed from uniform samples, so for augmentation, most values fall on the stretched side. Step 7 of 6.3 clips z < 0 to 0, but on BBOB f_min is the true minimum and z < 0 occurs only by rounding, so `transform` does not change (D43).

Errors (the three-line `RubberRulerError` of 5.6): `lower_bounds` and `upper_bounds` have different shapes, are not one-dimensional (including empty), cannot be converted to arrays of numbers, are not finite, or have a dimension with lower ≥ upper; `seed` is outside 0 to 2³²−1; `n_samples` is below 2; `objective` raised an exception on the sample (the original exception is chained); there are fewer than two distinct finite values.

Test values (the sphere function `numpy.sum(x**2, axis=-1)`, d = 2, bounds [−5, 5]², seed = 0, n_samples = 10,000; computed with NumPy 2.5 and SciPy 1.18; absolute tolerance 1e-9):

| Quantity | Value |
|---|---|
| f̂_min, f̂_max | `0.0026513298523528042`, `49.88435942364988` |
| h((0, 0)) (z < 0) | `-5.315234689572469e-05` |
| h((1, 2)) (0 < z < 1) | `0.13974027323189725` |
| h((5, 5)) (z > 1) | `1.0023182962406312` |

---

## 7. Quality Commitments

### 7.1 Reproducibility (R-REP-1)

- With the same arguments, `result.table` is the same for any value of `n_jobs` and any order of execution.
- Each worker limits BLAS and OpenMP to one thread (`threadpoolctl`).
- The randomness of the predictor is set by the user (for example, `random_state=0`). rubberruler does not touch it.
- Feature values that differ only by rounding are unified before fitting and prediction (6.6; D38), so that the table does not change when the rounding changes with the version of NumPy or with the CPU. [Verified] The table of the quickstart was identical to the last digit on Python 3.11 (NumPy 2.4), 3.12, and 3.14 (2026-10-01, on Linux on the development machine and on GitHub's CI; the same after the rule that decides closeness value by value).
- On another OS, the values of the table can differ in the third decimal place. The card was the same (in the evaluation of 2026-10-01, the table on macOS (Apple Silicon) differed from x86_64 by at most 1.7 × 10⁻³, because the sample for the features differs in the last digit).

### 7.2 Cache and Resumption (R-CACHE-1)

- Finished runs are appended in Parquet to `cache_dir/runs/` (other files directly in `cache_dir` are not read). Only the parent process writes.
- After an interruption, the same call resumes. When a function is modified, only the runs of that function are computed again. Functions are told apart by the text of their source, so a change of only a comment or of white space counts as a modification.
- An algorithm is identified by an identifier made from its key in the portfolio, its source, its default values, its closure values, and the global values it refers to (recursively for the user's functions, by version for libraries). The procedure is in `implementation.md` 3.5. A function that cannot be fully identified triggers the warning of 5.6.
- To delete the cache, the user deletes the folder (there is no function for it).
- Features are not cached (they are cheap compared with the runs).

### 7.3 Compatibility

- Python 3.11 or later, without an upper bound (D34). Dependencies and lower bounds: numpy 1.24, scipy 1.10, pandas 2.0, pyarrow 25.0, ioh 0.3.17, joblib 1.6, cloudpickle 3.0 (imported directly by the pre-check), threadpoolctl 3.7, rich 13.7 (the progress on stderr and the panel of the card), matplotlib 3.10 (imported only to write the figure of the answer; 3.4), and psutil 7.2 (not imported by rubberruler; joblib uses it to stop the workers on Ctrl-C; D39). A lower bound is the older version of the two tested environments (Python 3.12 of uv.lock and Python 3.11 of the pflacco environment), and there are no upper bounds. scikit-learn is not a dependency (the user brings it with the predictor). In the pflacco environment (NumPy 1.24), pyarrow is kept below 26: pyproject has no upper bound, and the pflacco job of CI and the install commands for pflacco in `examples/` state `"pyarrow<26"` (D44).
- The wheels of ioh 0.3.22 cover only CPython 3.9 to 3.12 on Linux x86_64, macOS, and Windows (checked on PyPI on 2026-09-29). On 3.13 and later and on Linux aarch64, pip builds ioh from the sdist, which needs a C++ compiler and CMake. CI runs on Python 3.12 and 3.14 with uv.lock (3.14 builds ioh from source) and on Python 3.11 in the pflacco environment.
- A change of a default, of the generation of the transformation, of the derivation of random numbers, or of the computation of J and R² (including the unification of values in 6.6) can change the answer. It is treated as a change of the major version (of the minor version while the version is 0.x), and its entry in CHANGELOG.md starts with **Can change the answer**. A change that alters the values of runs also raises the version of the cache format (`implementation.md` 3.5), so that old runs are not used. The Versioning section of the README states this policy and asks the user to keep the version (`importlib.metadata.version("rubberruler")`) with the results.

---

## 8. Structure of the README

The readers of the README know BBOB, CMA-ES, ELA, algorithm selection, and invariance to monotone transformations, as participants of GECCO do, but they do not know rubberruler. Reading from the top, a reader obtains the first card and can state its meaning without getting lost. How to connect each tool is in `examples/README.md` (D42).

### 8.1 Writing Rules

The README and `examples/README.md` follow these rules, also when they are revised.

1. The main path comes first: installing, the quickstart, and how to read the card. Notes for only some environments go to Installation, and details for each tool go to `examples/README.md`.
2. One idea per paragraph. The answer comes first.
3. The words of the reader (BBOB, ELA, CMA-ES, invariant) are used. The terms of rubberruler (feature type, transformed test instances, degradation, J) are explained once, where they first appear.
4. The same thing is called by the same word. "Rubber ruler" is used only as the metaphor for ψ (likewise, "run" is used only for one execution of an algorithm, and the whole call is written `run_test`).
5. The code in the README has only the comments that the reader needs. It has no comments for tools (`pyright: ignore`, notes on SciPy versions) (D41).
6. A number, such as a duration, is stated in one place.
7. The language is academic American English (Section 9).

Correspondence with Diátaxis: the tutorial is the Quickstart, the how-to guides are Usage (Inputs) and `examples/README.md`, the explanation is How it works, and the reference is Usage (Settings and API).

### 8.2 Sections of the README

The opening and the sections, in order, as of 0.1.0.

1. **Opening**: the title, the badges (PyPI, Python versions, CI, license), one paragraph on what rubberruler does (given an algorithm portfolio, a feature extractor, and a performance predictor, it tells whether to compute the features value-based or rank-based), one paragraph on why (comparison-based algorithms such as CMA-ES and DE are invariant to strictly increasing transformations of the objective, but features computed from the objective values are not, so a predictor can fail although the best algorithm stays the same; rubberruler transforms BBOB test instances with random strictly increasing functions, the rubber rulers, and keeps the feature type whose predictions still hold), and the flow figure (`assets/flow-light.png` and `assets/flow-dark.png`; `implementation.md` 3.10).
2. **Installation**: `pip install rubberruler` and Python 3.11 or later. ioh decides how the installation goes: wheels for 3.11 and 3.12 on Linux (x86_64), macOS, and Windows; on 3.13 and later, pip builds ioh from source, which needs a C++ compiler and CMake and takes about five minutes (without a compiler, install Python 3.12 alongside; on Windows, `py -3.12`); Linux aarch64 (including Docker on Apple Silicon Macs) builds from source on every version, so a compiler comes first; Windows needs the Visual C++ Redistributable (X64), and without it, `import rubberruler` stops and says so (5.6). A link to Examples for the other packages.
3. **Quickstart** (tutorial): `pip install rubberruler scikit-learn`. A link to `examples/quickstart.py` and its code (a test checks that the code is the same as in the README). Numbered comments separate the three components (1. the portfolio; 2. the extractor, with the statement that y is rescaled or ranked; 3. the predictor); `portfolio` and `predictor` are variables, and the three are passed to `run_test` as positional arguments. The comment on the small settings (`dimension=2`, `budget_per_dimension=100`, `n_runs=5`) states the time of the default settings (about an hour on a 16-core PC). The time of the run (about half a minute on 24 CPUs and about five minutes on 2) and that the progress is shown on stderr. The card (3.2; run with the small settings, and a test checks that it is the same to the character). How to read the card (the first line is the answer; the second line is the evidence, where the transformed test instances are explained; the table shows R² for each feature type, where the feature type and the degradation, the difference of the values before rounding, are explained; the last lines give the numbers of algorithms and instances, and the settings). One line on `rich.print(result)` (3.2). The figure of the answer (3.4; `assets/rubber_ruler_test.png`).
4. **Usage**:
   - **Inputs** (how-to): the table of the three types of Chapter 4, and one item each for algorithms, extractors, and predictors (the two forms of `problem(x)` and its attributes; when the budget is spent, the next call ends the run with an exception that `except Exception` does not catch; algorithms written for IOHexperimenter work without changes; X in the search range, y in [0, 1], the same keys, NaN for values that cannot be computed; scikit-learn regressors and `Pipeline`, any object with `fit` and `predict`, copies are fitted (X holds the features of the training instances, y the performance of the algorithm); the user sets the predictor's seed).
   - **Settings** (reference): the table of the arguments of 5.2 (argument, default, meaning). The accepted values are not in the table but in the docstring of `run_test` (`help(rr.run_test)`).
   - **API** (reference): the table of the 10 names of 5.1, and a link to the example in `examples/README.md` that scores another predictor on the same data.
   - **Notes**: pre-checks (5.5: short runs with a small budget, the predictor on dummy data, the extractor on each function, the 60-second limit, the three-line error); caching and resuming (7.2); memory and `n_jobs`; progress (rich in a terminal or a notebook, plain lines at most every 30 seconds in a file, `RUBBERRULER_NO_PROGRESS=1`); Docker on Windows and macOS (keep `cache_dir` inside the container or in a named volume); augmentation (`rr.transform_objective`, with a link to `examples/README.md`); reproducibility (7.1).
5. **How it works** (explanation): the four stages of the progress, with the numbers of the default settings: preparing instances (training 11 to 110, test 1 to 10, the transformation g and ψ; above the estimated maximum, g only rescales, so g is strictly increasing everywhere), measuring performance (31 runs, budget 1000·d, J; a tie counts as half), computing features (50·d uniform random points, the value and rank preprocessing), and fitting predictors and scoring (one predictor per algorithm; R² over all pairs of instance and algorithm). The decision rule (3.3), its reason (J of comparison-based algorithms and rank-based features do not change, value-based features do), why the degradation does not decide, and the two cases of a weak answer (3.2). The figure of ψ (25 curves of ψ from 6.3, `assets/transformations-light.png` and `assets/transformations-dark.png`; the figure of the paper is not used; D24).
6. **Examples**: a link to `examples/` and `examples/README.md`, and the table of tools (ModularCMAES, ModDE, pymoo, pycma and Nevergrad, pflacco, DeepELA, PyTorch), each row linking to its section in `examples/README.md`.
7. **Versioning**: the policy of 7.3 (a change that can change the answer comes with a new major version and is marked in CHANGELOG.md) and keeping the version with the results.
8. **Citation**: a link to `CITATION.cff` ("Cite this repository" on GitHub). The paper is added once it is published (`implementation.md` 6.3).
9. **Development**: clone, `uv sync`, `uv run pytest`; `uv run pytest -m "not slow"` skips the slow tests; the other commands of CI; running the quickstart with `uv run python examples/quickstart.py`.
10. **License**: MIT.

Items removed from the README are written where users read them: the example of the progress on stderr is in the docstring of `examples/quickstart.py`; the line with the algorithms to compute again below the plan line, how fast Ctrl-C stops and that no workers remain, and two calls that use the same folder at once are in the docstring of `run_test`; the exception at the end of the budget and points outside the range are in the docstring of `Problem` and in Algorithms of `examples/README.md`; the unification of rounding errors and the warning on `random_state=None` are in Predictors of `examples/README.md`.

### 8.3 Structure of examples/README.md

The page starts with one sentence, the table of example scripts (script, what it shows, the install command), and the table of tools of the README (with links to the sections of this file). Each section has, in this order, how to install the tool (with the Python versions checked), the code, and one link to a runnable example. The code is the same as in the example file (checked by a test; `implementation.md` Chapter 4). The table of the README points to the anchors that GitHub makes from the headings, so a changed heading also changes the links in the README (checked by a test).

1. **Algorithms**: a call after the budget is spent ends the run with an exception that `except Exception` does not catch, and the run keeps the smallest value it saw. Points outside the search range can be evaluated, and they count toward the budget. SciPy's algorithms are `de` and `nelder_mead` of the quickstart.
   - ModularCMAES (`#modularcmaes`): `pip install modcma==1.2.0` (3.11 and 3.12). The call of 4.1 (seeds below 2³¹). A link to `examples/modular_ela.py`.
   - ModDE (`#modde`): `pip install modde==0.0.4` (3.11 and 3.12). Bounds of shape (d, 1), and ModDE reads `problem.state.optimum_found`. A link to `examples/modular_ela.py`.
   - pymoo (`#pymoo`): `pip install pymoo==0.6.2` (3.11 and 3.12, also next to pflacco). `PymooView` and `pymoo_algorithm`. Options can be passed with a lambda. A link to `examples/pymoo_portfolio.py`.
   - pycma and Nevergrad (`#pycma-and-nevergrad`): `pip install cma nevergrad` (the versions checked and Python 3.12). `run_cma_es` and `run_nevergrad_cma`. Nevergrad's `CMA` can emit pycma's `InjectionWarning`, which also appears without rubberruler, and the run ends normally. They are added to the portfolio of the quickstart.
   - IOHexperimenter and functions of another shape: the names of IOHexperimenter (4.1) and the one-line lambda.
2. **Feature extractors**:
   - pflacco (`#pflacco`): `pip install pflacco==1.2.2 "pyarrow<26"`, on Python 3.11 only. One sentence on why pyarrow stays below 26 (D44). The reason for Python 3.11 (pinned NumPy 1.24 and SciPy 1.10, and the errors on 3.12) and the reason to name the version. `ela_features` and why `costs_` is left out. Infinity becomes NaN, so a Pipeline that fills NaN is used. Why d = 2 is not used (the quickstart uses d = 2, so one paragraph before the estimate of the time). The estimate of the time (about 0.5 seconds per instance at d = 10; with the default settings, close to an hour for the features alone) and small settings to try first (d = 10 kept, training 11 to 15, test 1 to 3, `budget_per_dimension=100`, `n_runs=5`; about 3.5 minutes on 16 cores). A link to `examples/modular_ela.py` and one line that installs pflacco, ModularCMAES, and ModDE.
   - DeepELA (`#deepela`): one way to install it (fetch a pinned commit, `pip install torch lightning einops pyyaml`, and put it on PYTHONPATH). The versions checked (torch 2.14.0, lightning 2.6.6, Python 3.11 and 3.12). One item each on pinning the commit, not using install.sh, and the CPU build of PyTorch for CPU-only use. `deep_ela_features`. Why `torch.manual_seed` is set, why d ≤ 11, and the relation to DeepELA's "no preprocessing". A link to `examples/deepela.py`, and the time of the features with the default settings when `device="cuda"` is passed on an NVIDIA GPU (measured on an RTX 4070 Ti) and the difference from the CPU (up to about 1 × 10⁻⁵). The default of the example stays the CPU.
   - Extractors of another shape: a function that takes `(y, X)` and returns a pandas Series, connected by a one-line lambda (4.2).
3. **Predictors**: values of features that differ only by rounding are unified (6.6; D38), and the user sets the predictor's seed, with a warning if `random_state` is `None` (5.6).
   - A Pipeline that fills NaN: `make_pipeline(SimpleImputer(), RandomForestRegressor(random_state=0))`. A link to `examples/modular_ela.py`.
   - PyTorch (`#pytorch`): `pip install torch` (the version checked and Python 3.12). A small network with `fit` and `predict`.
4. **Another predictor on the same data** (`#another-predictor-on-the-same-data`): an example that measures R² of another predictor on the same performance data with `rr.fit_predictor` and `rr.r2_score`.
5. **Augmenting training data** (`#augmenting-training-data`): an example that applies 10 transformations with different `seed` values to Rastrigin written with NumPy, through `rr.transform_objective`, and compares the value-based and rank-based features of the original and of the copies (the same code as `examples/augmentation.py`, and its output). One sentence states that values better than the sample (near the optimum) are not stretched.

---

## 9. Language

Everything in the repository (documents, comments, docstrings, commit messages, the CHANGELOG) is written in academic American English, in the style of the paper.

- No contractions (do not, it is) and no colloquial expressions.
- American spelling (optimization, normalize, behavior), as GECCO (ACM) uses it.
- Only the English terms of the table in Chapter 2 are used. The same thing is not called by different words.
- `spec.md`, `decisions.md`, and `implementation.md` are written impersonally. The README, `examples/README.md`, and `ux-design.md` explain things to users and may address the reader as "you".
- Mathematical symbols (ψ, ≤, R², and so on) may appear as they are in Markdown documents. The text that rubberruler writes stays ASCII (D35).
