# Changelog

## 0.1.0 (2026-11-13)

- `rr.run_test(portfolio, feature_extractor, predictor)` answers whether to compute features value-based or rank-based (from the objective values or from their ranks, both min-max normalized), prints the card, and saves the figure.
- Problem suites: BBOB, MA-BBOB (`suite="ma-bbob"`), and your own objective functions (`rr.objective_instances`).
- `performance=` and `features=` reuse the tables that you already have.
- `rr.transform_objective` applies a random rubber ruler, the strictly increasing transformation of the paper, to any objective function.
- Each stage of the test is also available as a function.
- Runs and features use every CPU (`n_jobs`), and finished runs are cached.
- Python 3.11 or later.
