# Changelog

## 0.2.0 (unreleased)

The answer for the same inputs does not change.

### Added

- MA-BBOB: `rr.run_test(..., suite="ma-bbob")` and `rr.ma_bbob_instances(...)`. Each instance number gives one MA-BBOB instance; the default instance numbers are 1 to 2400 for training and 2401 to 2640 for test. BBOB stays as it was: the same random numbers, instance ids, cached runs, and card.

### Changed

- In `run_test`, the default of `train_instance_numbers` and `test_instance_numbers` is `None`, which takes the default instance numbers of the suite (on BBOB, 11 to 110 and 1 to 10, as before). `result.config` also holds `suite`.
- `ProblemInstance` has the attribute `suite`, and its `function_id` is `None` on MA-BBOB.
- Instance numbers above 2**31 - 1 stop with a `RubberRulerError` in the check of the settings, on both suites. Before, ioh raised a TypeError.

## 0.1.0 (unreleased; to be published with the paper)

The first release.

- `rr.run_test` answers whether the features should be computed value-based or rank-based on BBOB, and its component functions repeat its stages.
- `rr.transform_objective(objective, lower_bounds, upper_bounds, seed=...)` applies a random rubber ruler to any objective function, to augment training data. It does not change `run_test` or its answer.
- The card uses the terms in full: "strictly increasing transformations of the objective" (the term of the paper), "R^2 on the transformed test instances", a header of two lines over "test instances" and "transformed test instances", and "2400 training instances, 240 test instances, 240 transformed test instances". Lines that would exceed 80 characters are broken, and long warnings wrap. The figure of the answer names its blocks "test instances" and "transformed test instances".
- Messages say "received" (for example, "transform received str."), and the warning on failed runs ends with "(+inf if there was none; see the log)".
