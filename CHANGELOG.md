# Changelog

## 0.1.0 (unreleased; to be published with the paper)

The first release.

- `rr.run_test` answers whether the features should be computed value-based or rank-based on BBOB, and its component functions repeat its stages.
- `rr.transform_objective(objective, lower_bounds, upper_bounds, seed=...)` applies a random rubber ruler to any objective function, to augment training data. It does not change `run_test` or its answer.
- The card uses the terms in full: "strictly increasing transformations of the objective" (the term of the paper), "R^2 on the transformed test instances", a header of two lines over "test instances" and "transformed test instances", and "2400 training instances, 240 test instances, 240 transformed test instances". Lines that would exceed 80 characters are broken, and long warnings wrap. The figure of the answer names its blocks "test instances" and "transformed test instances".
- Messages say "received" (for example, "transform received str."), and the warning on failed runs ends with "(+inf if there was none; see the log)".
