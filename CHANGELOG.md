# Changelog

## Unreleased

### Added

- `rr.transform_objective(objective, lower_bounds, upper_bounds, seed=...)` applies a random rubber ruler to any objective function, to augment training data. It does not change `run_test` or its answer.

### Changed

- The card says "strictly increasing transformations of the objective" instead of "monotone transformations of the objective", the term of the paper. The answer and the numbers do not change.

## 0.1.0

The first release, published with the paper.
