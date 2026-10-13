# CLAUDE.md

rubberruler is a Python library that decides on BBOB, for the user's algorithm portfolio, feature extractor, and performance predictor, whether the features should be computed value-based or rank-based. The user only matches three input and output types and receives R² after strictly increasing transformations, together with the answer.

## Read First

1. `docs/spec.md`: what is built. The single source of truth for the specification.
2. `docs/implementation.md`: the order of construction (M0 to M9, completed for 0.1.0), verified facts and pitfalls, tests, and tools.
3. `docs/decisions.md`: why. The decisions there are not reconsidered.
4. `docs/ux-design.md`: the grounds of the rules on the user experience (human factors and software engineering). Read it only when needed. If it disagrees with spec.md, spec.md is correct.

## Rules

- **Build nothing that is not in the specification.** The public names are exactly the 10 names of spec 5.1. If an argument, attribute, convenience function, or option seems worth adding, do not add it; ask Shuhei. In particular, add nothing from "Out of Scope" in spec 1.3.
- **Each input has one type.** As in the table of spec Chapter 4. No branches that accept other forms.
- **The vocabulary is the table of spec Chapter 2.** Use only its English terms, and do not call the same thing by different words. No abbreviations (`dimension`, not `dim`). No public function that starts with `test` and no public class that starts with `Test`.
- **Write everything in the repository in academic English** (documents, comments, docstrings, commit messages, and the CHANGELOG), in the style of the paper: no contractions or colloquial expressions, and American spelling (spec Chapter 9).
- **One error type, `RubberRulerError`, with three lines: what / Why / Fix.** spec 5.6.
- **Write the random numbers and the transformation from scratch, exactly as in spec Chapter 6.** Do not consult the experimental code of the paper (D24). Check `derive_seed`, ψ, and f̂_max against the test values of spec 6.1 and 6.3.
- **A change that can change the answer (a default, the transformation, the derivation of random numbers, the computation of J and R²) is a change of the specification.** Revise `docs/spec.md` first and confirm with Shuhei.
- The license is MIT, and the copyright holder is The University of Fukuchiyama (D23).

## Way of Working

- Each change passes the checks below, as CI runs them, before it is committed.
- The docstring of a new test names the requirement ID of the specification that it checks (for example, `R-TRF-1`). The tests of 0.1.0 do not carry IDs.
- When in doubt, remove rather than add.

## Commands

The same commands as the job `check` of CI (there is no justfile):

```
uv sync                                   # install the dependencies
uv run ruff format --check                # formatting (uv run ruff format to apply)
uv run ruff check                         # lint
uv run --python 3.11 --isolated pyright   # types for Python 3.11
uv run pyright --pythonversion 3.12       # types for Python 3.12
uv run pytest                             # all tests (-m "not slow" skips the slow ones)
uv run python examples/quickstart.py      # the quickstart with small settings
```

Python 3.11 or later (3.13 and later have no wheels of ioh, so pip builds it from source; D34; 3.11 serves users of pflacco; D27). Development and uv.lock use 3.12.
