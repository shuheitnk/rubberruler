import importlib.resources
import inspect
import pkgutil

import rubberruler

PUBLIC_NAMES = {
    "run_test",
    "bbob_instances",
    "ma_bbob_instances",
    "transform",
    "transform_objective",
    "measure_performance",
    "compute_features",
    "fit_predictor",
    "r2_score",
    "Problem",
    "RubberRulerError",
}


def test_all_is_exactly_the_eleven_public_names() -> None:
    """R-API-1: rubberruler.__all__ is exactly these 11 names."""
    assert sorted(rubberruler.__all__) == sorted(PUBLIC_NAMES)
    for name in rubberruler.__all__:
        assert hasattr(rubberruler, name)


def test_modules_are_private_and_names_are_not_collected_by_pytest() -> None:
    """Every module starts with _, and no public name starts with test."""
    modules = [module.name for module in pkgutil.iter_modules(rubberruler.__path__)]
    assert modules
    assert all(name.startswith("_") for name in modules)
    assert not any(name.lower().startswith("test") for name in rubberruler.__all__)


def test_py_typed_is_shipped() -> None:
    """The package ships py.typed, so type checkers read its annotations."""
    assert importlib.resources.files("rubberruler").joinpath("py.typed").is_file()


def test_help_explains_every_public_name() -> None:
    """help() has a docstring for every public name, and run_test says what it
    returns."""
    for name in rubberruler.__all__:
        assert inspect.getdoc(getattr(rubberruler, name)), name
    assert "Returns:" in (inspect.getdoc(rubberruler.run_test) or "")
