"""Unit tests for fitting/objective.py — objective assembly and failure handling."""

import importlib
import math

import numpy as np
import pytest

from manforge.fitting import fit_params
from manforge.fitting.objective import FailureLog, build_objective
from manforge.fitting.space import ParameterSpace
from manforge.fitting.target import FitTarget, Observation
from manforge.simulation.driver import StrainDriver
from manforge.simulation.integrator import PythonIntegrator
from manforge.simulation.types import FieldHistory, FieldType


@pytest.fixture
def strain_load():
    return FieldHistory(FieldType.STRAIN, "Strain", np.linspace(0.0, 5e-3, 10))


@pytest.fixture
def driver_factory():
    return lambda integrator: StrainDriver(integrator)


@pytest.fixture
def space():
    return ParameterSpace(
        free={"sigma_y0": (250.0, (100.0, 400.0))},
        fixed={"E": 210000.0, "nu": 0.3, "H": 1000.0},
    )


@pytest.fixture
def target(model, strain_load):
    """A target whose values come from the model itself, so the optimum is exact."""
    stress = StrainDriver(PythonIntegrator(model)).run(strain_load).stress[:, 0]
    return FitTarget(
        load=strain_load,
        observations=[Observation(values=stress, component=0)],
        name="self",
    )


def test_objective_importable():
    mod = importlib.import_module("manforge.fitting.objective")
    assert mod is not None


def test_objective_is_zero_at_true_params(model, driver_factory, target, space):
    obj = build_objective(model, driver_factory, [target], space)
    assert obj(space.x0) == pytest.approx(0.0)


def test_objective_sums_over_targets(model, driver_factory, target, space):
    """Duplicating a target scales the objective, confirming targets combine."""
    one = build_objective(model, driver_factory, [target], space)
    three = build_objective(model, driver_factory, [target] * 3, space)
    x = [300.0]  # away from the optimum so the value is nonzero
    assert three(x) == pytest.approx(3.0 * one(x))


# ---------------------------------------------------------------------------
# Failure handling — an unsolvable parameter value must not kill the fit
# ---------------------------------------------------------------------------

class _ExplodingDriver:
    """Driver that fails the way an unconverged return mapping does."""

    def __init__(self, integrator):
        self.integrator = integrator

    def run(self, *args, **kwargs):
        raise RuntimeError("NR did not converge at step 3")


def test_failed_evaluation_returns_inf(model, target, space):
    obj = build_objective(model, _ExplodingDriver, [target], space)
    assert obj(space.x0) == math.inf


def test_failed_evaluation_is_recorded(model, target, space):
    failures = FailureLog()
    obj = build_objective(model, _ExplodingDriver, [target], space, failures)
    obj(space.x0)
    obj([300.0])

    assert failures.n_failed == 2
    assert "NR did not converge" in failures.first_message
    # The first failure's params are kept, not the latest, so the record points
    # at the earliest reproducible case.
    assert failures.first_params["sigma_y0"] == pytest.approx(250.0)


def test_failure_log_is_optional(model, target, space):
    """Penalisation must not depend on a log being supplied."""
    obj = build_objective(model, _ExplodingDriver, [target], space, None)
    assert obj(space.x0) == math.inf


def test_non_finite_objective_is_penalised(model, driver_factory, target, space):
    """A model that reaches nan without raising is still not an optimum."""
    nan_target = FitTarget(
        load=target.load,
        observations=[Observation(values=np.full(10, np.nan), component=0)],
        name="nan",
    )
    failures = FailureLog()
    obj = build_objective(model, driver_factory, [nan_target], space, failures)

    assert obj(space.x0) == math.inf
    assert failures.n_failed == 1
    assert "nan" in failures.first_message


def test_setup_errors_still_propagate(model, driver_factory, target, space):
    """A shape mismatch is a broken setup, not an unsolvable parameter value."""
    bad = FitTarget(
        load=target.load,
        observations=[Observation(values=np.zeros(10), component=None)],
        name="bad-shape",
    )
    obj = build_objective(model, driver_factory, [bad], space)
    with pytest.raises(ValueError, match="set component= to select"):
        obj(space.x0)


def test_fit_survives_a_failing_region(model, target):
    """The optimiser may probe unsolvable params and must still return a fit."""
    # Start away from the optimum (250) and make the unsolvable band sit beyond
    # it, so the optimiser walks into the band while closing in.
    space = ParameterSpace(
        free={"sigma_y0": (180.0, (100.0, 400.0))},
        fixed={"E": 210000.0, "nu": 0.3, "H": 1000.0},
    )

    def flaky_factory(integrator):
        # Fail on a parameter band, the way a model with no real root there
        # would -- not on a call count, which would also catch the x0 probe and
        # turn this into an unsolvable-start case instead.
        if integrator._model.sigma_y0 > 249.0:
            return _ExplodingDriver(integrator)
        return StrainDriver(integrator)

    result = fit_params(model, flaky_factory, target, space, max_iter=5)

    assert result.n_failed > 0
    assert math.isfinite(result.residual)
    assert "NR did not converge" in result.failure_message
    assert result.failure_params


def test_fit_reports_zero_failures_on_a_clean_run(
    model, driver_factory, target, space
):
    result = fit_params(model, driver_factory, target, space, max_iter=3)
    assert result.n_failed == 0
    assert result.failure_message == ""
    assert result.failure_params == {}


def test_unsolvable_initial_point_is_diagnosed_as_setup(model, target, space):
    """x0 failing is a setup problem, not a bad search region.

    The two cases need different fixes — initial values vs bounds — so the
    message must not blame the search when the fit never got to start.
    """
    with pytest.raises(RuntimeError, match="fitting cannot start") as exc:
        fit_params(model, _ExplodingDriver, target, space, max_iter=3)

    assert "NR did not converge" in str(exc.value)
    assert "before any search happens" in str(exc.value)


def test_initial_point_failure_mentions_decimation(model, target, space):
    """Decimation is the usual cause in practice, so name it explicitly."""
    with pytest.raises(RuntimeError, match="decimated"):
        fit_params(model, _ExplodingDriver, target, space, max_iter=3)


def test_out_of_bounds_initial_value_is_reported(model, target):
    """scipy clips these silently, so a failure must call them out."""
    space = ParameterSpace(
        free={"sigma_y0": (900.0, (100.0, 400.0))},  # init above its own bound
        fixed={"E": 210000.0, "nu": 0.3, "H": 1000.0},
    )
    with pytest.raises(RuntimeError, match="outside their own bounds") as exc:
        fit_params(model, _ExplodingDriver, target, space, max_iter=3)

    assert "sigma_y0=900 outside (100, 400)" in str(exc.value)


def test_in_bounds_initials_are_not_reported(model, target, space):
    """No spurious bounds warning when every initial value is legitimate."""
    with pytest.raises(RuntimeError) as exc:
        fit_params(model, _ExplodingDriver, target, space, max_iter=3)

    assert "outside their own bounds" not in str(exc.value)


def test_search_leaving_the_solvable_region_is_distinguished(model, target):
    """x0 solvable but everything after it failing points at the bounds."""
    space = ParameterSpace(
        free={"sigma_y0": (250.0, (100.0, 400.0))},
        fixed={"E": 210000.0, "nu": 0.3, "H": 1000.0},
    )
    calls = {"n": 0}

    def factory(integrator):
        # Let only the very first evaluation (the x0 probe) succeed.
        calls["n"] += 1
        return StrainDriver(integrator) if calls["n"] <= 1 else _ExplodingDriver(integrator)

    with pytest.raises(RuntimeError, match="no point the optimiser tried was") as exc:
        fit_params(model, factory, target, space, max_iter=3)

    assert "tightening the bounds" in str(exc.value)
    assert "fitting cannot start" not in str(exc.value)
