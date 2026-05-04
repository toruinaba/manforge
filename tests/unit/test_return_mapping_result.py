"""Tests for ReturnMappingResult dataclass fields."""

import autograd.numpy as anp
import numpy as np
import pytest

from manforge.core.stress_update import ReturnMappingResult, StressUpdateResult
from manforge.simulation.integrator import PythonIntegrator, PythonAnalyticalIntegrator


def test_result_is_dataclass(model, initial_state):
    deps = anp.zeros(6)
    result = PythonIntegrator(model).stress_update(deps, anp.zeros(6), initial_state)
    assert isinstance(result, StressUpdateResult)


def test_elastic_step_fields(model, initial_state):
    deps = anp.array([1e-4, 0, 0, 0, 0, 0])
    result = PythonIntegrator(model).stress_update(deps, anp.zeros(6), initial_state)

    assert result.is_plastic is False
    assert float(result.dlambda) == pytest.approx(0.0)
    np.testing.assert_allclose(
        np.array(result.stress_trial), np.array(result.stress), rtol=1e-12
    )
    assert result.ddsdde.shape == (6, 6)
    assert result.stress.shape == (6,)


def test_plastic_step_fields(model, initial_state):
    # Large uniaxial strain — clearly plastic
    deps = anp.array([3e-3, 0, 0, 0, 0, 0])
    result = PythonIntegrator(model).stress_update(deps, anp.zeros(6), initial_state)

    assert result.is_plastic is True
    assert float(result.dlambda) > 0.0
    assert result.stress.shape == (6,)
    assert result.ddsdde.shape == (6, 6)
    assert result.stress_trial.shape == (6,)

    # Yield function should be ~0 after plastic step
    f = model.yield_function(result.stress, result.state)
    assert abs(float(f)) < 1e-8


def test_stress_trial_is_elastic_prediction(model, initial_state):
    C = model.elastic_stiffness()
    deps = anp.array([3e-3, 0, 0, 0, 0, 0])
    result = PythonIntegrator(model).stress_update(deps, anp.zeros(6), initial_state)

    expected_trial = anp.zeros(6) + C @ deps
    np.testing.assert_allclose(
        np.array(result.stress_trial), np.array(expected_trial), rtol=1e-12
    )


def test_analytical_method_sets_is_plastic(model, initial_state):
    deps = anp.array([3e-3, 0, 0, 0, 0, 0])
    result = PythonAnalyticalIntegrator(model).stress_update(deps, anp.zeros(6), initial_state)
    assert result.is_plastic is True
    assert float(result.dlambda) > 0.0


def test_state_updated_in_plastic_step(model, initial_state):
    deps = anp.array([3e-3, 0, 0, 0, 0, 0])
    result = PythonIntegrator(model).stress_update(deps, anp.zeros(6), initial_state)
    assert float(result.state["ep"]) > 0.0


def test_state_unchanged_in_elastic_step(model, initial_state):
    deps = anp.array([1e-4, 0, 0, 0, 0, 0])
    result = PythonIntegrator(model).stress_update(deps, anp.zeros(6), initial_state)
    assert float(result.state["ep"]) == pytest.approx(0.0)
