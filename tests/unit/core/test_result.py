"""Unit tests for ReturnMappingResult / StressUpdateResult dataclasses.

Absorbed from tests/unit/test_return_mapping_result.py.
"""

import autograd.numpy as anp
import numpy as np
import pytest

from manforge.core.result import ReturnMappingResult, StressUpdateResult
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
    np.testing.assert_allclose(np.array(result.stress_trial), np.array(result.stress), rtol=1e-12)
    assert result.ddsdde.shape == (6, 6)
    assert result.stress.shape == (6,)


def test_plastic_step_fields(model, initial_state):
    deps = anp.array([3e-3, 0, 0, 0, 0, 0])
    result = PythonIntegrator(model).stress_update(deps, anp.zeros(6), initial_state)
    assert result.is_plastic is True
    assert float(result.dlambda) > 0.0
    assert result.stress.shape == (6,)
    assert result.ddsdde.shape == (6, 6)
    assert result.stress_trial.shape == (6,)
    f = model.yield_function(result.state)
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


def test_elastic_step_state_stress_advances(model, initial_state):
    deps = anp.array([1e-4, 0, 0, 0, 0, 0])
    result = PythonIntegrator(model).stress_update(deps, anp.zeros(6), initial_state)
    assert result.is_plastic is False
    np.testing.assert_allclose(
        np.array(result.state["stress"]), np.array(result.stress), rtol=1e-14
    )


def test_plastic_tangent_is_lazy_and_cached(model, initial_state, monkeypatch):
    """The autodiff tangent must not be computed until ddsdde is read.

    Strain-controlled stepping never consumes the tangent, and its Jacobian
    costs as much as several NR iterations, so stress_update defers it.
    """
    from manforge.simulation.integrator import PythonNumericalIntegrator
    from manforge.simulation.integrator.base import _PythonIntegratorBase

    calls = {"n": 0}
    orig = _PythonIntegratorBase._consistent_tangent

    def counting(self, *args, **kwargs):
        calls["n"] += 1
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(_PythonIntegratorBase, "_consistent_tangent", counting)

    deps = anp.array([3e-3, 0, 0, 0, 0, 0])
    result = PythonNumericalIntegrator(model).stress_update(deps, anp.zeros(6), initial_state)
    assert result.is_plastic is True
    assert calls["n"] == 0

    first = result.ddsdde
    assert calls["n"] == 1
    assert first.shape == (6, 6)
    assert result.ddsdde is first  # cached, not recomputed
    assert calls["n"] == 1


def test_lazy_tangent_matches_eager_value(model, initial_state):
    """Deferral must not change the tangent — compare against a direct call."""
    from manforge.simulation.integrator import PythonNumericalIntegrator

    integ = PythonNumericalIntegrator(model)
    deps = anp.array([3e-3, 0, 0, 0, 0, 0])
    result = integ.stress_update(deps, anp.zeros(6), initial_state)
    direct = integ._consistent_tangent(
        result.return_mapping, anp.zeros(6), initial_state, deps, result.stress_trial
    )
    np.testing.assert_allclose(np.asarray(result.ddsdde), np.asarray(direct), rtol=1e-12)


def test_elastic_step_state_stress_matches_next_step_input(model, initial_state):
    """state["stress"] must equal the stress_n the driver carries to step n+1."""
    integrator = PythonIntegrator(model)
    deps = anp.array([1e-4, 0, 0, 0, 0, 0])
    stress_n = anp.zeros(6)
    state_n = initial_state
    for _ in range(3):
        result = integrator.stress_update(deps, stress_n, state_n)
        assert result.is_plastic is False
        np.testing.assert_allclose(
            np.array(result.state["stress"]), np.array(result.stress), rtol=1e-14
        )
        stress_n = result.stress
        state_n = result.state
