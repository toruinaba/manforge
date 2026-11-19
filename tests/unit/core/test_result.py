"""Unit tests for ReturnMappingResult / StressUpdateResult dataclasses.

Absorbed from tests/unit/test_return_mapping_result.py.
"""

import autograd.numpy as anp
import numpy as np
import pytest

from manforge.core import ReturnMappingResult, StressUpdateResult
from manforge.simulation import PythonIntegrator, PythonAnalyticalIntegrator


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
    C = model.elastic_stiffness(model.initial_state())
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
    from manforge.simulation import PythonNumericalIntegrator
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
    from manforge.simulation import PythonNumericalIntegrator

    integ = PythonNumericalIntegrator(model)
    deps = anp.array([3e-3, 0, 0, 0, 0, 0])
    result = integ.stress_update(deps, anp.zeros(6), initial_state)
    direct = integ._consistent_tangent(result.return_mapping, result.increment)
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


def test_result_carries_its_increment(model, initial_state):
    """state_n, Δε and the trial state travel with the result."""
    integrator = PythonIntegrator(model)
    stress_n = anp.array([50.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for deps in (anp.array([1e-5, 0, 0, 0, 0, 0]), anp.array([3e-3, 0, 0, 0, 0, 0])):
        r = integrator.stress_update(deps, stress_n, initial_state)
        inc = r.increment
        np.testing.assert_array_equal(np.asarray(inc.strain_inc), np.asarray(deps))
        np.testing.assert_array_equal(np.asarray(inc.state_n["stress"]), np.asarray(stress_n))
        np.testing.assert_array_equal(np.asarray(inc.state_n["ep"]), np.asarray(initial_state["ep"]))
        assert r.stress_trial is inc.trial["stress"]
        expected_trial = stress_n + model.elastic_stiffness(initial_state) @ deps
        np.testing.assert_allclose(np.asarray(inc.trial["stress"]), np.asarray(expected_trial))


def test_elastic_state_is_a_fresh_dict(model, initial_state):
    """Mutating a returned state must not reach the result or the increment."""
    r = PythonIntegrator(model).stress_update(
        anp.array([1e-5, 0, 0, 0, 0, 0]), anp.zeros(6), initial_state)
    state = r.state
    state["ep"] = anp.array(123.0)
    assert float(r.state["ep"]) == 0.0
    assert float(r.increment.trial["ep"]) == 0.0


# ---------------------------------------------------------------------------
# Every field of a return-mapping result is stated
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("left_out", ["n_iterations", "residual_history", "converged"])
def test_return_mapping_result_has_no_defaults(left_out):
    """A hand-written NR that ran out of iterations must say so.

    With ``converged=True`` as the default, leaving the flag out reported
    success; with ``residual_history=[]`` the solution check had no solver
    tolerance to allow for.
    """
    fields = dict(
        stress=np.zeros(6), state={"stress": np.zeros(6)}, dlambda=np.array(0.0),
        n_iterations=1, residual_history=[1.0, 0.0], converged=True,
    )
    ReturnMappingResult(**fields)
    del fields[left_out]
    with pytest.raises(TypeError, match=left_out):
        ReturnMappingResult(**fields)
