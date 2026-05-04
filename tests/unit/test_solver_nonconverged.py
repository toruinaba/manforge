"""P1: raise_on_nonconverged=False の挙動を検証。"""
import autograd.numpy as anp
import pytest

import manforge  # noqa: F401
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.simulation.integrator import PythonIntegrator, PythonNumericalIntegrator


@pytest.fixture
def j2():
    return J2Isotropic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)


def _build_plastic_trial(j2):
    """Return (stress_trial, state_n) clearly in the plastic regime."""
    state_n = j2.initial_state()
    stress_trial = anp.array([500.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # f_trial > 0
    return stress_trial, state_n


class TestRaiseOnNonconverged:
    def test_default_raises(self, j2):
        st, s_n = _build_plastic_trial(j2)
        with pytest.raises(RuntimeError, match="did not converge"):
            PythonNumericalIntegrator(j2, max_iter=0).return_mapping(st, s_n)

    def test_false_returns_nonconverged_result(self, j2):
        st, s_n = _build_plastic_trial(j2)
        result = PythonNumericalIntegrator(
            j2, max_iter=0, raise_on_nonconverged=False
        ).return_mapping(st, s_n)
        assert result.converged is False
        assert result.n_iterations == 0
        assert isinstance(result.residual_history, list)

    def test_converged_case_has_flag_true(self, j2):
        st, s_n = _build_plastic_trial(j2)
        result = PythonNumericalIntegrator(j2).return_mapping(st, s_n)
        assert result.converged is True

    def test_stress_update_elastic_always_converged(self, j2):
        deps = anp.array([1e-6, 0, 0, 0, 0, 0])
        r = PythonIntegrator(j2).stress_update(deps, anp.zeros(6), j2.initial_state())
        assert r.converged is True
        assert r.is_plastic is False

    def test_stress_update_forwards_flag(self, j2):
        deps = anp.array([3e-3, 0, 0, 0, 0, 0])
        r = PythonNumericalIntegrator(
            j2, max_iter=0, raise_on_nonconverged=False
        ).stress_update(deps, anp.zeros(6), j2.initial_state())
        assert r.is_plastic is True
        assert r.converged is False
