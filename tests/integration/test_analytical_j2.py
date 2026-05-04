"""Tests for the J2Isotropic3D analytical return-mapping path.

Covers:
- plastic_corrector and analytical_tangent as standalone methods
- PythonAnalyticalIntegrator vs PythonNumericalIntegrator agreement
- PythonIntegrator (auto) selects the analytical path
- check_tangent with PythonAnalyticalIntegrator (FD verification of closed-form tangent)
- PythonAnalyticalIntegrator raises NotImplementedError on a model without hooks
"""

import numpy as np
import autograd.numpy as anp
import pytest

from manforge.core.material import MaterialModel3D
from manforge.simulation.integrator import (
    PythonIntegrator,
    PythonNumericalIntegrator,
    PythonAnalyticalIntegrator,
)
from manforge.verification.fd_check import check_tangent


# ---------------------------------------------------------------------------
# plastic_corrector — standalone
# ---------------------------------------------------------------------------

def test_plastic_corrector_elastic_path_not_called(model):
    """plastic_corrector is only invoked in the plastic regime.

    When return_mapping detects an elastic step (f_trial ≤ 0), it returns
    before calling plastic_corrector.  This test verifies the elastic step
    still works under PythonAnalyticalIntegrator.
    """
    strain_inc = anp.array([1e-4, 0.0, 0.0, 0.0, 0.0, 0.0])  # tiny, stays elastic
    stress_n = anp.zeros(6)
    state_n = model.initial_state()

    _r = PythonAnalyticalIntegrator(model).stress_update(strain_inc, stress_n, state_n)
    stress_new, state_new, ddsdde = _r.stress, _r.state, _r.ddsdde
    C = model.elastic_stiffness()
    np.testing.assert_allclose(np.asarray(stress_new), np.asarray(C @ strain_inc), rtol=1e-10)
    np.testing.assert_allclose(np.asarray(ddsdde), np.asarray(C), rtol=1e-10)


def test_plastic_corrector_standalone_plastic(model):
    """plastic_corrector returns correct (stress, state, dlambda) for a plastic step."""
    C = model.elastic_stiffness()
    deps11 = 2e-3
    strain_inc = anp.array([deps11, 0.0, 0.0, 0.0, 0.0, 0.0])
    stress_trial = C @ strain_inc
    state_n = model.initial_state()

    rm = model.user_defined_return_mapping(stress_trial, C, state_n)
    assert rm is not None, "plastic_corrector should return a result for plastic step"

    # Yield consistency: f(σ_new, state_new) ≈ 0
    f_final = model.yield_function(rm.stress, rm.state)
    assert abs(float(f_final)) < 1e-8, f"|f| = {float(abs(f_final)):.3e}"

    # State update: ep_new = ep_n + dlambda
    ep_n = float(state_n["ep"])
    assert abs(float(rm.state["ep"]) - (ep_n + float(rm.dlambda))) < 1e-12


def test_plastic_corrector_dlambda_positive(model):
    """Δλ must be positive for a genuinely plastic increment."""
    C = model.elastic_stiffness()
    strain_inc = anp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    stress_trial = C @ strain_inc
    state_n = model.initial_state()

    rm = model.user_defined_return_mapping(stress_trial, C, state_n)
    assert float(rm.dlambda) > 0.0


# ---------------------------------------------------------------------------
# PythonAnalyticalIntegrator vs PythonNumericalIntegrator — stress and state
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strain_inc_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],          # uniaxial
    [1e-3, -5e-4, -5e-4, 0.0, 0.0, 0.0],       # triaxial isochoric
    [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],           # shear
    [1e-3, 5e-4, 2e-4, 1e-3, 5e-4, 2e-4],      # mixed
])
def test_analytical_stress_matches_autodiff(model, strain_inc_vec):
    """PythonAnalyticalIntegrator stress must match PythonNumericalIntegrator to tight tolerance."""
    strain_inc = anp.array(strain_inc_vec)
    stress_n = anp.zeros(6)
    state_n = model.initial_state()

    _r_ad = PythonNumericalIntegrator(model).stress_update(strain_inc, stress_n, state_n)
    s_ad, st_ad = _r_ad.stress, _r_ad.state
    _r_an = PythonAnalyticalIntegrator(model).stress_update(strain_inc, stress_n, state_n)
    s_an, st_an = _r_an.stress, _r_an.state

    np.testing.assert_allclose(
        np.asarray(s_an), np.asarray(s_ad), atol=1e-6,
        err_msg=f"max stress diff = {float(anp.max(anp.abs(s_an - s_ad))):.3e}",
    )
    assert abs(float(st_an["ep"]) - float(st_ad["ep"])) < 1e-10


def test_analytical_stress_matches_autodiff_nonzero_initial_stress(model):
    """Works correctly from a pre-stressed state."""
    # First step to build up pre-stress
    strain_inc1 = anp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    _r1 = PythonIntegrator(model).stress_update(strain_inc1, anp.zeros(6), model.initial_state())
    s1, st1 = _r1.stress, _r1.state

    # Second plastic step
    strain_inc2 = anp.array([1e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    s_ad = PythonNumericalIntegrator(model).stress_update(strain_inc2, s1, st1).stress
    s_an = PythonAnalyticalIntegrator(model).stress_update(strain_inc2, s1, st1).stress

    np.testing.assert_allclose(np.asarray(s_an), np.asarray(s_ad), atol=1e-6)


# ---------------------------------------------------------------------------
# PythonAnalyticalIntegrator vs PythonNumericalIntegrator — tangent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strain_inc_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1e-3, -5e-4, -5e-4, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
])
def test_analytical_tangent_matches_autodiff(model, strain_inc_vec):
    """Analytical DDSDDE must match autodiff DDSDDE to 1e-5 relative tolerance."""
    strain_inc = anp.array(strain_inc_vec)
    stress_n = anp.zeros(6)
    state_n = model.initial_state()

    D_ad = PythonNumericalIntegrator(model).stress_update(strain_inc, stress_n, state_n).ddsdde
    D_an = PythonAnalyticalIntegrator(model).stress_update(strain_inc, stress_n, state_n).ddsdde

    rel_err = anp.abs(D_an - D_ad) / (anp.abs(D_ad) + 1.0)
    assert float(anp.max(rel_err)) < 1e-5, \
        f"max tangent rel err = {float(anp.max(rel_err)):.3e}"


# ---------------------------------------------------------------------------
# PythonIntegrator (auto) uses the analytical path when available
# ---------------------------------------------------------------------------

def test_method_auto_uses_analytical(model):
    """PythonIntegrator should use plastic_corrector when it returns non-None."""
    strain_inc = anp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    stress_n = anp.zeros(6)
    state_n = model.initial_state()

    _r_auto = PythonIntegrator(model).stress_update(strain_inc, stress_n, state_n)
    s_auto, D_auto = _r_auto.stress, _r_auto.ddsdde
    _r_an = PythonAnalyticalIntegrator(model).stress_update(strain_inc, stress_n, state_n)
    s_an, D_an = _r_an.stress, _r_an.ddsdde

    np.testing.assert_allclose(np.asarray(s_auto), np.asarray(s_an), atol=1e-12,
                               err_msg="auto should match analytical path")
    np.testing.assert_allclose(np.asarray(D_auto), np.asarray(D_an), atol=1e-12,
                               err_msg="auto should match analytical tangent")


# ---------------------------------------------------------------------------
# Finite-difference verification of analytical tangent
# ---------------------------------------------------------------------------

def test_analytical_tangent_fd_check_elastic(model):
    """Elastic step: analytical tangent = C = FD tangent."""
    result = check_tangent(
        PythonAnalyticalIntegrator(model),
        anp.zeros(6),
        model.initial_state(),
        anp.array([1e-4, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    assert result.passed, f"FD check failed: max_rel_err = {result.max_rel_err:.3e}"


@pytest.mark.parametrize("strain_inc_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1e-3, -5e-4, -5e-4, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
])
def test_analytical_tangent_fd_check_plastic(model, strain_inc_vec):
    """Plastic step: analytical tangent passes finite-difference check."""
    result = check_tangent(
        PythonAnalyticalIntegrator(model),
        anp.zeros(6),
        model.initial_state(),
        anp.array(strain_inc_vec),
    )
    assert result.passed, f"FD check failed: max_rel_err = {result.max_rel_err:.3e}"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_method_analytical_raises_if_no_hooks():
    """PythonAnalyticalIntegrator raises NotImplementedError for a model without hooks."""

    class MinimalModel(MaterialModel3D):
        param_names = ["E", "nu", "sigma_y0"]
        state_names = []

        def __init__(self):
            super().__init__()
            self.E = 210000.0
            self.nu = 0.3
            self.sigma_y0 = 250.0

        def elastic_stiffness(self):
            mu = self.E / (2.0 * (1.0 + self.nu))
            lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
            return self.isotropic_C(lam, mu)

        def yield_function(self, stress, state):
            return self._vonmises(stress) - self.sigma_y0

        def update_state(self, dlambda, stress, state):
            return {}

    minimal_model = MinimalModel()
    strain_inc = anp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])

    with pytest.raises(NotImplementedError):
        PythonAnalyticalIntegrator(minimal_model).stress_update(
            strain_inc, anp.zeros(6), {}
        )


# ---------------------------------------------------------------------------
# FD tangent verification (J2 numerical path)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strain_inc_vec,stress_n_fn,description", [
    ([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0], lambda m: anp.zeros(6), "elastic"),
    ([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0], lambda m: anp.zeros(6), "plastic_uniaxial"),
    ([1.5e-3, -0.5e-3, -0.5e-3, 0.5e-3, 0.0, 0.0], lambda m: anp.zeros(6), "plastic_multiaxial"),
    ([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
     lambda m: anp.array([m.sigma_y0 / 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "plastic_prestress"),
])
def test_j2_fd_tangent(model, initial_state, strain_inc_vec, stress_n_fn, description):
    """J2 consistent tangent (numerical NR path) matches central-difference FD."""
    stress_n = stress_n_fn(model)
    state_n = model.initial_state() if stress_n[0] == 0.0 else initial_state
    result = check_tangent(
        PythonNumericalIntegrator(model), stress_n, state_n, anp.array(strain_inc_vec)
    )
    assert result.passed, (
        f"[{description}] FD tangent check failed: max_rel_err = {result.max_rel_err:.3e}"
    )
