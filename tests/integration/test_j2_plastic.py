"""Test return mapping in the plastic domain for J2Isotropic3D.

Analytical solution for uniaxial tension with linear isotropic hardening:

  σ_vm_trial = σ_n + E Δε   (purely uniaxial, elastic trial)
  Δλ         = (σ_vm_trial - σ_y) / (3μ + H)
  σ_new[0]   = σ_vm_trial - (2/3)(3μ) Δλ  = σ_y + H Δλ  (after correction)

where σ_y = σ_y0 + H ep_n.
"""

import numpy as np
import autograd.numpy as anp
import pytest

from manforge.core.stress_update import stress_update


# ---------------------------------------------------------------------------
# Uniaxial plastic step from virgin state
# ---------------------------------------------------------------------------

def test_plastic_stress_uniaxial(model, lame_constants):
    """Plastic uniaxial strain step: σ11_new matches analytic correction.

    Note: strain_inc = [deps11, 0, 0, ...] is a *uniaxial strain* increment,
    which produces a *triaxial* trial stress via the elastic stiffness:
        σ_trial = C @ Δε = [(λ+2μ)deps11, λdeps11, λdeps11, 0, 0, 0]

    The von Mises trial stress is therefore 2μ·deps11 (not E·deps11).
    The analytic σ11 after return mapping is:
        σ11_new = (λ+2μ)deps11 - 2μ·Δλ
    """
    lam, mu = lame_constants
    sigma_y0 = model.sigma_y0
    H = model.H

    deps11 = 2e-3
    strain_inc = anp.array([deps11, 0.0, 0.0, 0.0, 0.0, 0.0])
    stress_n = anp.zeros(6)
    state_n = model.initial_state()

    _r = stress_update(model, strain_inc, stress_n, state_n)
    stress_new, state_new = _r.stress, _r.state

    # Correct analytic Δλ: use vonmises of the triaxial trial stress
    C = model.elastic_stiffness()
    stress_trial = C @ strain_inc
    sigma_vm_trial = float(2.0 * mu * deps11)  # = vonmises(stress_trial)

    dlambda_analytic = (sigma_vm_trial - sigma_y0) / (3.0 * mu + H)
    # Analytic σ11 after radial return: n[0] = 1, C·n[0] = 2μ
    sigma11_analytic = float((lam + 2.0 * mu) * deps11 - 2.0 * mu * dlambda_analytic)

    assert dlambda_analytic > 0, "should be in plastic domain"
    np.testing.assert_allclose(
        float(stress_new[0]), sigma11_analytic, rtol=1e-6,
        err_msg=f"Expected σ11 ≈ {sigma11_analytic:.4f}, got {float(stress_new[0]):.4f}",
    )


def test_plastic_ep_updated(model):
    """Equivalent plastic strain must increase after plastic step."""
    deps11 = 2e-3
    strain_inc = anp.array([deps11, 0.0, 0.0, 0.0, 0.0, 0.0])
    state_n = model.initial_state()

    state_new = stress_update(
        model, strain_inc, anp.zeros(6), state_n
    ).state

    assert float(state_new["ep"]) > 0.0


def test_plastic_ep_matches_dlambda(model, lame_constants):
    """ep_new = Δλ for J2 (Δep = Δλ from consistency condition).

    σ_vm_trial must be computed via vonmises(σ_trial), not σ_trial[0],
    because a uniaxial strain increment produces a triaxial stress state.
    """
    lam, mu = lame_constants
    sigma_y0 = model.sigma_y0
    H = model.H

    deps11 = 2e-3
    strain_inc = anp.array([deps11, 0.0, 0.0, 0.0, 0.0, 0.0])
    state_n = model.initial_state()

    # Correct σ_vm_trial = 2μ·deps11  (for uniaxial strain increment)
    sigma_vm_trial = 2.0 * mu * deps11
    dlambda_analytic = (sigma_vm_trial - sigma_y0) / (3.0 * mu + H)

    state_new = stress_update(
        model, strain_inc, anp.zeros(6), state_n
    ).state

    np.testing.assert_allclose(float(state_new["ep"]), dlambda_analytic, rtol=1e-6)


def test_plastic_yield_consistency(model):
    """After plastic step, updated stress should lie on the yield surface."""
    deps11 = 2e-3
    strain_inc = anp.array([deps11, 0.0, 0.0, 0.0, 0.0, 0.0])
    state_n = model.initial_state()

    _r = stress_update(model, strain_inc, anp.zeros(6), state_n)
    stress_new, state_new = _r.stress, _r.state

    f_final = model.yield_function(stress_new, state_new)
    assert anp.abs(f_final) < 1e-8, f"|f| = {float(anp.abs(f_final)):.3e}"


def test_plastic_ddsdde_differs_from_C(model):
    """In plastic domain, DDSDDE must differ from elastic stiffness."""
    C = model.elastic_stiffness()
    deps11 = 2e-3
    strain_inc = anp.array([deps11, 0.0, 0.0, 0.0, 0.0, 0.0])

    ddsdde = stress_update(
        model, strain_inc, anp.zeros(6), model.initial_state()
    ).ddsdde

    assert not np.allclose(np.asarray(ddsdde), np.asarray(C), atol=1.0), \
        "Plastic tangent should differ from elastic stiffness"
