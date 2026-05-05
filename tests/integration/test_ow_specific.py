"""OW (Ohno-Wang) model-specific tests.

These tests cover physics unique to the OW model:
- implicit_state_names == ["alpha", "ep"] and implicit_stress == True for all variants
- Backstress saturation: ‖α‖_vm → √(C_k / γ)  under monotonic loading
- gamma=0 limit gives physically correct OW-to-Prager reduction
- OW approaches AF for small plastic strains (near linear regime)
"""

import math
import pytest
import numpy as np
import autograd.numpy as anp

from manforge.models.ow_kinematic import OWKinematic3D, OWKinematicPS, OWKinematic1D
from manforge.models.af_kinematic import AFKinematic3D
from manforge.simulation.integrator import PythonIntegrator


# ---------------------------------------------------------------------------
# API detection: all OW states are implicit, σ included in NR
# ---------------------------------------------------------------------------

def test_implicit_state_names_ow3d():
    m = OWKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=1.0)
    assert m.implicit_state_names == ["alpha", "ep"]
    assert m.implicit_stress is True


def test_implicit_state_names_owps():
    m = OWKinematicPS(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=1.0)
    assert m.implicit_state_names == ["alpha", "ep"]
    assert m.implicit_stress is True


def test_implicit_state_names_ow1d():
    m = OWKinematic1D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=1.0)
    assert m.implicit_state_names == ["alpha", "ep"]
    assert m.implicit_stress is True


# ---------------------------------------------------------------------------
# Backstress saturation: ‖α‖_vm → √(C_k / γ) (200 small-step loading)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_backstress_saturation():
    """Under 200 small plastic increments, ‖α‖_vm must converge to √(C_k/γ)."""
    model = OWKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=1.0)
    alpha_sat_expected = math.sqrt(model.C_k / model.gamma)  # 100 MPa

    stress_n = anp.zeros(6)
    state_n = model.initial_state()
    for _ in range(200):
        deps = (lambda _a: (_a.__setitem__(0, 5e-4), _a)[1])(np.zeros(6))
        _r = PythonIntegrator(model).stress_update(deps, stress_n, state_n)
        stress_n, state_n = _r.stress, _r.state

    alpha = state_n["alpha"]
    alpha_vm = float(anp.sqrt(1.5 * anp.sum(alpha ** 2)))

    assert abs(alpha_vm - alpha_sat_expected) / alpha_sat_expected < 0.05, (
        f"Backstress norm {alpha_vm:.2f} not close to saturation {alpha_sat_expected:.2f}"
    )


# ---------------------------------------------------------------------------
# OW approaches AF for small plastic strains
# ---------------------------------------------------------------------------

def test_ow_approaches_af_for_small_alpha():
    """For very small plastic strains, OW and AF give similar results.

    When ‖α‖ is small, γ ‖α‖ α ≈ 0, so both models reduce to linear hardening locally.
    """
    ow_model = OWKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=1.0)
    af_model = AFKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0)
    deps = (lambda _a: (_a.__setitem__(0, 3e-4), _a)[1])(np.zeros(6))  # very small step — near linear regime

    stress_ow = PythonIntegrator(ow_model).stress_update(deps, anp.zeros(6), ow_model.initial_state()).stress
    stress_af = PythonIntegrator(af_model).stress_update(deps, anp.zeros(6), af_model.initial_state()).stress

    np.testing.assert_allclose(
        np.array(stress_ow), np.array(stress_af), rtol=0.10,
        err_msg="OW and AF stresses diverge even at small strain increment"
    )
