"""Test return mapping in the elastic domain for J2Isotropic3D."""

import numpy as np
import autograd.numpy as anp

from manforge.core.stress_update import stress_update


# ---------------------------------------------------------------------------
# Elastic step: small strain well within the yield surface
# ---------------------------------------------------------------------------

def test_elastic_stress_update(model, initial_state):
    """Stress update in elastic domain: σ_new = σ_n + C Δε."""
    C = model.elastic_stiffness()
    strain_inc = anp.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])

    _r = stress_update(model, strain_inc, anp.zeros(6), initial_state)
    stress_new, state_new, ddsdde = _r.stress, _r.state, _r.ddsdde

    expected_stress = C @ strain_inc
    np.testing.assert_allclose(np.asarray(stress_new), np.asarray(expected_stress), rtol=1e-10)


def test_elastic_tangent_equals_C(model, initial_state):
    """DDSDDE in elastic domain equals the elastic stiffness tensor."""
    C = model.elastic_stiffness()
    strain_inc = anp.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])

    ddsdde = stress_update(
        model, strain_inc, anp.zeros(6), initial_state
    ).ddsdde

    np.testing.assert_allclose(np.asarray(ddsdde), np.asarray(C), rtol=1e-10)


def test_elastic_state_unchanged(model, initial_state):
    """Internal state must not change in an elastic step."""
    strain_inc = anp.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])

    state_new = stress_update(
        model, strain_inc, anp.zeros(6), initial_state
    ).state

    np.testing.assert_allclose(np.asarray(state_new["ep"]), np.asarray(initial_state["ep"]))


def test_elastic_multiaxial(model, initial_state):
    """Elastic response under multiaxial strain increment."""
    C = model.elastic_stiffness()
    strain_inc = anp.array([5e-6, 3e-6, -2e-6, 1e-6, 0.0, 0.0])

    _r = stress_update(model, strain_inc, anp.zeros(6), initial_state)
    stress_new, ddsdde = _r.stress, _r.ddsdde

    np.testing.assert_allclose(np.asarray(stress_new), np.asarray(C @ strain_inc), rtol=1e-10)
    np.testing.assert_allclose(np.asarray(ddsdde), np.asarray(C), rtol=1e-10)


def test_elastic_from_prestress(model):
    """Elastic step from a non-zero pre-stress within yield surface."""
    C = model.elastic_stiffness()

    # Pre-stress: 100 MPa uniaxial (well below σ_y0 = 250 MPa)
    stress_n = anp.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    state_n = model.initial_state()
    strain_inc = anp.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])

    _r = stress_update(model, strain_inc, stress_n, state_n)
    stress_new, ddsdde = _r.stress, _r.ddsdde

    np.testing.assert_allclose(np.asarray(stress_new), np.asarray(stress_n + C @ strain_inc), rtol=1e-10)
    np.testing.assert_allclose(np.asarray(ddsdde), np.asarray(C), rtol=1e-10)
