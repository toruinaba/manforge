"""Test return mapping in the elastic domain for J2Isotropic3D."""

import numpy as np
import jax.numpy as jnp

from manforge.core.return_mapping import return_mapping


# ---------------------------------------------------------------------------
# Elastic step: small strain well within the yield surface
# ---------------------------------------------------------------------------

def test_elastic_stress_update(model, initial_state):
    """Stress update in elastic domain: σ_new = σ_n + C Δε."""
    C = model.elastic_stiffness()
    strain_inc = jnp.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])

    stress_new, state_new, ddsdde = return_mapping(
        model, strain_inc, jnp.zeros(6), initial_state
    )

    expected_stress = C @ strain_inc
    np.testing.assert_allclose(np.asarray(stress_new), np.asarray(expected_stress), rtol=1e-10)


def test_elastic_tangent_equals_C(model, initial_state):
    """DDSDDE in elastic domain equals the elastic stiffness tensor."""
    C = model.elastic_stiffness()
    strain_inc = jnp.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])

    _, _, ddsdde = return_mapping(
        model, strain_inc, jnp.zeros(6), initial_state
    )

    np.testing.assert_allclose(np.asarray(ddsdde), np.asarray(C), rtol=1e-10)


def test_elastic_state_unchanged(model, initial_state):
    """Internal state must not change in an elastic step."""
    strain_inc = jnp.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])

    _, state_new, _ = return_mapping(
        model, strain_inc, jnp.zeros(6), initial_state
    )

    np.testing.assert_allclose(np.asarray(state_new["ep"]), np.asarray(initial_state["ep"]))


def test_elastic_multiaxial(model, initial_state):
    """Elastic response under multiaxial strain increment."""
    C = model.elastic_stiffness()
    strain_inc = jnp.array([5e-6, 3e-6, -2e-6, 1e-6, 0.0, 0.0])

    stress_new, _, ddsdde = return_mapping(
        model, strain_inc, jnp.zeros(6), initial_state
    )

    np.testing.assert_allclose(np.asarray(stress_new), np.asarray(C @ strain_inc), rtol=1e-10)
    np.testing.assert_allclose(np.asarray(ddsdde), np.asarray(C), rtol=1e-10)


def test_elastic_from_prestress(model):
    """Elastic step from a non-zero pre-stress within yield surface."""
    C = model.elastic_stiffness()

    # Pre-stress: 100 MPa uniaxial (well below σ_y0 = 250 MPa)
    stress_n = jnp.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    state_n = model.initial_state()
    strain_inc = jnp.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])

    stress_new, _, ddsdde = return_mapping(
        model, strain_inc, stress_n, state_n
    )

    np.testing.assert_allclose(np.asarray(stress_new), np.asarray(stress_n + C @ strain_inc), rtol=1e-10)
    np.testing.assert_allclose(np.asarray(ddsdde), np.asarray(C), rtol=1e-10)
