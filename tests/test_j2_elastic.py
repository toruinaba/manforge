"""Test return mapping in the elastic domain for J2Isotropic3D."""

import jax.numpy as jnp
import pytest

import manforge  # noqa: F401
from manforge.core.return_mapping import return_mapping
from manforge.models.j2_isotropic import J2Isotropic3D


@pytest.fixture
def model():
    return J2Isotropic3D()


@pytest.fixture
def initial_state(model):
    return model.initial_state()


# ---------------------------------------------------------------------------
# Elastic step: small strain well within the yield surface
# ---------------------------------------------------------------------------

def test_elastic_stress_update(model, steel_params, initial_state):
    """Stress update in elastic domain: σ_new = σ_n + C Δε."""
    C = model.elastic_stiffness(steel_params)
    strain_inc = jnp.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])

    stress_new, state_new, ddsdde = return_mapping(
        model, strain_inc, jnp.zeros(6), initial_state, steel_params
    )

    expected_stress = C @ strain_inc
    assert jnp.allclose(stress_new, expected_stress, rtol=1e-10)


def test_elastic_tangent_equals_C(model, steel_params, initial_state):
    """DDSDDE in elastic domain equals the elastic stiffness tensor."""
    C = model.elastic_stiffness(steel_params)
    strain_inc = jnp.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])

    _, _, ddsdde = return_mapping(
        model, strain_inc, jnp.zeros(6), initial_state, steel_params
    )

    assert jnp.allclose(ddsdde, C, rtol=1e-10)


def test_elastic_state_unchanged(model, steel_params, initial_state):
    """Internal state must not change in an elastic step."""
    strain_inc = jnp.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])

    _, state_new, _ = return_mapping(
        model, strain_inc, jnp.zeros(6), initial_state, steel_params
    )

    assert jnp.allclose(state_new["ep"], initial_state["ep"])


def test_elastic_multiaxial(model, steel_params, initial_state):
    """Elastic response under multiaxial strain increment."""
    C = model.elastic_stiffness(steel_params)
    strain_inc = jnp.array([5e-6, 3e-6, -2e-6, 1e-6, 0.0, 0.0])

    stress_new, _, ddsdde = return_mapping(
        model, strain_inc, jnp.zeros(6), initial_state, steel_params
    )

    assert jnp.allclose(stress_new, C @ strain_inc, rtol=1e-10)
    assert jnp.allclose(ddsdde, C, rtol=1e-10)


def test_elastic_from_prestress(model, steel_params):
    """Elastic step from a non-zero pre-stress within yield surface."""
    C = model.elastic_stiffness(steel_params)

    # Pre-stress: 100 MPa uniaxial (well below σ_y0 = 250 MPa)
    stress_n = jnp.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    state_n = model.initial_state()
    strain_inc = jnp.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])

    stress_new, _, ddsdde = return_mapping(
        model, strain_inc, stress_n, state_n, steel_params
    )

    assert jnp.allclose(stress_new, stress_n + C @ strain_inc, rtol=1e-10)
    assert jnp.allclose(ddsdde, C, rtol=1e-10)
