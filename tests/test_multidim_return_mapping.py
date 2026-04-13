"""Integration tests for multi-dimensionality: PLANE_STRAIN and constructor guards.

Covers:
- J2Isotropic3D constructor accepts SOLID_3D / PLANE_STRAIN, rejects others
- PLANE_STRAIN elastic step: shapes, stress == C@deps, tangent == C
- PLANE_STRAIN plastic step: yield consistency, ep > 0
- PLANE_STRAIN analytical tangent: finite-difference verification
- PLANE_STRAIN analytical vs autodiff cross-check
- Driver integration with 4-component arrays (UniaxialDriver, GeneralDriver)
- Plane-strain signature: sigma_33 != 0 under axial loading
- J2Isotropic3D(PLANE_STRAIN) autodiff path works correctly
- J2IsotropicPS (no plastic_corrector) raises NotImplementedError for method='analytical'
"""

import jax.numpy as jnp
import numpy as np
import pytest

from manforge.core.return_mapping import return_mapping
from manforge.core.stress_state import SOLID_3D, PLANE_STRAIN, PLANE_STRESS, UNIAXIAL_1D
from manforge.models.j2_isotropic import J2Isotropic3D, J2IsotropicPS
from manforge.simulation.driver import StrainDriver
from manforge.simulation.types import FieldHistory, FieldType
from manforge.verification.fd_check import check_tangent


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_j2isotropic3d_accepts_solid_3d():
    model = J2Isotropic3D(SOLID_3D)
    assert model.stress_state is SOLID_3D
    assert model.ntens == 6


def test_j2isotropic3d_accepts_plane_strain():
    model = J2Isotropic3D(PLANE_STRAIN)
    assert model.stress_state is PLANE_STRAIN
    assert model.ntens == 4


def test_j2isotropic3d_rejects_plane_stress():
    with pytest.raises(ValueError, match="ndi == ndi_phys"):
        J2Isotropic3D(PLANE_STRESS)


def test_j2isotropic3d_rejects_uniaxial_1d():
    with pytest.raises(ValueError, match="ndi == ndi_phys"):
        J2Isotropic3D(UNIAXIAL_1D)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pe_model():
    return J2Isotropic3D(PLANE_STRAIN)


@pytest.fixture
def pe_state(pe_model):
    return pe_model.initial_state()


# ---------------------------------------------------------------------------
# Elastic step — shape and value
# ---------------------------------------------------------------------------

def test_elastic_step_shapes(pe_model, pe_state, steel_params):
    """Elastic step produces stress (4,) and tangent (4, 4)."""
    deps = jnp.array([1e-4, 0.0, 0.0, 0.0])
    stress, state, ddsdde = return_mapping(
        pe_model, deps, jnp.zeros(4), pe_state, steel_params
    )
    assert stress.shape == (4,)
    assert ddsdde.shape == (4, 4)


def test_elastic_step_stress_equals_C_deps(pe_model, pe_state, steel_params):
    """Elastic stress must equal C @ deps."""
    deps = jnp.array([1e-4, 0.0, 0.0, 0.0])
    C = pe_model.elastic_stiffness(steel_params)
    stress, _, _ = return_mapping(
        pe_model, deps, jnp.zeros(4), pe_state, steel_params
    )
    np.testing.assert_allclose(np.asarray(stress), np.asarray(C @ deps), rtol=1e-10)


def test_elastic_step_tangent_equals_C(pe_model, pe_state, steel_params):
    """Elastic tangent must equal the elastic stiffness C."""
    deps = jnp.array([1e-4, 0.0, 0.0, 0.0])
    C = pe_model.elastic_stiffness(steel_params)
    _, _, ddsdde = return_mapping(
        pe_model, deps, jnp.zeros(4), pe_state, steel_params
    )
    np.testing.assert_allclose(np.asarray(ddsdde), np.asarray(C), rtol=1e-10)


# ---------------------------------------------------------------------------
# Plastic step — yield consistency and ep update
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strain_inc_vec", [
    [2e-3, 0.0, 0.0, 0.0],
    [2e-3, -1e-3, -1e-3, 0.0],   # isochoric ×2 — vm≈485 MPa > sigma_y0
    [0.0, 0.0, 0.0, 2e-3],
    [2e-3, 1e-3, 4e-4, 2e-3],    # mixed ×2 — vm≈360 MPa > sigma_y0
])
def test_plastic_yield_consistency(pe_model, pe_state, steel_params, strain_inc_vec):
    """Plastic step: yield function ≈ 0 at converged state."""
    deps = jnp.array(strain_inc_vec)
    stress, state, _ = return_mapping(
        pe_model, deps, jnp.zeros(4), pe_state, steel_params
    )
    f = pe_model.yield_function(stress, state, steel_params)
    assert abs(float(f)) < 1e-8, f"|f| = {abs(float(f)):.3e}"


@pytest.mark.parametrize("strain_inc_vec", [
    [2e-3, 0.0, 0.0, 0.0],
    [2e-3, -1e-3, -1e-3, 0.0],   # isochoric ×2 — vm≈485 MPa > sigma_y0
    [0.0, 0.0, 0.0, 2e-3],
    [2e-3, 1e-3, 4e-4, 2e-3],    # mixed ×2 — vm≈360 MPa > sigma_y0
])
def test_plastic_ep_positive(pe_model, pe_state, steel_params, strain_inc_vec):
    """Plastic step: equivalent plastic strain must increase."""
    deps = jnp.array(strain_inc_vec)
    _, state, _ = return_mapping(
        pe_model, deps, jnp.zeros(4), pe_state, steel_params
    )
    assert float(state["ep"]) > 0.0


# ---------------------------------------------------------------------------
# Analytical tangent — finite-difference verification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strain_inc_vec", [
    [2e-3, 0.0, 0.0, 0.0],
    [1e-3, -5e-4, -5e-4, 0.0],
    [0.0, 0.0, 0.0, 2e-3],
    [1e-3, 5e-4, 2e-4, 1e-3],
])
def test_analytical_tangent_fd_check(pe_model, pe_state, steel_params, strain_inc_vec):
    """Plane-strain analytical tangent passes finite-difference check."""
    result = check_tangent(
        pe_model,
        jnp.zeros(4),
        pe_state,
        steel_params,
        jnp.array(strain_inc_vec),
        method="analytical",
    )
    assert result.passed, f"FD check failed: max_rel_err = {result.max_rel_err:.3e}"


# ---------------------------------------------------------------------------
# Analytical vs autodiff cross-check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strain_inc_vec", [
    [2e-3, 0.0, 0.0, 0.0],
    [1e-3, -5e-4, -5e-4, 0.0],
    [0.0, 0.0, 0.0, 2e-3],
    [1e-3, 5e-4, 2e-4, 1e-3],
])
def test_analytical_stress_matches_autodiff(pe_model, pe_state, steel_params, strain_inc_vec):
    """Analytical and autodiff stress must agree to atol=1e-6."""
    deps = jnp.array(strain_inc_vec)
    s_ad, _, _ = return_mapping(
        pe_model, deps, jnp.zeros(4), pe_state, steel_params, method="autodiff"
    )
    s_an, _, _ = return_mapping(
        pe_model, deps, jnp.zeros(4), pe_state, steel_params, method="analytical"
    )
    np.testing.assert_allclose(
        np.asarray(s_an), np.asarray(s_ad), atol=1e-6,
        err_msg=f"max stress diff = {float(jnp.max(jnp.abs(s_an - s_ad))):.3e}",
    )


@pytest.mark.parametrize("strain_inc_vec", [
    [2e-3, 0.0, 0.0, 0.0],
    [1e-3, -5e-4, -5e-4, 0.0],
    [0.0, 0.0, 0.0, 2e-3],
])
def test_analytical_tangent_matches_autodiff(pe_model, pe_state, steel_params, strain_inc_vec):
    """Analytical and autodiff tangent must agree within 1e-5 relative error."""
    deps = jnp.array(strain_inc_vec)
    _, _, D_ad = return_mapping(
        pe_model, deps, jnp.zeros(4), pe_state, steel_params, method="autodiff"
    )
    _, _, D_an = return_mapping(
        pe_model, deps, jnp.zeros(4), pe_state, steel_params, method="analytical"
    )
    rel_err = jnp.abs(D_an - D_ad) / (jnp.abs(D_ad) + 1.0)
    assert float(jnp.max(rel_err)) < 1e-5, \
        f"max tangent rel err = {float(jnp.max(rel_err)):.3e}"


# ---------------------------------------------------------------------------
# Driver integration
# ---------------------------------------------------------------------------

def test_uniaxial_driver_plane_strain(pe_model, steel_params):
    """StrainDriver (uniaxial) works with a PLANE_STRAIN model."""
    eps_history = np.linspace(0, 5e-3, 20)
    load = FieldHistory(FieldType.STRAIN, "Strain", eps_history)
    result = StrainDriver().run(pe_model, load, steel_params)
    assert result.stress.shape == (20, 4)
    # σ11 must increase monotonically for hardening material
    assert np.all(np.diff(result.stress[:, 0]) >= 0)


def test_general_driver_plane_strain_shapes(pe_model, steel_params):
    """StrainDriver (general) produces (N, 4) stress output for PLANE_STRAIN model."""
    N = 15
    strain_history = np.zeros((N, 4))
    strain_history[:, 0] = np.linspace(0, 5e-3, N)  # ramp eps_11
    load = FieldHistory(FieldType.STRAIN, "Strain", strain_history)
    result = StrainDriver().run(pe_model, load, steel_params)
    assert result.stress.shape == (N, 4)


def test_plane_strain_sigma33_nonzero(pe_model, steel_params):
    """Plane-strain constraint produces non-zero sigma_33 under axial loading."""
    eps_history = np.zeros((10, 4))
    eps_history[:, 0] = np.linspace(0, 5e-3, 10)  # ramp eps_11 only
    load = FieldHistory(FieldType.STRAIN, "Strain", eps_history)
    result = StrainDriver().run(pe_model, load, steel_params)
    # sigma_33 (index 2) must be non-zero due to plane-strain lateral constraint
    assert np.any(np.abs(result.stress[:, 2]) > 1.0), \
        f"sigma_33 unexpectedly near zero: {result.stress[:, 2]}"


# ---------------------------------------------------------------------------
# Autodiff path and analytical-raises behavior
# ---------------------------------------------------------------------------

def test_j2isotropic3d_autodiff_plane_strain(pe_state, steel_params):
    """J2Isotropic3D(PLANE_STRAIN) with method='autodiff' works correctly."""
    model = J2Isotropic3D(PLANE_STRAIN)
    deps = jnp.array([2e-3, 0.0, 0.0, 0.0])
    stress, state, ddsdde = return_mapping(
        model, deps, jnp.zeros(4), pe_state, steel_params, method="autodiff"
    )
    assert stress.shape == (4,)
    assert ddsdde.shape == (4, 4)
    # Yield consistency
    f = model.yield_function(stress, state, steel_params)
    assert abs(float(f)) < 1e-8


def test_autodiff_only_model_analytical_raises(steel_params):
    """J2IsotropicPS (no plastic_corrector) raises NotImplementedError for method='analytical'."""
    model = J2IsotropicPS()
    deps = jnp.array([2e-3, 0.0, 0.0])
    state0 = model.initial_state()
    with pytest.raises(NotImplementedError):
        return_mapping(
            model, deps, jnp.zeros(3), state0, steel_params, method="analytical"
        )
