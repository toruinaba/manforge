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
- J2IsotropicPS (no plastic_corrector) raises NotImplementedError for method='user_defined'
"""

import autograd.numpy as anp
import numpy as np
import pytest

from manforge.core.stress_update import stress_update
from manforge.core.stress_state import SOLID_3D, PLANE_STRAIN, PLANE_STRESS, UNIAXIAL_1D
from manforge.models.j2_isotropic import J2Isotropic3D, J2IsotropicPS
from manforge.simulation.driver import StrainDriver
from manforge.simulation.types import FieldHistory, FieldType
from manforge.verification.fd_check import check_tangent


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_j2isotropic3d_accepts_solid_3d():
    model = J2Isotropic3D(SOLID_3D, E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
    assert model.stress_state is SOLID_3D
    assert model.ntens == 6


def test_j2isotropic3d_accepts_plane_strain():
    model = J2Isotropic3D(PLANE_STRAIN, E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
    assert model.stress_state is PLANE_STRAIN
    assert model.ntens == 4


def test_j2isotropic3d_rejects_plane_stress():
    with pytest.raises(ValueError, match="ndi == ndi_phys"):
        J2Isotropic3D(PLANE_STRESS, E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)


def test_j2isotropic3d_rejects_uniaxial_1d():
    with pytest.raises(ValueError, match="ndi == ndi_phys"):
        J2Isotropic3D(UNIAXIAL_1D, E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pe_model():
    return J2Isotropic3D(PLANE_STRAIN, E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)


@pytest.fixture
def pe_state(pe_model):
    return pe_model.initial_state()


# ---------------------------------------------------------------------------
# Elastic step — shape and value
# ---------------------------------------------------------------------------

def test_elastic_step_shapes(pe_model, pe_state):
    """Elastic step produces stress (4,) and tangent (4, 4)."""
    deps = anp.array([1e-4, 0.0, 0.0, 0.0])
    _r = stress_update(pe_model, deps, anp.zeros(4), pe_state)
    stress, state, ddsdde = _r.stress, _r.state, _r.ddsdde
    assert stress.shape == (4,)
    assert ddsdde.shape == (4, 4)


def test_elastic_step_stress_equals_C_deps(pe_model, pe_state):
    """Elastic stress must equal C @ deps."""
    deps = anp.array([1e-4, 0.0, 0.0, 0.0])
    C = pe_model.elastic_stiffness()
    stress = stress_update(
        pe_model, deps, anp.zeros(4), pe_state
    ).stress
    np.testing.assert_allclose(np.asarray(stress), np.asarray(C @ deps), rtol=1e-10)


def test_elastic_step_tangent_equals_C(pe_model, pe_state):
    """Elastic tangent must equal the elastic stiffness C."""
    deps = anp.array([1e-4, 0.0, 0.0, 0.0])
    C = pe_model.elastic_stiffness()
    ddsdde = stress_update(
        pe_model, deps, anp.zeros(4), pe_state
    ).ddsdde
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
def test_plastic_yield_consistency(pe_model, pe_state, strain_inc_vec):
    """Plastic step: yield function ≈ 0 at converged state."""
    deps = anp.array(strain_inc_vec)
    _r = stress_update(pe_model, deps, anp.zeros(4), pe_state)
    stress, state = _r.stress, _r.state
    f = pe_model.yield_function(stress, state)
    assert abs(float(f)) < 1e-8, f"|f| = {abs(float(f)):.3e}"


@pytest.mark.parametrize("strain_inc_vec", [
    [2e-3, 0.0, 0.0, 0.0],
    [2e-3, -1e-3, -1e-3, 0.0],   # isochoric ×2 — vm≈485 MPa > sigma_y0
    [0.0, 0.0, 0.0, 2e-3],
    [2e-3, 1e-3, 4e-4, 2e-3],    # mixed ×2 — vm≈360 MPa > sigma_y0
])
def test_plastic_ep_positive(pe_model, pe_state, strain_inc_vec):
    """Plastic step: equivalent plastic strain must increase."""
    deps = anp.array(strain_inc_vec)
    state = stress_update(
        pe_model, deps, anp.zeros(4), pe_state
    ).state
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
def test_analytical_tangent_fd_check(pe_model, pe_state, strain_inc_vec):
    """Plane-strain analytical tangent passes finite-difference check."""
    result = check_tangent(
        pe_model,
        anp.zeros(4),
        pe_state,
        anp.array(strain_inc_vec),
        method="user_defined",
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
def test_analytical_stress_matches_autodiff(pe_model, pe_state, strain_inc_vec):
    """Analytical and autodiff stress must agree to atol=1e-6."""
    deps = anp.array(strain_inc_vec)
    s_ad = stress_update(
        pe_model, deps, anp.zeros(4), pe_state, method="numerical_newton"
    ).stress
    s_an = stress_update(
        pe_model, deps, anp.zeros(4), pe_state, method="user_defined"
    ).stress
    np.testing.assert_allclose(
        np.asarray(s_an), np.asarray(s_ad), atol=1e-6,
        err_msg=f"max stress diff = {float(anp.max(anp.abs(s_an - s_ad))):.3e}",
    )


@pytest.mark.parametrize("strain_inc_vec", [
    [2e-3, 0.0, 0.0, 0.0],
    [1e-3, -5e-4, -5e-4, 0.0],
    [0.0, 0.0, 0.0, 2e-3],
])
def test_analytical_tangent_matches_autodiff(pe_model, pe_state, strain_inc_vec):
    """Analytical and autodiff tangent must agree within 1e-5 relative error."""
    deps = anp.array(strain_inc_vec)
    D_ad = stress_update(
        pe_model, deps, anp.zeros(4), pe_state, method="numerical_newton"
    ).ddsdde
    D_an = stress_update(
        pe_model, deps, anp.zeros(4), pe_state, method="user_defined"
    ).ddsdde
    rel_err = anp.abs(D_an - D_ad) / (anp.abs(D_ad) + 1.0)
    assert float(anp.max(rel_err)) < 1e-5, \
        f"max tangent rel err = {float(anp.max(rel_err)):.3e}"


# ---------------------------------------------------------------------------
# Driver integration
# ---------------------------------------------------------------------------

def test_uniaxial_driver_plane_strain(pe_model):
    """StrainDriver (uniaxial) works with a PLANE_STRAIN model."""
    eps_history = np.linspace(0, 5e-3, 20)
    load = FieldHistory(FieldType.STRAIN, "Strain", eps_history)
    result = StrainDriver().run(pe_model, load)
    assert result.stress.shape == (20, 4)
    # σ11 must increase monotonically for hardening material
    assert np.all(np.diff(result.stress[:, 0]) >= 0)


def test_general_driver_plane_strain_shapes(pe_model):
    """StrainDriver (general) produces (N, 4) stress output for PLANE_STRAIN model."""
    N = 15
    strain_history = np.zeros((N, 4))
    strain_history[:, 0] = np.linspace(0, 5e-3, N)  # ramp eps_11
    load = FieldHistory(FieldType.STRAIN, "Strain", strain_history)
    result = StrainDriver().run(pe_model, load)
    assert result.stress.shape == (N, 4)


def test_plane_strain_sigma33_nonzero(pe_model):
    """Plane-strain constraint produces non-zero sigma_33 under axial loading."""
    eps_history = np.zeros((10, 4))
    eps_history[:, 0] = np.linspace(0, 5e-3, 10)  # ramp eps_11 only
    load = FieldHistory(FieldType.STRAIN, "Strain", eps_history)
    result = StrainDriver().run(pe_model, load)
    # sigma_33 (index 2) must be non-zero due to plane-strain lateral constraint
    assert np.any(np.abs(result.stress[:, 2]) > 1.0), \
        f"sigma_33 unexpectedly near zero: {result.stress[:, 2]}"


# ---------------------------------------------------------------------------
# Autodiff path and analytical-raises behavior
# ---------------------------------------------------------------------------

def test_j2isotropic3d_autodiff_plane_strain(pe_state):
    """J2Isotropic3D(PLANE_STRAIN) with method='numerical_newton' works correctly."""
    model = J2Isotropic3D(PLANE_STRAIN, E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
    deps = anp.array([2e-3, 0.0, 0.0, 0.0])
    _r = stress_update(model, deps, anp.zeros(4), pe_state, method="numerical_newton")
    stress, state, ddsdde = _r.stress, _r.state, _r.ddsdde
    assert stress.shape == (4,)
    assert ddsdde.shape == (4, 4)
    # Yield consistency
    f = model.yield_function(stress, state)
    assert abs(float(f)) < 1e-8


def test_autodiff_only_model_analytical_raises():
    """J2IsotropicPS (no plastic_corrector) raises NotImplementedError for method='user_defined'."""
    model = J2IsotropicPS(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
    deps = anp.array([2e-3, 0.0, 0.0])
    state0 = model.initial_state()
    with pytest.raises(NotImplementedError):
        stress_update(
            model, deps, anp.zeros(3), state0, method="user_defined"
        )
