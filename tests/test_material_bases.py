"""Tests for stress-state base classes (MaterialModel3D, ...).

Covers:
- Constructor validation (accepted / rejected StressStates)
- _hydrostatic, _dev, _vonmises operators — values and shapes
- isotropic_C — shape, symmetry, known values
- _I_vol, _I_dev — projection identity: P_vol + P_dev == I
"""

import jax.numpy as jnp
import numpy as np
import pytest

import manforge  # noqa: F401
from manforge.core.material import MaterialModel3D
from manforge.core.stress_state import SOLID_3D, PLANE_STRAIN, PLANE_STRESS, UNIAXIAL_1D


# ---------------------------------------------------------------------------
# Minimal concrete subclass for testing (does not implement material physics)
# ---------------------------------------------------------------------------

class _Stub3D(MaterialModel3D):
    """Concrete stub — lets us instantiate MaterialModel3D for operator tests."""
    param_names = []
    state_names = []

    def elastic_stiffness(self, params):
        raise NotImplementedError

    def yield_function(self, stress, state, params):
        raise NotImplementedError

    def hardening_increment(self, dlambda, stress, state, params):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_materialmodel3d_accepts_solid_3d():
    m = _Stub3D(SOLID_3D)
    assert m.stress_state is SOLID_3D
    assert m.ntens == 6


def test_materialmodel3d_accepts_plane_strain():
    m = _Stub3D(PLANE_STRAIN)
    assert m.stress_state is PLANE_STRAIN
    assert m.ntens == 4


def test_materialmodel3d_rejects_plane_stress():
    with pytest.raises(ValueError, match="ndi == ndi_phys"):
        _Stub3D(PLANE_STRESS)


def test_materialmodel3d_rejects_uniaxial_1d():
    with pytest.raises(ValueError, match="ndi == ndi_phys"):
        _Stub3D(UNIAXIAL_1D)


# ---------------------------------------------------------------------------
# _hydrostatic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ss", [SOLID_3D, PLANE_STRAIN])
def test_hydrostatic_isotropic(ss):
    """Hydrostatic stress of isotropic state p*delta equals p."""
    m = _Stub3D(ss)
    p = 150.0
    stress = jnp.array([p, p, p] + [0.0] * (ss.ntens - 3))
    assert jnp.allclose(m._hydrostatic(stress), p)


@pytest.mark.parametrize("ss", [SOLID_3D, PLANE_STRAIN])
def test_hydrostatic_uniaxial(ss):
    """Uniaxial stress σ11=σ gives p = σ/3."""
    m = _Stub3D(ss)
    sigma = 300.0
    stress = jnp.zeros(ss.ntens).at[0].set(sigma)
    assert jnp.allclose(m._hydrostatic(stress), sigma / 3.0)


def test_hydrostatic_shear_only():
    """Pure shear contributes nothing to hydrostatic pressure."""
    m = _Stub3D(SOLID_3D)
    stress = jnp.array([0.0, 0.0, 0.0, 100.0, 50.0, 25.0])
    assert jnp.allclose(m._hydrostatic(stress), 0.0)


# ---------------------------------------------------------------------------
# _dev
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ss", [SOLID_3D, PLANE_STRAIN])
def test_dev_trace_zero(ss):
    """Trace of deviatoric stress must be zero."""
    m = _Stub3D(ss)
    stress = jnp.array([100.0, -50.0, 30.0] + [20.0] * (ss.ntens - 3))
    s = m._dev(stress)
    # Trace = sum of direct components
    assert jnp.allclose(jnp.sum(s[:3]), 0.0, atol=1e-12)


@pytest.mark.parametrize("ss", [SOLID_3D, PLANE_STRAIN])
def test_dev_shape(ss):
    """_dev returns a vector of the same shape as the input."""
    m = _Stub3D(ss)
    stress = jnp.ones(ss.ntens)
    assert m._dev(stress).shape == (ss.ntens,)


def test_dev_isotropic_stress_is_zero():
    """Deviatoric of an isotropic stress p*delta is zero."""
    m = _Stub3D(SOLID_3D)
    p = 200.0
    stress = jnp.array([p, p, p, 0.0, 0.0, 0.0])
    assert jnp.allclose(m._dev(stress), 0.0)


# ---------------------------------------------------------------------------
# _vonmises
# ---------------------------------------------------------------------------

def test_vonmises_uniaxial_solid_3d(steel_params):
    """Uniaxial σ11: von Mises = σ11."""
    m = _Stub3D(SOLID_3D)
    sigma = 300.0
    stress = jnp.array([sigma, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert jnp.allclose(m._vonmises(stress), sigma, rtol=1e-6)


def test_vonmises_pure_shear_solid_3d():
    """Pure shear σ12: von Mises = √3 · σ12."""
    m = _Stub3D(SOLID_3D)
    tau = 100.0
    stress = jnp.array([0.0, 0.0, 0.0, tau, 0.0, 0.0])
    expected = jnp.sqrt(3.0) * tau
    assert jnp.allclose(m._vonmises(stress), expected, rtol=1e-6)


def test_vonmises_plane_strain_uniaxial():
    """Uniaxial loading in PLANE_STRAIN: von Mises = σ11."""
    m = _Stub3D(PLANE_STRAIN)
    sigma = 300.0
    stress = jnp.array([sigma, 0.0, 0.0, 0.0])
    assert jnp.allclose(m._vonmises(stress), sigma, rtol=1e-6)


@pytest.mark.parametrize("ss", [SOLID_3D, PLANE_STRAIN])
def test_vonmises_nonnegative(ss):
    """Von Mises stress is always non-negative."""
    m = _Stub3D(ss)
    stress = jnp.array([-200.0, 100.0, 50.0] + [-30.0] * (ss.ntens - 3))
    assert float(m._vonmises(stress)) >= 0.0


# ---------------------------------------------------------------------------
# isotropic_C
# ---------------------------------------------------------------------------

def test_isotropic_C_shape_solid_3d():
    m = _Stub3D(SOLID_3D)
    C = m.isotropic_C(lam=121153.8, mu=80769.2)
    assert C.shape == (6, 6)


def test_isotropic_C_shape_plane_strain():
    m = _Stub3D(PLANE_STRAIN)
    C = m.isotropic_C(lam=121153.8, mu=80769.2)
    assert C.shape == (4, 4)


@pytest.mark.parametrize("ss", [SOLID_3D, PLANE_STRAIN])
def test_isotropic_C_symmetry(ss):
    """Stiffness tensor must be symmetric."""
    m = _Stub3D(ss)
    C = m.isotropic_C(lam=121153.8, mu=80769.2)
    assert jnp.allclose(C, C.T, atol=1e-10)


def test_isotropic_C_solid_3d_diagonal(steel_params):
    """C[0,0] = λ + 2μ and C[0,1] = λ for SOLID_3D."""
    E, nu = steel_params["E"], steel_params["nu"]
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    m = _Stub3D(SOLID_3D)
    C = m.isotropic_C(lam, mu)
    assert jnp.allclose(C[0, 0], lam + 2 * mu, rtol=1e-8)
    assert jnp.allclose(C[0, 1], lam, rtol=1e-8)
    assert jnp.allclose(C[3, 3], mu, rtol=1e-8)


def test_isotropic_C_plane_strain_is_submatrix(steel_params):
    """PLANE_STRAIN C must equal the top-left 4×4 of SOLID_3D C."""
    E, nu = steel_params["E"], steel_params["nu"]
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    C3d = _Stub3D(SOLID_3D).isotropic_C(lam, mu)
    Cpe = _Stub3D(PLANE_STRAIN).isotropic_C(lam, mu)
    assert jnp.allclose(Cpe, C3d[:4, :4], atol=1e-10)


# ---------------------------------------------------------------------------
# _I_vol and _I_dev
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ss", [SOLID_3D, PLANE_STRAIN])
def test_I_vol_plus_I_dev_equals_identity(ss):
    """P_vol + P_dev must equal the ntens×ntens identity."""
    m = _Stub3D(ss)
    I = jnp.eye(ss.ntens)
    assert jnp.allclose(m._I_vol() + m._I_dev(), I, atol=1e-12)


@pytest.mark.parametrize("ss", [SOLID_3D, PLANE_STRAIN])
def test_I_vol_projects_to_hydrostatic(ss):
    """P_vol @ σ must equal p δ (volumetric part only)."""
    m = _Stub3D(ss)
    stress = jnp.array([100.0, -50.0, 30.0] + [20.0] * (ss.ntens - 3))
    p = m._hydrostatic(stress)
    vol_part = m._I_vol() @ stress
    expected = p * ss.identity_jnp
    assert jnp.allclose(vol_part, expected, atol=1e-10)


@pytest.mark.parametrize("ss", [SOLID_3D, PLANE_STRAIN])
def test_I_dev_projects_to_deviatoric(ss):
    """P_dev @ σ must equal _dev(σ)."""
    m = _Stub3D(ss)
    stress = jnp.array([100.0, -50.0, 30.0] + [20.0] * (ss.ntens - 3))
    assert jnp.allclose(m._I_dev() @ stress, m._dev(stress), atol=1e-10)
