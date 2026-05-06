"""Tests for StressState and parameterized operators."""

import math

import autograd.numpy as anp
import numpy as np
import pytest

from manforge.core.dimension import (
    PLANE_STRAIN,
    PLANE_STRESS,
    SOLID_3D,
    UNIAXIAL_1D,
    StressState,
)
from manforge.autodiff.operators import (
    I_dev_voigt,
    I_vol_voigt,
    dev,
    hydrostatic,
    identity_voigt,
    norm_mandel,
    vonmises,
)
from manforge.utils.voigt import from_mandel, to_mandel

_sqrt2 = math.sqrt(2.0)

ALL_SS = [SOLID_3D, PLANE_STRAIN, PLANE_STRESS, UNIAXIAL_1D]


# ---------------------------------------------------------------------------
# StressState structural invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ss", ALL_SS)
def test_ntens_equals_ndi_plus_nshr(ss):
    assert ss.ntens == ss.ndi + ss.nshr


@pytest.mark.parametrize("ss", ALL_SS)
def test_mandel_factors_length(ss):
    assert len(ss.mandel_factors) == ss.ntens


@pytest.mark.parametrize("ss", ALL_SS)
def test_normal_mandel_factors_are_one(ss):
    for i in range(ss.ndi):
        assert ss.mandel_factors[i] == 1.0


@pytest.mark.parametrize("ss", ALL_SS)
def test_shear_mandel_factors_are_sqrt2(ss):
    for i in range(ss.ndi, ss.ntens):
        assert abs(ss.mandel_factors[i] - _sqrt2) < 1e-14


def test_invalid_stress_state_raises():
    with pytest.raises(ValueError, match="ntens"):
        StressState("bad", ntens=5, ndi=3, nshr=3, ndi_phys=3,
                    mandel_factors=(1.0,) * 5)

    with pytest.raises(ValueError, match="len.mandel_factors."):
        StressState("bad2", ntens=6, ndi=3, nshr=3, ndi_phys=3,
                    mandel_factors=(1.0,) * 4)


# ---------------------------------------------------------------------------
# identity_voigt
# ---------------------------------------------------------------------------


def test_identity_voigt_3d():
    expected = anp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(identity_voigt(), expected)
    np.testing.assert_allclose(identity_voigt(SOLID_3D), expected)


def test_identity_voigt_plane_strain():
    expected = anp.array([1.0, 1.0, 1.0, 0.0])
    np.testing.assert_allclose(identity_voigt(PLANE_STRAIN), expected)


def test_identity_voigt_plane_stress():
    expected = anp.array([1.0, 1.0, 0.0])
    np.testing.assert_allclose(identity_voigt(PLANE_STRESS), expected)


def test_identity_voigt_1d():
    expected = anp.array([1.0])
    np.testing.assert_allclose(identity_voigt(UNIAXIAL_1D), expected)


# ---------------------------------------------------------------------------
# to_mandel / from_mandel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ss", ALL_SS)
def test_mandel_roundtrip(ss):
    """from_mandel(to_mandel(v)) == v for any stress state."""
    v = anp.ones(ss.ntens)
    np.testing.assert_allclose(from_mandel(to_mandel(v, ss), ss), v, atol=1e-12)


def test_to_mandel_3d_backward_compat():
    """to_mandel with no ss arg matches 6-component convention."""
    v = anp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    expected = anp.array([1.0, 2.0, 3.0, 4.0 * _sqrt2, 5.0 * _sqrt2, 6.0 * _sqrt2])
    np.testing.assert_allclose(to_mandel(v), expected, atol=1e-12)


def test_to_mandel_plane_strain():
    v = anp.array([1.0, 2.0, 3.0, 4.0])
    expected = anp.array([1.0, 2.0, 3.0, 4.0 * _sqrt2])
    np.testing.assert_allclose(to_mandel(v, PLANE_STRAIN), expected, atol=1e-12)


# ---------------------------------------------------------------------------
# hydrostatic
# ---------------------------------------------------------------------------


def test_hydrostatic_3d():
    stress = anp.array([90.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(hydrostatic(stress), 30.0, atol=1e-10)
    np.testing.assert_allclose(hydrostatic(stress, SOLID_3D), 30.0, atol=1e-10)


def test_hydrostatic_plane_strain():
    # All 3 direct components present; same arithmetic as 3D
    stress = anp.array([90.0, 30.0, 0.0, 0.0])
    np.testing.assert_allclose(hydrostatic(stress, PLANE_STRAIN), 40.0, atol=1e-10)


def test_hydrostatic_plane_stress():
    # ndi=2 stored, ndi_phys=3 → divide by 3 (sigma_33=0 implicitly)
    stress = anp.array([90.0, 30.0, 0.0])
    np.testing.assert_allclose(hydrostatic(stress, PLANE_STRESS), 40.0, atol=1e-10)


def test_hydrostatic_1d():
    # ndi_phys=3: uniaxial stress sigma_22=sigma_33=0, so p = sigma_11 / 3
    stress = anp.array([90.0])
    np.testing.assert_allclose(hydrostatic(stress, UNIAXIAL_1D), 30.0, atol=1e-10)


def test_vonmises_uniaxial_1d():
    """vonmises([sigma], UNIAXIAL_1D) == |sigma| for uniaxial stress.

    With ndi_phys=3, two direct components are missing (sigma_22=sigma_33=0).
    Each contributes deviatoric -p to the full Mandel norm.
    """
    sigma = 100.0
    stress = anp.array([sigma])
    np.testing.assert_allclose(vonmises(stress, UNIAXIAL_1D), sigma, atol=1e-8)


def test_vonmises_uniaxial_plane_stress():
    """vonmises([sigma, 0, 0], PLANE_STRESS) == sigma for uniaxial loading.

    With ndi_phys=3 and ndi=2, the unstored sigma_33=0 contributes deviatoric
    -p.  Accounting for this extra component gives the correct 3D von Mises.
    """
    sigma = 100.0
    stress = anp.array([sigma, 0.0, 0.0])
    np.testing.assert_allclose(vonmises(stress, PLANE_STRESS), sigma, atol=1e-8)


# ---------------------------------------------------------------------------
# dev: deviatoric component sum must vanish
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ss", [SOLID_3D, PLANE_STRAIN])
def test_dev_trace_zero(ss):
    """Sum of direct components of dev(sigma) == 0 when all 3 normals present."""
    n = ss.ntens
    stress = anp.arange(1.0, n + 1)
    s = dev(stress, ss)
    np.testing.assert_allclose(float(anp.sum(s[: ss.ndi])), 0.0, atol=1e-10)


def test_dev_3d_backward_compat():
    stress = anp.array([100.0, 200.0, 300.0, 0.0, 0.0, 0.0])
    s = dev(stress)
    np.testing.assert_allclose(anp.sum(s[:3]), 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# vonmises: known values
# ---------------------------------------------------------------------------


def test_vonmises_uniaxial_3d():
    sigma = 100.0
    stress = anp.array([sigma, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(vonmises(stress), sigma, atol=1e-8)


def test_vonmises_uniaxial_plane_strain():
    # Plane strain under uniaxial load: sigma_33 = nu * sigma_11 but here we
    # test a pure [sigma, 0, 0, 0] vector for simplicity.
    sigma = 100.0
    stress = anp.array([sigma, 0.0, 0.0, 0.0])
    # dev = [2/3, -1/3, -1/3, 0] * sigma
    # vm  = sqrt(3/2 * (4/9 + 1/9 + 1/9)) * sigma = sigma
    np.testing.assert_allclose(vonmises(stress, PLANE_STRAIN), sigma, atol=1e-8)


def test_vonmises_pure_shear_3d():
    # sigma_vm = sqrt(3) * tau for pure shear
    # s = dev([0,0,0,tau,0,0]) = [0,0,0,tau,0,0]
    # s_mandel = [0,0,0,sqrt(2)*tau,0,0]  → ||s_mandel|| = sqrt(2)*tau
    # sigma_vm = sqrt(1.5) * sqrt(2) * tau = sqrt(3) * tau
    tau = 100.0
    stress = anp.array([0.0, 0.0, 0.0, tau, 0.0, 0.0])
    np.testing.assert_allclose(vonmises(stress), tau * math.sqrt(3), atol=1e-8)


# ---------------------------------------------------------------------------
# I_dev + I_vol == I (for every stress state)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ss", ALL_SS)
def test_projection_identity(ss):
    I_d = I_dev_voigt(ss)
    I_v = I_vol_voigt(ss)
    I_sum = I_d + I_v
    np.testing.assert_allclose(I_sum, anp.eye(ss.ntens), atol=1e-12)


def test_projection_identity_no_ss():
    """Backward-compat: no ss argument uses 6-component tensors."""
    np.testing.assert_allclose(I_dev_voigt() + I_vol_voigt(), anp.eye(6), atol=1e-12)


# ---------------------------------------------------------------------------
# isotropic_C shapes and basic values
# ---------------------------------------------------------------------------


def test_isotropic_C_shapes():
    from manforge.models.j2_isotropic import J2Isotropic3D, J2IsotropicPS, J2Isotropic1D
    from manforge.core.dimension import PLANE_STRAIN, PLANE_STRESS, UNIAXIAL_1D

    for model, expected_shape in [
        (J2Isotropic3D(SOLID_3D, E=200e3, nu=0.3, sigma_y0=250.0, H=1000.0), (6, 6)),
        (J2Isotropic3D(PLANE_STRAIN, E=200e3, nu=0.3, sigma_y0=250.0, H=1000.0), (4, 4)),
        (J2IsotropicPS(PLANE_STRESS, E=200e3, nu=0.3, sigma_y0=250.0, H=1000.0), (3, 3)),
        (J2Isotropic1D(UNIAXIAL_1D, E=200e3, nu=0.3, sigma_y0=250.0, H=1000.0), (1, 1)),
    ]:
        C = model.elastic_stiffness()
        assert C.shape == expected_shape, f"{model.stress_state.name}: expected {expected_shape}, got {C.shape}"


def test_isotropic_C_3d_diagonal():
    """C_3D[0,0] == E(1-nu)/((1+nu)(1-2nu))."""
    from manforge.models.j2_isotropic import J2Isotropic3D
    E, nu = 200e3, 0.3
    model = J2Isotropic3D(E=E, nu=nu, sigma_y0=250.0, H=0.0)
    C = model.elastic_stiffness()
    expected_C11 = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    np.testing.assert_allclose(C[0, 0], expected_C11, rtol=1e-10)


def test_isotropic_C_1d():
    """C_1D[0,0] == E (Young's modulus for uniaxial stress)."""
    from manforge.models.j2_isotropic import J2Isotropic1D
    E, nu = 200e3, 0.3
    model = J2Isotropic1D(UNIAXIAL_1D, E=E, nu=nu, sigma_y0=250.0, H=0.0)
    C = model.elastic_stiffness()
    np.testing.assert_allclose(C[0, 0], E, rtol=1e-10)


def test_plane_stress_C_symmetry():
    """Plane-stress stiffness must be symmetric."""
    from manforge.models.j2_isotropic import J2IsotropicPS
    model = J2IsotropicPS(PLANE_STRESS, E=200e3, nu=0.3, sigma_y0=250.0, H=0.0)
    C = model.elastic_stiffness()
    np.testing.assert_allclose(C, C.T, atol=1e-10)
