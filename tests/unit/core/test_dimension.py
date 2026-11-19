"""Tests for StressDimension: structural invariants and its operators."""

import math

import autograd.numpy as anp
import numpy as np
import pytest

from manforge.core.dimension import (
    PLANE_STRAIN,
    PLANE_STRESS,
    SOLID_3D,
    UNIAXIAL_1D,
    StressDimension,
)

_sqrt2 = math.sqrt(2.0)

ALL_SS = [SOLID_3D, PLANE_STRAIN, PLANE_STRESS, UNIAXIAL_1D]


# ---------------------------------------------------------------------------
# StressDimension structural invariants
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
        StressDimension("bad", ntens=5, ndi=3, nshr=3, ndi_phys=3,
                    mandel_factors=(1.0,) * 5)

    with pytest.raises(ValueError, match="len.mandel_factors."):
        StressDimension("bad2", ntens=6, ndi=3, nshr=3, ndi_phys=3,
                    mandel_factors=(1.0,) * 4)


# ---------------------------------------------------------------------------
# identity_np
# ---------------------------------------------------------------------------


def test_identity_3d():
    expected = anp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(SOLID_3D.identity_np, expected)


def test_identity_plane_strain():
    expected = anp.array([1.0, 1.0, 1.0, 0.0])
    np.testing.assert_allclose(PLANE_STRAIN.identity_np, expected)


def test_identity_plane_stress():
    expected = anp.array([1.0, 1.0, 0.0])
    np.testing.assert_allclose(PLANE_STRESS.identity_np, expected)


def test_identity_1d():
    expected = anp.array([1.0])
    np.testing.assert_allclose(UNIAXIAL_1D.identity_np, expected)


# ---------------------------------------------------------------------------
# mandel_factors_np
# ---------------------------------------------------------------------------


def test_mandel_factors_np_3d():
    expected = np.array([1.0, 1.0, 1.0, _sqrt2, _sqrt2, _sqrt2])
    np.testing.assert_allclose(SOLID_3D.mandel_factors_np, expected, atol=1e-14)


def test_mandel_factors_np_plane_strain():
    expected = np.array([1.0, 1.0, 1.0, _sqrt2])
    np.testing.assert_allclose(PLANE_STRAIN.mandel_factors_np, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# hydrostatic
# ---------------------------------------------------------------------------


def test_hydrostatic_3d():
    stress = anp.array([90.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(SOLID_3D.hydrostatic(stress), 30.0, atol=1e-10)


def test_hydrostatic_3d_general():
    stress = anp.array([100.0, 200.0, 300.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(float(SOLID_3D.hydrostatic(stress)), 200.0)


def test_hydrostatic_3d_pure_shear():
    stress = anp.array([0.0, 0.0, 0.0, 50.0, 30.0, 10.0])
    np.testing.assert_allclose(float(SOLID_3D.hydrostatic(stress)), 0.0)


def test_hydrostatic_plane_strain():
    # All 3 direct components present; same arithmetic as 3D
    stress = anp.array([90.0, 30.0, 0.0, 0.0])
    np.testing.assert_allclose(PLANE_STRAIN.hydrostatic(stress), 40.0, atol=1e-10)


def test_hydrostatic_plane_stress():
    # ndi=2 stored, ndi_phys=3 → divide by 3 (sigma_33=0 implicitly)
    stress = anp.array([90.0, 30.0, 0.0])
    np.testing.assert_allclose(PLANE_STRESS.hydrostatic(stress), 40.0, atol=1e-10)


def test_hydrostatic_1d():
    # ndi_phys=3: uniaxial stress sigma_22=sigma_33=0, so p = sigma_11 / 3
    stress = anp.array([90.0])
    np.testing.assert_allclose(UNIAXIAL_1D.hydrostatic(stress), 30.0, atol=1e-10)


def test_vonmises_uniaxial_1d():
    """UNIAXIAL_1D.vonmises([sigma]) == |sigma| for uniaxial stress.

    With ndi_phys=3, two direct components are missing (sigma_22=sigma_33=0).
    Each contributes deviatoric -p to the full Mandel norm.
    """
    sigma = 100.0
    stress = anp.array([sigma])
    np.testing.assert_allclose(UNIAXIAL_1D.vonmises(stress), sigma, atol=1e-8)


def test_vonmises_uniaxial_plane_stress():
    """PLANE_STRESS.vonmises([sigma, 0, 0]) == sigma for uniaxial loading.

    With ndi_phys=3 and ndi=2, the unstored sigma_33=0 contributes deviatoric
    -p.  Accounting for this extra component gives the correct 3D von Mises.
    """
    sigma = 100.0
    stress = anp.array([sigma, 0.0, 0.0])
    np.testing.assert_allclose(PLANE_STRESS.vonmises(stress), sigma, atol=1e-8)


# ---------------------------------------------------------------------------
# dev: deviatoric component sum must vanish
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ss", [SOLID_3D, PLANE_STRAIN])
def test_dev_trace_zero(ss):
    """Sum of direct components of dev(sigma) == 0 when all 3 normals present."""
    n = ss.ntens
    stress = anp.arange(1.0, n + 1)
    s = ss.dev(stress)
    np.testing.assert_allclose(float(anp.sum(s[: ss.ndi])), 0.0, atol=1e-10)


def test_dev_3d_known_uniaxial():
    stress = anp.array([300.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    expected = anp.array([200.0, -100.0, -100.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(np.asarray(SOLID_3D.dev(stress)), np.asarray(expected))


def test_dev_3d_preserves_shear():
    stress = anp.array([0.0, 0.0, 0.0, 50.0, 30.0, 10.0])
    s = SOLID_3D.dev(stress)
    np.testing.assert_allclose(np.asarray(s[3:]), np.asarray(stress[3:]))


def test_dev_3d_equals_projection():
    stress = anp.array([100.0, 200.0, 300.0, 10.0, 20.0, 30.0])
    np.testing.assert_allclose(
        np.asarray(SOLID_3D.I_dev() @ stress), np.asarray(SOLID_3D.dev(stress)),
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# vonmises: known values
# ---------------------------------------------------------------------------


def test_vonmises_uniaxial_3d():
    sigma = 100.0
    stress = anp.array([sigma, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(SOLID_3D.vonmises(stress), sigma, atol=1e-8)


def test_vonmises_uniaxial_plane_strain():
    # Plane strain under uniaxial load: sigma_33 = nu * sigma_11 but here we
    # test a pure [sigma, 0, 0, 0] vector for simplicity.
    sigma = 100.0
    stress = anp.array([sigma, 0.0, 0.0, 0.0])
    # dev = [2/3, -1/3, -1/3, 0] * sigma
    # vm  = sqrt(3/2 * (4/9 + 1/9 + 1/9)) * sigma = sigma
    np.testing.assert_allclose(PLANE_STRAIN.vonmises(stress), sigma, atol=1e-8)


def test_vonmises_pure_shear_3d():
    # sigma_vm = sqrt(3) * tau for pure shear
    # s = dev([0,0,0,tau,0,0]) = [0,0,0,tau,0,0]
    # s_mandel = [0,0,0,sqrt(2)*tau,0,0]  → ||s_mandel|| = sqrt(2)*tau
    # sigma_vm = sqrt(1.5) * sqrt(2) * tau = sqrt(3) * tau
    tau = 100.0
    stress = anp.array([0.0, 0.0, 0.0, tau, 0.0, 0.0])
    np.testing.assert_allclose(SOLID_3D.vonmises(stress), tau * math.sqrt(3), atol=1e-8)


def test_vonmises_zero_3d():
    # Not exactly 0: vonmises_norm is a smooth_sqrt, floored at 1e-12 so the
    # gradient stays finite at the origin.
    np.testing.assert_allclose(float(SOLID_3D.vonmises(anp.zeros(6))), 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# inner_product: Mandel norm of a physical-shear vector
# ---------------------------------------------------------------------------


def test_mandel_norm_known_3d():
    # v = [1, 0, 0, 1, 0, 0]  → mandel = [1, 0, 0, √2, 0, 0]  → norm = √3
    v = anp.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        float(anp.sqrt(SOLID_3D.inner_product(v, v))), math.sqrt(3.0), rtol=1e-12)


def test_mandel_norm_normal_only_3d():
    # Normal-only vector: Mandel = Voigt, so the norm is the L2 norm.
    v = anp.array([3.0, 4.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        float(anp.sqrt(SOLID_3D.inner_product(v, v))), 5.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# I_dev + I_vol == I (for every stress state)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ss", ALL_SS)
def test_projection_identity(ss):
    I_d = ss.I_dev()
    I_v = ss.I_vol()
    I_sum = I_d + I_v
    np.testing.assert_allclose(I_sum, anp.eye(ss.ntens), atol=1e-12)


# ---------------------------------------------------------------------------
# isotropic_C shapes and basic values
# ---------------------------------------------------------------------------


def test_isotropic_C_shapes():
    from manforge.models.j2_isotropic import J2Isotropic, J2Isotropic3D, J2IsotropicPS, J2Isotropic1D
    from manforge.core.dimension import PLANE_STRAIN, PLANE_STRESS, UNIAXIAL_1D

    for model, expected_shape in [
        (J2Isotropic3D(E=200e3, nu=0.3, sigma_y0=250.0, H=1000.0), (6, 6)),
        (J2Isotropic(dimension=PLANE_STRAIN, E=200e3, nu=0.3, sigma_y0=250.0, H=1000.0), (4, 4)),
        (J2IsotropicPS(E=200e3, nu=0.3, sigma_y0=250.0, H=1000.0), (3, 3)),
        (J2Isotropic1D(E=200e3, nu=0.3, sigma_y0=250.0, H=1000.0), (1, 1)),
    ]:
        C = model.elastic_stiffness(model.initial_state())
        assert C.shape == expected_shape, f"{model.dimension.name}: expected {expected_shape}, got {C.shape}"


def test_isotropic_C_3d_diagonal():
    """C_3D[0,0] == E(1-nu)/((1+nu)(1-2nu))."""
    from manforge.models.j2_isotropic import J2Isotropic3D
    E, nu = 200e3, 0.3
    model = J2Isotropic3D(E=E, nu=nu, sigma_y0=250.0, H=0.0)
    C = model.elastic_stiffness(model.initial_state())
    expected_C11 = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    np.testing.assert_allclose(C[0, 0], expected_C11, rtol=1e-10)


def test_isotropic_C_1d():
    """C_1D[0,0] == E (Young's modulus for uniaxial stress)."""
    from manforge.models.j2_isotropic import J2Isotropic1D
    E, nu = 200e3, 0.3
    model = J2Isotropic1D(E=E, nu=nu, sigma_y0=250.0, H=0.0)
    C = model.elastic_stiffness(model.initial_state())
    np.testing.assert_allclose(C[0, 0], E, rtol=1e-10)


def test_plane_stress_C_symmetry():
    """Plane-stress stiffness must be symmetric."""
    from manforge.models.j2_isotropic import J2IsotropicPS
    model = J2IsotropicPS(E=200e3, nu=0.3, sigma_y0=250.0, H=0.0)
    C = model.elastic_stiffness(model.initial_state())
    np.testing.assert_allclose(C, C.T, atol=1e-10)
