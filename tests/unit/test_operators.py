"""Unit tests for autodiff.operators."""

import math

import numpy as np
import autograd.numpy as anp

from manforge.autodiff.operators import (
    I_dev_voigt,
    I_vol_voigt,
    dev,
    hydrostatic,
    identity_voigt,
    norm_mandel,
    vonmises,
)
from manforge.models.j2_isotropic import J2Isotropic3D, J2IsotropicPS, J2Isotropic1D


# ---------------------------------------------------------------------------
# identity_voigt
# ---------------------------------------------------------------------------

def test_identity_voigt():
    expected = anp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(np.asarray(identity_voigt()), np.asarray(expected))


# ---------------------------------------------------------------------------
# hydrostatic
# ---------------------------------------------------------------------------

def test_hydrostatic_known():
    s = anp.array([100.0, 200.0, 300.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(float(hydrostatic(s)), 200.0)


def test_hydrostatic_pure_shear():
    s = anp.array([0.0, 0.0, 0.0, 50.0, 30.0, 10.0])
    np.testing.assert_allclose(float(hydrostatic(s)), 0.0)


# ---------------------------------------------------------------------------
# dev
# ---------------------------------------------------------------------------

def test_dev_trace_zero():
    s = anp.array([100.0, 200.0, 300.0, 10.0, 20.0, 30.0])
    d = dev(s)
    np.testing.assert_allclose(float(d[0] + d[1] + d[2]), 0.0, atol=1e-12)


def test_dev_known_uniaxial():
    # Uniaxial σ11 = 300, all others zero
    s = anp.array([300.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    expected = anp.array([200.0, -100.0, -100.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(np.asarray(dev(s)), np.asarray(expected))


def test_dev_preserves_shear():
    s = anp.array([0.0, 0.0, 0.0, 50.0, 30.0, 10.0])
    d = dev(s)
    np.testing.assert_allclose(np.asarray(d[3:]), np.asarray(s[3:]))


# ---------------------------------------------------------------------------
# vonmises
# ---------------------------------------------------------------------------

def test_vonmises_uniaxial():
    # Under uniaxial tension σ11 = σ, σ_vm should equal σ
    sigma = 250.0
    s = anp.array([sigma, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(float(vonmises(s)), sigma, rtol=1e-6)


def test_vonmises_pure_shear():
    # Under pure shear τ_12 = τ, σ_vm = √3 · τ
    tau = 100.0
    s = anp.array([0.0, 0.0, 0.0, tau, 0.0, 0.0])
    np.testing.assert_allclose(float(vonmises(s)), float(anp.sqrt(3.0) * tau), rtol=1e-6)


def test_vonmises_zero():
    s = anp.zeros(6)
    np.testing.assert_allclose(float(vonmises(s)), 0.0)


# ---------------------------------------------------------------------------
# norm_mandel
# ---------------------------------------------------------------------------

def test_norm_mandel_known():
    # v = [1, 0, 0, 1, 0, 0]  → mandel = [1, 0, 0, √2, 0, 0]
    # norm = √(1 + 2) = √3
    v = anp.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(float(norm_mandel(v)), float(anp.sqrt(3.0)), rtol=1e-6)


def test_norm_mandel_normal_only():
    # Normal-only vector: Mandel = Voigt, so norm is just L2 norm
    v = anp.array([3.0, 4.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(float(norm_mandel(v)), 5.0, rtol=1e-6)


# ---------------------------------------------------------------------------
# I_dev_voigt / I_vol_voigt
# ---------------------------------------------------------------------------

def test_I_dev_plus_I_vol_equals_identity():
    I_dev = I_dev_voigt()
    I_vol = I_vol_voigt()
    np.testing.assert_allclose(np.asarray(I_dev + I_vol), np.eye(6), atol=1e-12)


def test_dev_via_projection():
    s = anp.array([100.0, 200.0, 300.0, 10.0, 20.0, 30.0])
    np.testing.assert_allclose(np.asarray(I_dev_voigt() @ s), np.asarray(dev(s)), atol=1e-10)


def test_I_vol_projects_to_hydrostatic():
    s = anp.array([100.0, 200.0, 300.0, 10.0, 20.0, 30.0])
    p = hydrostatic(s)
    expected = p * identity_voigt()
    np.testing.assert_allclose(np.asarray(I_vol_voigt() @ s), np.asarray(expected), atol=1e-10)


# ---------------------------------------------------------------------------
# inner_product (MaterialModel method)
# ---------------------------------------------------------------------------

def _model3d():
    return J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)

def _model_ps():
    return J2IsotropicPS(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)

def _model1d():
    return J2Isotropic1D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)


def test_inner_product_uniaxial_3d():
    m = _model3d()
    a = anp.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    b = anp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(float(m.inner_product(a, b)), 100.0, atol=1e-12)


def test_inner_product_pure_shear_doubled():
    # Physical shear σ12=1: Mandel factor √2 → inner_product = (√2)² = 2
    m = _model3d()
    v = anp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(float(m.inner_product(v, v)), 2.0, atol=1e-12)


def test_inner_product_agrees_with_to_mandel():
    from manforge.utils.voigt import to_mandel
    m = _model3d()
    a = anp.array([100.0, 50.0, -30.0, 10.0, 5.0, 2.0])
    b = anp.array([20.0, -10.0, 15.0, 3.0, 1.0, 4.0])
    expected = float(anp.dot(to_mandel(a), to_mandel(b)))
    np.testing.assert_allclose(float(m.inner_product(a, b)), expected, atol=1e-12)


def test_inner_product_ps():
    m = _model_ps()
    a = anp.array([100.0, 50.0, 10.0])
    b = anp.array([20.0, -10.0, 3.0])
    # Mandel factors for PS: [1, 1, √2]
    sqrt2 = math.sqrt(2.0)
    expected = 100.0*20.0 + 50.0*(-10.0) + (sqrt2*10.0)*(sqrt2*3.0)
    np.testing.assert_allclose(float(m.inner_product(a, b)), expected, atol=1e-12)


def test_inner_product_1d():
    m = _model1d()
    a = anp.array([100.0])
    b = anp.array([3.0])
    np.testing.assert_allclose(float(m.inner_product(a, b)), 300.0, atol=1e-12)


# ---------------------------------------------------------------------------
# deviatoric_inner_product
# ---------------------------------------------------------------------------

def test_deviatoric_inner_product_3d_same_as_inner():
    m = _model3d()
    stress = anp.array([300.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s = anp.array(m.dev(stress))
    np.testing.assert_allclose(
        float(m.deviatoric_inner_product(s, s)),
        float(m.inner_product(s, s)),
        atol=1e-12,
    )


def test_deviatoric_inner_product_ps_explicit():
    m = _model_ps()
    # Deviatoric stress in PS: s = [s11, s22, s12], s33 = -(s11+s22)
    s11, s22, s12 = 80.0, -30.0, 15.0
    s = anp.array([s11, s22, s12])
    s33 = -(s11 + s22)
    expected = s11*s11 + s22*s22 + s33*s33 + 2.0*s12*s12
    np.testing.assert_allclose(
        float(m.deviatoric_inner_product(s, s)), expected, atol=1e-12
    )


def test_deviatoric_inner_product_1d():
    m = _model1d()
    sigma = 300.0
    # dev([σ11]) = [2/3 σ11], s22 = s33 = -s11/2
    s = anp.array(m.dev(anp.array([sigma])))
    s11_dev = float(s[0])
    expected = s11_dev**2 + 2.0 * (s11_dev / 2.0)**2
    np.testing.assert_allclose(
        float(m.deviatoric_inner_product(s, s)), expected, atol=1e-12
    )


def test_deviatoric_inner_product_consistent_with_vonmises_norm():
    """√(3/2 dev_inner(s, s)) == vonmises_norm(s) for all stress states."""
    for m, stress in [
        (_model3d(), anp.array([300.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        (_model_ps(), anp.array([300.0, 0.0, 0.0])),
        (_model1d(), anp.array([300.0])),
    ]:
        s = anp.array(m.dev(stress))
        via_dev_inner = float(anp.sqrt(1.5 * m.deviatoric_inner_product(s, s)))
        via_vonmises_norm = float(m.vonmises_norm(s))
        np.testing.assert_allclose(via_dev_inner, via_vonmises_norm, rtol=1e-10)


# ---------------------------------------------------------------------------
# strain_norm
# ---------------------------------------------------------------------------

def test_strain_norm_uniaxial_3d():
    # Isochoric uniaxial plastic strain: ε=[ε11, -ε11/2, -ε11/2, 0, 0, 0] → ε11
    m = _model3d()
    e11 = 0.01
    eps = anp.array([e11, -e11/2, -e11/2, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(float(m.strain_norm(eps)), e11, rtol=1e-10)


def test_strain_norm_uniaxial_ps():
    m = _model_ps()
    e11 = 0.01
    eps = anp.array([e11, -e11/2, 0.0])
    np.testing.assert_allclose(float(m.strain_norm(eps)), e11, rtol=1e-10)


def test_strain_norm_uniaxial_1d():
    m = _model1d()
    e11 = 0.01
    eps = anp.array([e11])
    np.testing.assert_allclose(float(m.strain_norm(eps)), e11, rtol=1e-10)


def test_strain_norm_pure_shear_3d():
    # Physical shear ε12=γ_phys → ε_eq = γ_phys * 2/√3
    m = _model3d()
    g = 0.005
    eps = anp.array([0.0, 0.0, 0.0, g, 0.0, 0.0])
    expected = g * 2.0 / math.sqrt(3.0)
    np.testing.assert_allclose(float(m.strain_norm(eps)), expected, rtol=1e-10)


def test_strain_norm_conjugate_to_vonmises():
    """For purely deviatoric isochoric strain, strain_norm(ε) * vonmises(σ) equals
    the plastic dissipation rate when σ is uniaxial at yield."""
    m = _model3d()
    sigma_y = 250.0
    stress = anp.array([sigma_y, 0.0, 0.0, 0.0, 0.0, 0.0])
    e11 = 0.01
    eps_p = anp.array([e11, -e11/2, -e11/2, 0.0, 0.0, 0.0])
    # σ:ε = σ11 * ε11 = σ_y * e11 (uniaxial)
    # vonmises(σ) * strain_norm(ε) should equal same
    plastic_dissipation = sigma_y * e11
    vm_times_enorm = float(m.vonmises(stress)) * float(m.strain_norm(eps_p))
    np.testing.assert_allclose(vm_times_enorm, plastic_dissipation, rtol=1e-10)

