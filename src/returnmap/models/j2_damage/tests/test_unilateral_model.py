"""Pure-logic tests for J2IsotropicDamageUnilateral: potential, Y, σ̃ → σ inversion."""

import autograd
import numpy as np
import pytest

import returnmap  # noqa: F401 — enables float64
from returnmap.models import (
    J2IsotropicDamageUnilateral3D, J2IsotropicDamageUnilateralPS,
    J2IsotropicDamageUnilateral1D,
)

P = dict(E=200000.0, nu=0.3, sigma_y0=250.0, H=2000.0, S=0.5, s=1.0, p_D=0.01)
CLASSES = [J2IsotropicDamageUnilateral3D, J2IsotropicDamageUnilateralPS,
           J2IsotropicDamageUnilateral1D]


@pytest.fixture
def rng():
    return np.random.default_rng(11)


def _compliance_energy(m, sig):
    C0 = np.asarray(m.undamaged_stiffness())
    return 0.5 * sig @ np.linalg.solve(C0, sig)


@pytest.mark.parametrize("cls", CLASSES)
def test_h1_has_no_split(cls, rng):
    """h = 1: ψ* = σ:C0⁻¹:σ / (2(1−D)), so σ = (1−D) σ̃ and Y = σ̃:C0⁻¹:σ̃/2."""
    m = cls(**P, h=1.0)
    sig, D = 100.0 * rng.standard_normal(m.ntens), 0.3
    assert float(m.complementary_energy(sig, D)) == pytest.approx(
        _compliance_energy(m, sig) / (1 - D), rel=1e-12)
    s_eff = sig / (1 - D)
    assert float(m.energy_release_rate(sig, D)) == pytest.approx(
        _compliance_energy(m, s_eff), rel=1e-12)
    np.testing.assert_allclose(np.asarray(m.nominal_from_effective(s_eff, D)), sig,
                               rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("h", [0.0, 0.2, 1.0])
def test_uniaxial_strain_tension_and_compression(h):
    m = J2IsotropicDamageUnilateral3D(**P, h=h)
    D, s = 0.3, 100.0
    for sign, k in [(1.0, 1 - D), (-1.0, 1 - h * D)]:
        eps = np.asarray(m.elastic_strain(np.array([sign * s, 0, 0, 0, 0, 0]), D))
        expected = sign * s / (k * P["E"]) * np.array([1, -P["nu"], -P["nu"], 0, 0, 0])
        np.testing.assert_allclose(eps, expected, rtol=1e-12, atol=1e-18)


def test_energy_release_rate_nonnegative(rng):
    m = J2IsotropicDamageUnilateral3D(**P, h=0.2)
    for _ in range(200):
        assert float(m.energy_release_rate(300.0 * rng.standard_normal(6), 0.4)) >= 0.0


@pytest.mark.parametrize("cls", CLASSES)
def test_inversion_round_trip(cls, rng):
    m = cls(**P, h=0.2)
    C0 = np.asarray(m.undamaged_stiffness())
    for _ in range(20):
        s_eff, D = 300.0 * rng.standard_normal(m.ntens), 0.4
        sig = np.asarray(m.nominal_from_effective(s_eff, D))
        np.testing.assert_allclose(C0 @ np.asarray(m.elastic_strain(sig, D)), s_eff,
                                   rtol=1e-10, atol=1e-9)


@pytest.mark.parametrize("cls", CLASSES)
def test_inversion_derivative_matches_fd_and_is_symmetric(cls, rng):
    m = cls(**P, h=0.2)
    s_eff, D = 300.0 * rng.standard_normal(m.ntens), 0.4
    J = np.asarray(autograd.jacobian(lambda s: m.nominal_from_effective(s, D))(s_eff))
    eps = 1e-4
    J_fd = np.column_stack([
        (np.asarray(m.nominal_from_effective(s_eff + eps * e, D))
         - np.asarray(m.nominal_from_effective(s_eff - eps * e, D))) / (2 * eps)
        for e in np.eye(m.ntens)])
    np.testing.assert_allclose(J, J_fd, atol=1e-7)
    # dσ/dε_e = J C0 is the elastic tangent; Mandel-symmetric ⟺ a potential exists.
    mf = np.asarray(m.dimension.mandel_factors_np)
    K = J @ np.asarray(m.undamaged_stiffness())
    Km = np.diag(mf) @ K @ np.diag(mf)
    np.testing.assert_allclose(Km, Km.T, atol=1e-9 * np.max(np.abs(Km)))


def test_inversion_derivative_wrt_damage_raises():
    m = J2IsotropicDamageUnilateral3D(**P, h=0.2)
    with pytest.raises(Exception):
        autograd.grad(lambda D: m.nominal_from_effective(np.ones(6) * 10.0, D)[0])(0.3)
