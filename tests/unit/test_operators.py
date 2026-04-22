"""Unit tests for autodiff.operators."""

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


