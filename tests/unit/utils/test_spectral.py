"""Tests for manforge.utils.spectral — principal-value functions and their gradient."""

import autograd
import numpy as np
import pytest

import manforge  # noqa: F401 — enables float64
from manforge.core import PLANE_STRAIN, PLANE_STRESS, SOLID_3D, UNIAXIAL_1D
from manforge.utils import (
    macaulay_sq, neg_macaulay_sq, negative_part, negative_part_sq, positive_part,
    positive_part_sq, principal_values, spectral_trace, voigt_stress_to_tensor,
)

DIMS = [SOLID_3D, PLANE_STRAIN, PLANE_STRESS, UNIAXIAL_1D]


def _fd_grad(fun, x, eps=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = eps
        g[i] = (fun(x + e) - fun(x - e)) / (2 * eps)
    return g


@pytest.fixture
def rng():
    return np.random.default_rng(3)


def test_tensor_layout_3d():
    T = voigt_stress_to_tensor(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), SOLID_3D)
    np.testing.assert_array_equal(T, [[1, 4, 5], [4, 2, 6], [5, 6, 3]])


def test_tensor_layout_plane_stress_and_1d():
    T = voigt_stress_to_tensor(np.array([1.0, 2.0, 3.0]), PLANE_STRESS)
    np.testing.assert_array_equal(T, [[1, 3, 0], [3, 2, 0], [0, 0, 0]])
    T = voigt_stress_to_tensor(np.array([7.0]), UNIAXIAL_1D)
    np.testing.assert_array_equal(T, np.diag([7.0, 0.0, 0.0]))


@pytest.mark.parametrize("dim", DIMS)
def test_sum_of_squares_matches_double_contraction(dim, rng):
    """Σλ² = σ:σ, and its gradient is the Voigt gradient of σ:σ (shear ×2)."""
    sig = 100.0 * rng.standard_normal(dim.ntens)
    F = spectral_trace(sig, dim, lambda l: l ** 2, lambda l: 2 * l)
    m2 = np.asarray(dim.mandel_factors_np) ** 2
    assert F == pytest.approx(float(np.sum(m2 * sig ** 2)), rel=1e-12)
    g = autograd.grad(lambda s: spectral_trace(s, dim, lambda l: l ** 2, lambda l: 2 * l))(sig)
    np.testing.assert_allclose(g, 2 * m2 * sig, rtol=1e-10)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("fun", [positive_part_sq, negative_part_sq])
def test_split_gradient_matches_fd(dim, fun, rng):
    sig = 100.0 * rng.standard_normal(dim.ntens)
    g = autograd.grad(lambda s: fun(s, dim))(sig)
    np.testing.assert_allclose(g, _fd_grad(lambda s: fun(s, dim), sig, eps=1e-3), rtol=1e-7, atol=1e-7)


def test_split_parts_add_up(rng):
    sig = 100.0 * rng.standard_normal(6)
    lam = principal_values(sig, SOLID_3D)
    assert positive_part_sq(sig, SOLID_3D) + negative_part_sq(sig, SOLID_3D) == \
        pytest.approx(float(np.sum(lam ** 2)))


@pytest.mark.parametrize("sig", [
    np.array([300.0, 0, 0, 0, 0, 0]),        # uniaxial: λ2 = λ3 = 0
    np.array([-200.0, -200.0, -200.0, 0, 0, 0]),  # hydrostatic: triple root
    np.array([0.0, 0, 0, 150.0, 0, 0]),      # pure shear: λ = ±τ, 0
    np.zeros(6),                             # all zero
])
def test_gradient_finite_and_exact_at_repeated_principal_values(sig):
    """The eigh derivative is NaN here; the matrix-function gradient is not."""
    g = autograd.grad(lambda s: positive_part_sq(s, SOLID_3D))(sig)
    assert np.all(np.isfinite(g))
    np.testing.assert_allclose(
        g, _fd_grad(lambda s: positive_part_sq(s, SOLID_3D), sig), atol=1e-6
    )


def test_trace_without_second_derivative_fails_loudly():
    """No f'' supplied: a Hessian must raise, not be silently wrong."""
    hess = autograd.hessian(
        lambda s: spectral_trace(s, SOLID_3D, lambda l: l ** 4, lambda l: 4 * l ** 3))
    with pytest.raises(Exception):
        hess(np.array([300.0, 10.0, -5.0, 1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# σ₊ / σ₋ and P₊ = ∂σ₊/∂σ (Daleckii–Krein, H(0) = c)
# ---------------------------------------------------------------------------

def _mandel(dim):
    return np.asarray(dim.mandel_factors_np, dtype=float)


@pytest.mark.parametrize("dim", DIMS)
def test_positive_and_negative_parts_add_up(dim, rng):
    sig = 100.0 * rng.standard_normal(dim.ntens)
    np.testing.assert_allclose(positive_part(sig, dim) + negative_part(sig, dim), sig,
                               atol=1e-10)


@pytest.mark.parametrize("dim", DIMS)
def test_p_plus_matches_fd_away_from_kinks(dim, rng):
    sig = 100.0 * rng.standard_normal(dim.ntens)
    J = autograd.jacobian(lambda s: positive_part(s, dim))(sig)
    eps = 1e-4
    J_fd = np.column_stack([
        (np.asarray(positive_part(sig + eps * e, dim)) - np.asarray(positive_part(sig - eps * e, dim)))
        / (2 * eps) for e in np.eye(dim.ntens)])
    np.testing.assert_allclose(J, J_fd, atol=1e-7)


@pytest.mark.parametrize("dim", DIMS)
def test_p_plus_is_major_symmetric_and_complements_p_minus(dim, rng):
    sig = 100.0 * rng.standard_normal(dim.ntens)
    Jp = np.asarray(autograd.jacobian(lambda s: positive_part(s, dim))(sig))
    Jm = np.asarray(autograd.jacobian(lambda s: negative_part(s, dim))(sig))
    m = _mandel(dim)
    Pm = np.diag(m) @ Jp @ np.diag(1 / m)       # Mandel form
    np.testing.assert_allclose(Pm, Pm.T, atol=1e-12)
    np.testing.assert_allclose(Jp + Jm, np.eye(dim.ntens), atol=1e-12)


@pytest.mark.parametrize("dim", DIMS)
def test_hessian_of_positive_part_sq_is_twice_p_plus(dim, rng):
    sig = 100.0 * rng.standard_normal(dim.ntens)
    Hs = np.asarray(autograd.hessian(lambda s: positive_part_sq(s, dim))(sig))
    Jp = np.asarray(autograd.jacobian(lambda s: positive_part(s, dim))(sig))
    shear2 = np.array([1.0] * dim.ndi + [2.0] * dim.nshr)
    np.testing.assert_allclose(Hs, 2.0 * np.diag(shear2) @ Jp, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("c", [0.0, 0.5, 1.0])
def test_uniaxial_repeated_zero_uses_convention(c):
    """σ = (σ, 0, 0): the doubly repeated zero block takes Φ = c; mixed pairs Φ = 1."""
    sig = np.array([300.0, 0, 0, 0, 0, 0])
    J = np.asarray(autograd.jacobian(lambda s: positive_part(s, SOLID_3D, c=c))(sig))
    assert np.all(np.isfinite(J))
    assert J[0, 0] == pytest.approx(1.0)          # λ1 > 0
    assert J[1, 1] == pytest.approx(c)            # dσ22 inside the zero eigenspace
    assert J[2, 2] == pytest.approx(c)
    assert J[5, 5] == pytest.approx(c)            # σ23: both indices in the zero block
    assert J[3, 3] == pytest.approx(1.0)          # σ12: Φ(300, 0) = 1
    assert J[4, 4] == pytest.approx(1.0)


@pytest.mark.parametrize("c", [0.0, 0.5, 1.0])
def test_zero_state_gives_c_times_identity(c):
    J = np.asarray(autograd.jacobian(lambda s: positive_part(s, SOLID_3D, c=c))(np.zeros(6)))
    np.testing.assert_allclose(J, c * np.eye(6), atol=1e-15)


def test_rotated_zero_eigenspace_is_not_decided_by_eigh_noise():
    """A rotated uniaxial state: eigh returns ±1e-14 for the zeros.

    A perturbation inside the zero eigenspace must still map with factor c.
    """
    th = 0.3
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    T = R @ np.diag([300.0, 0.0, 0.0]) @ R.T
    to_v = lambda A: np.array([A[0, 0], A[1, 1], A[2, 2], A[0, 1], A[0, 2], A[1, 2]])  # noqa: E731
    n2 = R[:, 1]
    d_T = np.outer(n2, n2)
    for c in (0.0, 0.5, 1.0):
        J = np.asarray(autograd.jacobian(lambda s: positive_part(s, SOLID_3D, c=c))(to_v(T)))
        np.testing.assert_allclose(J @ to_v(d_T), c * to_v(d_T), atol=1e-12)


@pytest.mark.parametrize("c", [0.0, 0.5, 1.0])
def test_scalar_brackets_use_convention(c):
    d2p = autograd.grad(autograd.grad(lambda x: macaulay_sq(x, c)))
    d2n = autograd.grad(autograd.grad(lambda x: neg_macaulay_sq(x, c)))
    assert float(d2p(0.0)) == pytest.approx(2 * c)
    assert float(d2n(0.0)) == pytest.approx(2 * (1 - c))
    assert float(d2p(3.0)) == pytest.approx(2.0) and float(d2p(-3.0)) == pytest.approx(0.0)
    assert float(macaulay_sq(-2.0)) == 0.0 and float(neg_macaulay_sq(-2.0)) == 4.0
    assert float(autograd.grad(lambda x: neg_macaulay_sq(x, c))(-2.0)) == pytest.approx(-4.0)
