"""Tests for manforge.utils.spectral — principal-value functions and their gradient."""

import autograd
import numpy as np
import pytest

import manforge  # noqa: F401 — enables float64
from manforge.core.dimension import PLANE_STRAIN, PLANE_STRESS, SOLID_3D, UNIAXIAL_1D
from manforge.utils.spectral import (
    negative_part_sq, positive_part_sq, principal_values, spectral_trace,
    voigt_stress_to_tensor,
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


def test_second_derivative_fails_loudly():
    """Only first derivatives are provided; a Hessian must not be silently wrong."""
    hess = autograd.hessian(lambda s: positive_part_sq(s, SOLID_3D))
    with pytest.raises(Exception):
        hess(np.array([300.0, 10.0, -5.0, 1.0, 2.0, 3.0]))
