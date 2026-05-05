"""Tests for smooth mathematical utility functions.

Verifies that each function:
1. Returns finite, non-NaN values at the singular point (zero).
2. Has a finite gradient at the singular point.
3. Matches the non-smooth version away from the singular point.
4. Is consistent with the (x=0) limit.
"""

import pytest
import numpy as np
import autograd
import autograd.numpy as anp

import manforge  # enables float64
from manforge.utils.smooth import (
    smooth_sqrt,
    smooth_abs,
    smooth_norm,
    smooth_macaulay,
    smooth_direction,
    _DEFAULT_EPS,
)

EPS = _DEFAULT_EPS


# ---------------------------------------------------------------------------
# smooth_sqrt
# ---------------------------------------------------------------------------

class TestSmoothSqrt:
    def test_positive_value_matches_sqrt(self):
        """Away from zero: smooth_sqrt ≈ sqrt."""
        x = anp.array(4.0)
        np.testing.assert_allclose(float(smooth_sqrt(x)), 2.0, rtol=1e-10)

    def test_at_zero_returns_eps(self):
        """At x=0: value = sqrt(eps²) = eps."""
        val = float(smooth_sqrt(anp.array(0.0)))
        np.testing.assert_allclose(val, EPS, rtol=1e-6)

    def test_gradient_finite_at_zero(self):
        """At x=0: gradient is finite (not inf)."""
        grad = float(autograd.grad(smooth_sqrt)(anp.array(0.0)))
        assert np.isfinite(grad), f"gradient = {grad}"

    def test_gradient_matches_sqrt_away_from_zero(self):
        """Away from zero: d/dx smooth_sqrt ≈ d/dx sqrt."""
        x = anp.array(9.0)
        g_smooth = float(autograd.grad(smooth_sqrt)(x))
        g_exact = float(0.5 / anp.sqrt(x))
        np.testing.assert_allclose(g_smooth, g_exact, rtol=1e-8)

    def test_array_input(self):
        """Works on arrays element-wise."""
        x = anp.array([0.0, 1.0, 4.0, 9.0])
        result = smooth_sqrt(x)
        expected = anp.array([EPS, anp.sqrt(1.0 + EPS**2),
                               anp.sqrt(4.0 + EPS**2), anp.sqrt(9.0 + EPS**2)])
        np.testing.assert_allclose(np.array(result), np.array(expected), rtol=1e-10)


# ---------------------------------------------------------------------------
# smooth_abs
# ---------------------------------------------------------------------------

class TestSmoothAbs:
    def test_positive(self):
        """smooth_abs(x) ≈ x for x >> eps."""
        x = anp.array(1.0)
        np.testing.assert_allclose(float(smooth_abs(x)), 1.0, rtol=1e-10)

    def test_negative(self):
        """smooth_abs(-x) ≈ x for |x| >> eps."""
        x = anp.array(-3.0)
        np.testing.assert_allclose(float(smooth_abs(x)), 3.0, rtol=1e-10)

    def test_at_zero_value(self):
        """At x=0: value = sqrt(0 + eps²) = eps."""
        val = float(smooth_abs(anp.array(0.0)))
        np.testing.assert_allclose(val, EPS, rtol=1e-6)

    def test_gradient_at_zero_is_zero(self):
        """At x=0: gradient = 0 (smooth minimum, not kink)."""
        grad = float(autograd.grad(smooth_abs)(anp.array(0.0)))
        np.testing.assert_allclose(grad, 0.0, atol=1e-10)

    def test_gradient_positive_side(self):
        """For x >> eps: d/dx smooth_abs ≈ +1."""
        x = anp.array(1.0)
        grad = float(autograd.grad(smooth_abs)(x))
        np.testing.assert_allclose(grad, 1.0, rtol=1e-8)

    def test_gradient_negative_side(self):
        """For x << -eps: d/dx smooth_abs ≈ -1."""
        x = anp.array(-1.0)
        grad = float(autograd.grad(smooth_abs)(x))
        np.testing.assert_allclose(grad, -1.0, rtol=1e-8)

    def test_no_nan_anywhere(self):
        """No NaN for any finite input."""
        xs = anp.linspace(-10.0, 10.0, 1000)
        vals = smooth_abs(xs)
        assert anp.all(anp.isfinite(vals))

    def test_symmetric(self):
        """smooth_abs(-x) == smooth_abs(x)."""
        xs = anp.linspace(-5.0, 5.0, 100)
        np.testing.assert_allclose(
            np.array(smooth_abs(xs)),
            np.array(smooth_abs(-xs)),
            rtol=1e-12,
        )


# ---------------------------------------------------------------------------
# smooth_norm
# ---------------------------------------------------------------------------

class TestSmoothNorm:
    def test_unit_vector(self):
        """smooth_norm of a unit vector ≈ 1."""
        v = anp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(float(smooth_norm(v)), 1.0, rtol=1e-10)

    def test_known_norm(self):
        """smooth_norm([3, 4]) ≈ 5."""
        v = anp.array([3.0, 4.0])
        np.testing.assert_allclose(float(smooth_norm(v)), 5.0, rtol=1e-10)

    def test_at_zero_returns_eps(self):
        """At v=0: value = eps."""
        v = anp.zeros(6)
        val = float(smooth_norm(v))
        np.testing.assert_allclose(val, EPS, rtol=1e-6)

    def test_gradient_finite_at_zero(self):
        """At v=0: gradient is finite."""
        v = anp.zeros(3)
        grad = autograd.grad(smooth_norm)(v)
        assert anp.all(anp.isfinite(grad)), f"gradient has inf/NaN: {grad}"

    def test_gradient_zero_vector_is_zero(self):
        """At v=0: gradient = 0 (smooth minimum)."""
        v = anp.zeros(3)
        grad = autograd.grad(smooth_norm)(v)
        np.testing.assert_allclose(np.array(grad), np.zeros(3), atol=1e-10)

    def test_gradient_matches_norm_away_from_zero(self):
        """Away from zero: gradient ≈ v / ||v||."""
        v = anp.array([3.0, 4.0])
        grad = autograd.grad(smooth_norm)(v)
        expected = v / 5.0  # exact gradient of ||v|| at (3, 4)
        np.testing.assert_allclose(np.array(grad), np.array(expected), rtol=1e-8)


# ---------------------------------------------------------------------------
# smooth_macaulay
# ---------------------------------------------------------------------------

class TestSmoothMacaulay:
    def test_positive_value(self):
        """For x >> eps: smooth_macaulay(x) ≈ x."""
        x = anp.array(5.0)
        np.testing.assert_allclose(float(smooth_macaulay(x)), 5.0, rtol=1e-10)

    def test_negative_value(self):
        """For x << -eps: smooth_macaulay(x) ≈ 0."""
        x = anp.array(-5.0)
        val = float(smooth_macaulay(x))
        assert val < 1e-10, f"Expected ≈ 0, got {val}"

    def test_at_zero(self):
        """At x=0: value = eps/2 (small positive)."""
        val = float(smooth_macaulay(anp.array(0.0)))
        np.testing.assert_allclose(val, EPS / 2.0, rtol=1e-6)

    def test_gradient_at_zero(self):
        """At x=0: gradient = 0.5."""
        grad = float(autograd.grad(smooth_macaulay)(anp.array(0.0)))
        np.testing.assert_allclose(grad, 0.5, atol=1e-10)

    def test_gradient_positive_side(self):
        """For x >> eps: gradient ≈ 1."""
        x = anp.array(5.0)
        grad = float(autograd.grad(smooth_macaulay)(x))
        np.testing.assert_allclose(grad, 1.0, rtol=1e-8)

    def test_gradient_negative_side(self):
        """For x << -eps: gradient ≈ 0."""
        x = anp.array(-5.0)
        grad = float(autograd.grad(smooth_macaulay)(x))
        assert abs(grad) < 1e-10, f"Expected ≈ 0, got {grad}"

    def test_non_negative(self):
        """smooth_macaulay is non-negative for all inputs."""
        xs = anp.linspace(-10.0, 10.0, 1000)
        vals = smooth_macaulay(xs)
        assert anp.all(vals >= 0.0)


# ---------------------------------------------------------------------------
# smooth_direction
# ---------------------------------------------------------------------------

class TestSmoothDirection:
    def test_unit_vector_unchanged(self):
        """smooth_direction of a unit vector ≈ that vector."""
        v = anp.array([1.0, 0.0, 0.0])
        result = smooth_direction(v)
        np.testing.assert_allclose(np.array(result), np.array(v), rtol=1e-10)

    def test_known_direction(self):
        """smooth_direction([3, 4]) ≈ [0.6, 0.8]."""
        v = anp.array([3.0, 4.0])
        result = smooth_direction(v)
        expected = anp.array([0.6, 0.8])
        np.testing.assert_allclose(np.array(result), np.array(expected), rtol=1e-8)

    def test_zero_vector_no_nan(self):
        """smooth_direction(0) returns finite values (not NaN)."""
        v = anp.zeros(6)
        result = smooth_direction(v)
        assert anp.all(anp.isfinite(result)), f"NaN/inf in result: {result}"

    def test_zero_vector_returns_zero(self):
        """smooth_direction(0) = 0 / eps = 0."""
        v = anp.zeros(4)
        result = smooth_direction(v)
        np.testing.assert_allclose(np.array(result), np.zeros(4), atol=1e-10)

    def test_gradient_finite_at_zero(self):
        """Gradient of smooth_direction at v=0 is finite."""
        v = anp.zeros(3)
        jac = autograd.jacobian(smooth_direction)(v)
        assert anp.all(anp.isfinite(jac)), f"Jacobian has inf/NaN at zero"

    def test_negative_vector(self):
        """smooth_direction(-v) ≈ -smooth_direction(v) for v ≠ 0."""
        v = anp.array([1.0, 2.0, -3.0])
        np.testing.assert_allclose(
            np.array(smooth_direction(-v)),
            np.array(-smooth_direction(v)),
            rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# Integration: smooth_abs in AF kinematic hardening pattern
# ---------------------------------------------------------------------------

class TestAFPattern:
    """Verify the AF model's n_hat computation pattern works with smooth_abs."""

    def test_n_hat_finite_at_zero_xi(self):
        """When xi = 0 (stress = alpha), n_hat should be finite, not NaN."""
        import manforge
        from manforge.models.af_kinematic import AFKinematic3D

        model = AFKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0)
        xi = anp.zeros(6)
        s_xi = model._dev(xi)
        vm_safe = smooth_abs(model._vonmises(xi))
        n_hat = s_xi / vm_safe
        assert anp.all(anp.isfinite(n_hat)), f"n_hat has NaN/inf at xi=0: {n_hat}"

    def test_n_hat_finite_gradient_at_zero_xi(self):
        """Gradient of alpha w.r.t. dlambda should be finite when xi ≈ 0.

        Uses sum(alpha**2) as the scalar output to avoid the non-differentiable
        anp.linalg.norm(zeros) that would arise if alpha stays at zero.
        """
        from manforge.models.af_kinematic import AFKinematic3D

        model = AFKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0)
        state0 = model.initial_state()
        stress = anp.zeros(6)

        def alpha_sq_norm(dl):
            items = model.update_state(dl, stress, state0)
            st = {item.name: item.value for item in items}
            return anp.sum(st["alpha"] ** 2)

        # Gradient at dlambda=0 (where xi = stress - alpha = 0)
        grad = autograd.grad(alpha_sq_norm)(anp.array(0.0))
        assert np.isfinite(float(grad)), f"gradient is not finite: {grad}"
