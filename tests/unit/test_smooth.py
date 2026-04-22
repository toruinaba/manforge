"""Tests for smooth mathematical utility functions.

Verifies that each function:
1. Returns finite, non-NaN values at the singular point (zero).
2. Has a finite gradient at the singular point.
3. Matches the non-smooth version away from the singular point.
4. Is consistent with the (x=0) limit.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

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
        x = jnp.array(4.0)
        np.testing.assert_allclose(float(smooth_sqrt(x)), 2.0, rtol=1e-10)

    def test_at_zero_returns_eps(self):
        """At x=0: value = sqrt(eps²) = eps."""
        val = float(smooth_sqrt(jnp.array(0.0)))
        np.testing.assert_allclose(val, EPS, rtol=1e-6)

    def test_gradient_finite_at_zero(self):
        """At x=0: gradient is finite (not inf)."""
        grad = float(jax.grad(smooth_sqrt)(jnp.array(0.0)))
        assert np.isfinite(grad), f"gradient = {grad}"

    def test_gradient_matches_sqrt_away_from_zero(self):
        """Away from zero: d/dx smooth_sqrt ≈ d/dx sqrt."""
        x = jnp.array(9.0)
        g_smooth = float(jax.grad(smooth_sqrt)(x))
        g_exact = float(0.5 / jnp.sqrt(x))
        np.testing.assert_allclose(g_smooth, g_exact, rtol=1e-8)

    def test_array_input(self):
        """Works on arrays element-wise."""
        x = jnp.array([0.0, 1.0, 4.0, 9.0])
        result = smooth_sqrt(x)
        expected = jnp.array([EPS, jnp.sqrt(1.0 + EPS**2),
                               jnp.sqrt(4.0 + EPS**2), jnp.sqrt(9.0 + EPS**2)])
        np.testing.assert_allclose(np.array(result), np.array(expected), rtol=1e-10)


# ---------------------------------------------------------------------------
# smooth_abs
# ---------------------------------------------------------------------------

class TestSmoothAbs:
    def test_positive(self):
        """smooth_abs(x) ≈ x for x >> eps."""
        x = jnp.array(1.0)
        np.testing.assert_allclose(float(smooth_abs(x)), 1.0, rtol=1e-10)

    def test_negative(self):
        """smooth_abs(-x) ≈ x for |x| >> eps."""
        x = jnp.array(-3.0)
        np.testing.assert_allclose(float(smooth_abs(x)), 3.0, rtol=1e-10)

    def test_at_zero_value(self):
        """At x=0: value = sqrt(0 + eps²) = eps."""
        val = float(smooth_abs(jnp.array(0.0)))
        np.testing.assert_allclose(val, EPS, rtol=1e-6)

    def test_gradient_at_zero_is_zero(self):
        """At x=0: gradient = 0 (smooth minimum, not kink)."""
        grad = float(jax.grad(smooth_abs)(jnp.array(0.0)))
        np.testing.assert_allclose(grad, 0.0, atol=1e-10)

    def test_gradient_positive_side(self):
        """For x >> eps: d/dx smooth_abs ≈ +1."""
        x = jnp.array(1.0)
        grad = float(jax.grad(smooth_abs)(x))
        np.testing.assert_allclose(grad, 1.0, rtol=1e-8)

    def test_gradient_negative_side(self):
        """For x << -eps: d/dx smooth_abs ≈ -1."""
        x = jnp.array(-1.0)
        grad = float(jax.grad(smooth_abs)(x))
        np.testing.assert_allclose(grad, -1.0, rtol=1e-8)

    def test_no_nan_anywhere(self):
        """No NaN for any finite input."""
        xs = jnp.linspace(-10.0, 10.0, 1000)
        vals = smooth_abs(xs)
        assert jnp.all(jnp.isfinite(vals))

    def test_symmetric(self):
        """smooth_abs(-x) == smooth_abs(x)."""
        xs = jnp.linspace(-5.0, 5.0, 100)
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
        v = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(float(smooth_norm(v)), 1.0, rtol=1e-10)

    def test_known_norm(self):
        """smooth_norm([3, 4]) ≈ 5."""
        v = jnp.array([3.0, 4.0])
        np.testing.assert_allclose(float(smooth_norm(v)), 5.0, rtol=1e-10)

    def test_at_zero_returns_eps(self):
        """At v=0: value = eps."""
        v = jnp.zeros(6)
        val = float(smooth_norm(v))
        np.testing.assert_allclose(val, EPS, rtol=1e-6)

    def test_gradient_finite_at_zero(self):
        """At v=0: gradient is finite."""
        v = jnp.zeros(3)
        grad = jax.grad(smooth_norm)(v)
        assert jnp.all(jnp.isfinite(grad)), f"gradient has inf/NaN: {grad}"

    def test_gradient_zero_vector_is_zero(self):
        """At v=0: gradient = 0 (smooth minimum)."""
        v = jnp.zeros(3)
        grad = jax.grad(smooth_norm)(v)
        np.testing.assert_allclose(np.array(grad), np.zeros(3), atol=1e-10)

    def test_gradient_matches_norm_away_from_zero(self):
        """Away from zero: gradient ≈ v / ||v||."""
        v = jnp.array([3.0, 4.0])
        grad = jax.grad(smooth_norm)(v)
        expected = v / 5.0  # exact gradient of ||v|| at (3, 4)
        np.testing.assert_allclose(np.array(grad), np.array(expected), rtol=1e-8)


# ---------------------------------------------------------------------------
# smooth_macaulay
# ---------------------------------------------------------------------------

class TestSmoothMacaulay:
    def test_positive_value(self):
        """For x >> eps: smooth_macaulay(x) ≈ x."""
        x = jnp.array(5.0)
        np.testing.assert_allclose(float(smooth_macaulay(x)), 5.0, rtol=1e-10)

    def test_negative_value(self):
        """For x << -eps: smooth_macaulay(x) ≈ 0."""
        x = jnp.array(-5.0)
        val = float(smooth_macaulay(x))
        assert val < 1e-10, f"Expected ≈ 0, got {val}"

    def test_at_zero(self):
        """At x=0: value = eps/2 (small positive)."""
        val = float(smooth_macaulay(jnp.array(0.0)))
        np.testing.assert_allclose(val, EPS / 2.0, rtol=1e-6)

    def test_gradient_at_zero(self):
        """At x=0: gradient = 0.5."""
        grad = float(jax.grad(smooth_macaulay)(jnp.array(0.0)))
        np.testing.assert_allclose(grad, 0.5, atol=1e-10)

    def test_gradient_positive_side(self):
        """For x >> eps: gradient ≈ 1."""
        x = jnp.array(5.0)
        grad = float(jax.grad(smooth_macaulay)(x))
        np.testing.assert_allclose(grad, 1.0, rtol=1e-8)

    def test_gradient_negative_side(self):
        """For x << -eps: gradient ≈ 0."""
        x = jnp.array(-5.0)
        grad = float(jax.grad(smooth_macaulay)(x))
        assert abs(grad) < 1e-10, f"Expected ≈ 0, got {grad}"

    def test_non_negative(self):
        """smooth_macaulay is non-negative for all inputs."""
        xs = jnp.linspace(-10.0, 10.0, 1000)
        vals = smooth_macaulay(xs)
        assert jnp.all(vals >= 0.0)


# ---------------------------------------------------------------------------
# smooth_direction
# ---------------------------------------------------------------------------

class TestSmoothDirection:
    def test_unit_vector_unchanged(self):
        """smooth_direction of a unit vector ≈ that vector."""
        v = jnp.array([1.0, 0.0, 0.0])
        result = smooth_direction(v)
        np.testing.assert_allclose(np.array(result), np.array(v), rtol=1e-10)

    def test_known_direction(self):
        """smooth_direction([3, 4]) ≈ [0.6, 0.8]."""
        v = jnp.array([3.0, 4.0])
        result = smooth_direction(v)
        expected = jnp.array([0.6, 0.8])
        np.testing.assert_allclose(np.array(result), np.array(expected), rtol=1e-8)

    def test_zero_vector_no_nan(self):
        """smooth_direction(0) returns finite values (not NaN)."""
        v = jnp.zeros(6)
        result = smooth_direction(v)
        assert jnp.all(jnp.isfinite(result)), f"NaN/inf in result: {result}"

    def test_zero_vector_returns_zero(self):
        """smooth_direction(0) = 0 / eps = 0."""
        v = jnp.zeros(4)
        result = smooth_direction(v)
        np.testing.assert_allclose(np.array(result), np.zeros(4), atol=1e-10)

    def test_gradient_finite_at_zero(self):
        """Gradient of smooth_direction at v=0 is finite."""
        v = jnp.zeros(3)
        jac = jax.jacobian(smooth_direction)(v)
        assert jnp.all(jnp.isfinite(jac)), f"Jacobian has inf/NaN at zero"

    def test_negative_vector(self):
        """smooth_direction(-v) ≈ -smooth_direction(v) for v ≠ 0."""
        v = jnp.array([1.0, 2.0, -3.0])
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
        xi = jnp.zeros(6)
        s_xi = model._dev(xi)
        vm_safe = smooth_abs(model._vonmises(xi))
        n_hat = s_xi / vm_safe
        assert jnp.all(jnp.isfinite(n_hat)), f"n_hat has NaN/inf at xi=0: {n_hat}"

    def test_n_hat_finite_gradient_at_zero_xi(self):
        """Gradient of alpha w.r.t. dlambda should be finite when xi ≈ 0.

        Uses sum(alpha**2) as the scalar output to avoid the non-differentiable
        jnp.linalg.norm(zeros) that would arise if alpha stays at zero.
        """
        from manforge.models.af_kinematic import AFKinematic3D

        model = AFKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0)
        state0 = model.initial_state()
        stress = jnp.zeros(6)

        def alpha_sq_norm(dl):
            st = model.update_state(dl, stress, state0)
            return jnp.sum(st["alpha"] ** 2)

        # Gradient at dlambda=0 (where xi = stress - alpha = 0)
        grad = jax.grad(alpha_sq_norm)(jnp.array(0.0))
        assert np.isfinite(float(grad)), f"gradient is not finite: {grad}"
