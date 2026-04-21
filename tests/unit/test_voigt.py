"""Unit tests for utils.voigt conversion functions."""

import numpy as np
import jax.numpy as jnp

from manforge.utils.voigt import from_mandel, from_voigt, to_mandel, to_voigt


def test_voigt_roundtrip():
    T = jnp.array([[1.0, 2.0, 3.0],
                   [2.0, 4.0, 5.0],
                   [3.0, 5.0, 6.0]])
    np.testing.assert_allclose(np.asarray(from_voigt(to_voigt(T))), np.asarray(T))


def test_mandel_roundtrip():
    v = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    np.testing.assert_allclose(np.asarray(from_mandel(to_mandel(v))), np.asarray(v), atol=1e-12)


def test_mandel_inner_product_equals_double_contraction():
    # A:B = a_mandel · b_mandel  (diagonal tensors)
    a = jnp.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    b = jnp.array([4.0, 5.0, 6.0, 0.0, 0.0, 0.0])
    # A:B = 1*4 + 2*5 + 3*6 = 32
    np.testing.assert_allclose(float(jnp.dot(to_mandel(a), to_mandel(b))), 32.0)
