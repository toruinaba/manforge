"""Smooth approximations of non-differentiable mathematical functions.

All functions are composed entirely of JAX primitives and are therefore
fully differentiable everywhere, including at the singular points (zero,
kinks) of their non-smooth counterparts.

The smoothing parameter ``eps`` controls how closely the smooth version
approximates the original:

- For ``|x| >> sqrt(eps)``, the smooth value matches the exact value to
  floating-point precision.
- At ``x = 0`` (or ``v = 0``), the smooth value equals ``sqrt(eps)`` and
  the gradient is finite (approximately ``1 / (2 * sqrt(eps))`` for
  ``smooth_sqrt``).

Default: ``eps = 1e-30``.  With float64, ``sqrt(1e-30) ≈ 3e-16``, which
is negligible against stress values in MPa.

Functions
---------
smooth_sqrt
    √(x + ε²) — smooth square root; the primitive from which all others derive.
smooth_abs
    √(x² + ε²) — smooth absolute value.
smooth_norm
    √(v·v + ε²) — smooth Euclidean norm of a vector.
smooth_macaulay
    (x + smooth_abs(x)) / 2 — smooth Macaulay bracket ⟨x⟩ = max(x, 0).
smooth_direction
    v / smooth_norm(v) — smooth unit direction; safe at the zero vector.

Notes
-----
Why not ``jnp.maximum(vm, eps)`` (the original AF pattern)?

``jnp.maximum`` is a valid gradient gate: when ``vm < eps``, the gradient
through ``vm`` is zeroed out, which prevents the ``inf`` gradient from
``jnp.sqrt(0)`` from propagating.  However, it has a non-smooth kink at
the clamping boundary, which matters when the computation graph is
differentiated a second time (e.g. for ``jax.jacobian`` of a function
that itself calls ``jax.grad``).  ``smooth_abs`` is C∞ everywhere.
"""

import jax.numpy as jnp

# Smoothing parameter ε².  With float64, sqrt(1e-30) ≈ 3e-16.
_DEFAULT_EPS: float = 1e-30


def smooth_sqrt(x: jnp.ndarray, eps: float = _DEFAULT_EPS) -> jnp.ndarray:
    """Smooth square root: √(x + ε²).

    Replaces ``jnp.sqrt(x)`` where ``x`` may reach zero.

    Parameters
    ----------
    x : jnp.ndarray
        Non-negative scalar or array.  Negative values are handled
        gracefully (sqrt of a small positive number).
    eps : float, optional
        Smoothing parameter (default 1e-30).

    Returns
    -------
    jnp.ndarray
        Approximate square root, shape same as ``x``.

    Notes
    -----
    At ``x = 0``: value = ``sqrt(eps)`` ≈ 3e-16 (for eps=1e-30), gradient =
    ``1 / (2 * sqrt(eps))`` ≈ 1.67e15 * sqrt(eps).  Concretely, gradient
    at ``x=0`` is ``1 / (2 * eps^(1/2))`` which is finite for any ``eps > 0``.

    Compared to ``jnp.sqrt``:
    - ``jnp.sqrt(0)`` → value 0, gradient ``+inf``
    - ``smooth_sqrt(0)`` → value ``sqrt(eps)``, gradient finite
    """
    return jnp.sqrt(x + eps**2)


def smooth_abs(x: jnp.ndarray, eps: float = _DEFAULT_EPS) -> jnp.ndarray:
    """Smooth absolute value: √(x² + ε²).

    Replaces ``jnp.abs(x)`` where the kink at ``x = 0`` would cause
    subgradient issues in higher-order differentiation.

    Parameters
    ----------
    x : jnp.ndarray
        Scalar or array.
    eps : float, optional

    Returns
    -------
    jnp.ndarray
        Smooth approximation of ``|x|``, shape same as ``x``.

    Notes
    -----
    At ``x = 0``: value = ``eps``, gradient = 0 (not the subgradient ±1).
    For ``|x| >> eps``: value ≈ ``|x|`` to floating-point precision.
    """
    return smooth_sqrt(x**2, eps)


def smooth_norm(v: jnp.ndarray, eps: float = _DEFAULT_EPS) -> jnp.ndarray:
    """Smooth Euclidean norm: √(v·v + ε²).

    Replaces ``jnp.linalg.norm(v)`` or ``jnp.sqrt(jnp.dot(v, v))`` where
    ``v`` may be the zero vector.

    Parameters
    ----------
    v : jnp.ndarray, shape (n,)
        Any vector (Voigt, Mandel, or otherwise).
    eps : float, optional

    Returns
    -------
    jnp.ndarray
        Scalar smooth norm.

    Notes
    -----
    At ``v = 0``: value = ``eps``, gradient = zero vector (well-defined).
    For use with Mandel-scaled vectors, pass ``to_mandel(v, ss)`` as input
    to get the Mandel tensor norm √(T:T).
    """
    return smooth_sqrt(jnp.dot(v, v), eps)


def smooth_macaulay(x: jnp.ndarray, eps: float = _DEFAULT_EPS) -> jnp.ndarray:
    """Smooth Macaulay bracket: (x + smooth_abs(x)) / 2 ≈ max(x, 0).

    The Macaulay bracket ⟨x⟩ appears in loading/unloading criteria
    (e.g. in Yoshida-Uemori and subloading surface models).

    Parameters
    ----------
    x : jnp.ndarray
        Scalar or array.
    eps : float, optional

    Returns
    -------
    jnp.ndarray
        Smooth approximation of ``max(x, 0)``.

    Notes
    -----
    At ``x = 0``: value = ``eps / 2``, gradient = 0.5.
    For ``x >> eps``: value ≈ ``x``.
    For ``x << -eps``: value ≈ 0.

    Equivalent to the smooth ramp function used in penalty methods.
    """
    return (x + smooth_abs(x, eps)) / 2.0


def smooth_direction(v: jnp.ndarray, eps: float = _DEFAULT_EPS) -> jnp.ndarray:
    """Smooth unit direction: v / smooth_norm(v).

    Returns the unit direction of ``v`` in a form that is differentiable
    everywhere, including at ``v = 0``.

    Parameters
    ----------
    v : jnp.ndarray, shape (n,)
        Any vector.  Need not be in Mandel notation.
    eps : float, optional

    Returns
    -------
    jnp.ndarray, shape (n,)
        Smooth unit direction, shape same as ``v``.

    Notes
    -----
    At ``v = 0``: returns ``v / eps = 0 / eps = 0`` (not NaN).
    The gradient at ``v = 0`` is ``I / eps``, which is large but finite.

    Typical usage in hardening rules::

        n_hat = smooth_direction(s_xi_mandel)  # unit deviatoric direction

    For AF kinematic hardening, the Voigt deviatoric ``s_xi`` can be
    divided by a scalar smooth norm::

        vm_safe = smooth_abs(self._vonmises(xi))
        n_hat = s_xi / vm_safe
    """
    return v / smooth_norm(v, eps)
