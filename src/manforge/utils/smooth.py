"""Smooth approximations of non-differentiable mathematical functions.

All functions are composed entirely of autograd/numpy primitives and are
fully differentiable everywhere, including at the singular points of their
non-smooth counterparts.

Default: ``eps = 1e-30``.  With float64, ``sqrt(1e-30) ≈ 3e-16``.
"""

import autograd.numpy as anp

_DEFAULT_EPS: float = 1e-30


def smooth_sqrt(x: anp.ndarray, eps: float = _DEFAULT_EPS) -> anp.ndarray:
    """Smooth square root: √(x + ε²)."""
    return anp.sqrt(x + eps**2)


def smooth_abs(x: anp.ndarray, eps: float = _DEFAULT_EPS) -> anp.ndarray:
    """Smooth absolute value: √(x² + ε²)."""
    return smooth_sqrt(x**2, eps)


def smooth_norm(v: anp.ndarray, eps: float = _DEFAULT_EPS) -> anp.ndarray:
    """Smooth Euclidean norm: √(v·v + ε²)."""
    return smooth_sqrt(anp.dot(v, v), eps)


def smooth_macaulay(x: anp.ndarray, eps: float = _DEFAULT_EPS) -> anp.ndarray:
    """Smooth Macaulay bracket: (x + smooth_abs(x)) / 2 ≈ max(x, 0)."""
    return (x + smooth_abs(x, eps)) / 2.0


def smooth_direction(v: anp.ndarray, eps: float = _DEFAULT_EPS) -> anp.ndarray:
    """Smooth unit direction: v / smooth_norm(v)."""
    return v / smooth_norm(v, eps)
