"""Smooth approximations of non-differentiable mathematical functions.

All functions are composed entirely of autograd/numpy primitives and are
fully differentiable everywhere, including at the singular points of their
non-smooth counterparts.

Functions and their parameters:

  smooth_sqrt(x, eps=1e-30)        √(x + ε²)                  ε: denominator stabiliser
  smooth_abs(x, eps=1e-30)         √(x² + ε²)                 ε: denominator stabiliser
  smooth_norm(v, eps=1e-30)        √(v·v + ε²)                ε: denominator stabiliser
  smooth_macaulay(x, eps=1e-30)    (x + smooth_abs(x)) / 2    ε: denominator stabiliser
  smooth_direction(v, eps=1e-30)   v / smooth_norm(v)          ε: denominator stabiliser
  smooth_heaviside(x, beta=50.0)   0.5·(1 + tanh(β·x/2))      β: transition sharpness
  smooth_min(a, b, eps=1e-30)      b + (d - smooth_abs(d)) / 2  ε: denominator stabiliser
  smooth_max(a, b, eps=1e-30)      b + (d + smooth_abs(d)) / 2  ε: denominator stabiliser

``eps`` regularises denominators at zero; with float64, ``sqrt(1e-30) ≈ 1e-15``.
``beta`` controls the steepness of the Heaviside step: larger β → sharper transition.
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


def smooth_heaviside(x: anp.ndarray, beta: float = 50.0) -> anp.ndarray:
    """Smooth Heaviside step: 0.5·(1 + tanh(β·x/2)).

    tanh formulation avoids exp overflow that would arise from 1/(1+exp(-βx)).
    At x=0 returns exactly 0.5; larger β gives a sharper transition.
    """
    return 0.5 * (1.0 + anp.tanh(0.5 * beta * x))


def smooth_min(a: anp.ndarray, b: anp.ndarray, eps: float = _DEFAULT_EPS) -> anp.ndarray:
    """Smooth minimum: b + (d - smooth_abs(d)) / 2, where d = a - b.

    Avoids the a+b intermediate to prevent overflow when a and b are large
    with the same sign.
    """
    d = a - b
    return b + 0.5 * (d - smooth_abs(d, eps))


def smooth_max(a: anp.ndarray, b: anp.ndarray, eps: float = _DEFAULT_EPS) -> anp.ndarray:
    """Smooth maximum: b + (d + smooth_abs(d)) / 2, where d = a - b.

    Avoids the a+b intermediate to prevent overflow when a and b are large
    with the same sign.
    """
    d = a - b
    return b + 0.5 * (d + smooth_abs(d, eps))
