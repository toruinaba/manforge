"""Spectral (principal-value) functions of a Voigt stress, with robust derivatives.

Tension/compression splits such as σ₊ = Σ⟨σ_i⟩ n_i⊗n_i or Σ⟨σ_i⟩₊² need the
principal values σ_i.  Differentiating through ``anp.linalg.eigh`` is the
obvious route and the wrong one: its derivative carries 1/(λ_i − λ_j), which
is NaN whenever two principal values coincide — and that is the *common*
case.  Uniaxial stress (σ2 = σ3 = 0), plane stress, pure shear and the zero
state all have repeated principal values.

Two primitives avoid it:

:func:`spectral_trace` — F(σ) = Σ f(λ_i).
    Gradient is the matrix function ∂F/∂σ = f'(σ) = Σ f'(λ_i) n_i⊗n_i, unique
    even inside a degenerate eigenspace.  It is evaluated through
    :func:`matrix_function`, so when f'' is supplied the gradient is itself
    differentiable and the Hessian exists.

:func:`matrix_function` — g(σ) = Σ g(λ_i) n_i⊗n_i (as a stress Voigt vector).
    Derivative by the Daleckii–Krein formula, in the principal basis::

        dg = N (Φ ∘ (Nᵀ dσ N)) Nᵀ,
        Φ_ij = (g(λ_i) − g(λ_j)) / (λ_i − λ_j)   (λ_i ≠ λ_j)
        Φ_ij = g'(λ_i)                           (λ_i = λ_j)

    Φ is evaluated without ever dividing by a vanishing gap, so the result is
    exact at repeated principal values.  Only first derivatives of g are
    provided; a third derivative of a spectral trace fails loudly.

Zero-eigenvalue convention
--------------------------
For the Macaulay split g(λ) = ⟨λ⟩ the derivative g'(0) does not exist: the
response genuinely has a kink (a lateral stress of either sign switches the
stiffness).  That is part of the constitutive model, not a numerical
artefact, so it is fixed by convention rather than smoothed away::

    H(0) := c,    default  c = DEFAULT_HEAVISIDE_AT_ZERO = 1/2

c enters Φ on every pair of (numerically) equal principal values at zero —
including a doubly repeated zero, where it also makes P₊ = ∂σ₊/∂σ unique
regardless of the eigenvector choice.  c = 1/2 averages the one-sided
derivatives; it is **provisional** until the Fortran UMAT's convention is
fixed, and every public function takes ``c`` so the value can follow it.
The scalar brackets (:func:`macaulay_sq`, :func:`neg_macaulay_sq`, used for
⟨tr σ⟩²) apply the same c.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import autograd.numpy as anp
import numpy as np
from autograd.extend import defvjp, primitive

if TYPE_CHECKING:
    # Annotations only.  returnmap.core.dimension imports returnmap.utils, so a
    # runtime import here would be circular.
    from returnmap.core.dimension import StressDimension

#: Value taken by the Heaviside step at 0 (provisional; see module docstring).
DEFAULT_HEAVISIDE_AT_ZERO: float = 0.5

# Tensor index pairs for the stored shear components, in Voigt order
# [12, 13, 23]; dimensions with fewer shears keep the leading ones.
_SHEAR_PAIRS = ((0, 1), (0, 2), (1, 2))

# Principal values closer than this (relative to the largest |λ|) are treated
# as equal in Φ.  The divided difference loses ~half the digits as the gap
# closes; switching to the derivative well before that keeps Φ accurate.
_EQUAL_RTOL = 1e-8


# ---------------------------------------------------------------------------
# Voigt ↔ tensor
# ---------------------------------------------------------------------------

def voigt_stress_to_tensor(v: np.ndarray, dimension: StressDimension) -> np.ndarray:
    """Physical-shear stress Voigt vector → symmetric 3×3 tensor (numpy only).

    Unstored direct components are zero — the plane-stress / uniaxial meaning
    of a *stress* vector (σ33 = 0, σ22 = σ33 = 0).
    """
    v = np.asarray(v, dtype=float)
    T = np.zeros((3, 3))
    for i in range(dimension.ndi):
        T[i, i] = v[i]
    for k in range(dimension.nshr):
        a, b = _SHEAR_PAIRS[k]
        T[a, b] = T[b, a] = v[dimension.ndi + k]
    return T


def _tensor_to_voigt(T: np.ndarray, dimension: StressDimension) -> np.ndarray:
    """Stored components of a symmetric tensor as a stress Voigt vector."""
    out = np.empty(dimension.ntens)
    for i in range(dimension.ndi):
        out[i] = T[i, i]
    for k in range(dimension.nshr):
        a, b = _SHEAR_PAIRS[k]
        out[dimension.ndi + k] = T[a, b]
    return out


def _tensor_gradient_to_voigt(G: np.ndarray, dimension: StressDimension) -> np.ndarray:
    """∂F/∂T (symmetric) → ∂F/∂v for the stored Voigt components.

    A stored shear v_k sits at both T[a, b] and T[b, a], so its derivative is
    G[a, b] + G[b, a] = 2 G[a, b].
    """
    out = _tensor_to_voigt(G, dimension)
    out[dimension.ndi:] *= 2.0
    return out


def _voigt_cotangent_to_tensor(g: np.ndarray, dimension: StressDimension) -> np.ndarray:
    """Cotangent of a stress Voigt *output* → symmetric tensor cotangent Ĝ.

    Chosen so that ``Ĝ : dT == g · d(voigt(T))`` for symmetric dT: a stored
    shear output is T[a, b] alone, so its weight is split over both halves.
    """
    G = np.zeros((3, 3))
    for i in range(dimension.ndi):
        G[i, i] = g[i]
    for k in range(dimension.nshr):
        a, b = _SHEAR_PAIRS[k]
        G[a, b] = G[b, a] = 0.5 * g[dimension.ndi + k]
    return G


def principal_values(stress: np.ndarray, dimension: StressDimension) -> np.ndarray:
    """Principal stresses in ascending order (numpy; not differentiable)."""
    return np.linalg.eigvalsh(voigt_stress_to_tensor(stress, dimension))


# ---------------------------------------------------------------------------
# Matrix function g(σ) with Daleckii–Krein derivative
# ---------------------------------------------------------------------------

def _divided_differences(lam, g, dg):
    """Φ_ij — divided differences, g' on (numerically) equal pairs.

    Principal values within round-off of zero are snapped to exactly 0 first:
    eigh returns ±1e-14 for an exact zero of a sheared tensor, and the sign of
    that noise must not decide between g'(0⁺) and g'(0⁻) — the convention at a
    kink of g (H(0) = c) applies.
    """
    tol = _EQUAL_RTOL * max(1.0, float(np.max(np.abs(lam))))
    lam = np.where(np.abs(lam) <= tol, 0.0, lam)
    gl = g(lam)
    Phi = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            gap = lam[i] - lam[j]
            if abs(gap) <= tol:
                # g' at the pair's midpoint: at a shared zero this is H(0) = c.
                Phi[i, j] = dg(np.array([0.5 * (lam[i] + lam[j])]))[0]
            else:
                Phi[i, j] = (gl[i] - gl[j]) / gap
    return Phi


@primitive
def _matrix_function(stress, dimension, g, dg):
    lam, N = np.linalg.eigh(voigt_stress_to_tensor(stress, dimension))
    return _tensor_to_voigt((N * g(lam)) @ N.T, dimension)


def _matrix_function_vjp(ans, stress, dimension, g, dg):
    lam, N = np.linalg.eigh(voigt_stress_to_tensor(stress, dimension))
    Phi = _divided_differences(lam, g, dg)

    def vjp(cot):
        G_hat = _voigt_cotangent_to_tensor(np.asarray(cot, dtype=float), dimension)
        H = N @ (Phi * (N.T @ G_hat @ N)) @ N.T     # Φ symmetric ⇒ H symmetric
        return _tensor_gradient_to_voigt(H, dimension)

    return vjp


defvjp(_matrix_function, _matrix_function_vjp)


def matrix_function(
    stress,
    dimension: StressDimension,
    g: Callable[[np.ndarray], np.ndarray],
    dg: Callable[[np.ndarray], np.ndarray],
):
    """g(σ) = Σ g(σ_i) n_i⊗n_i as a stress Voigt vector, autograd-differentiable.

    ``g`` and ``dg`` are elementwise numpy functions; ``dg`` is used on
    (numerically) repeated principal values and must encode any convention at
    a kink of g.  Differentiable once (Daleckii–Krein).
    """
    return _matrix_function(stress, dimension, g, dg)


# ---------------------------------------------------------------------------
# Spectral trace F(σ) = Σ f(σ_i)
# ---------------------------------------------------------------------------

@primitive
def _spectral_trace(stress, dimension, f, df, d2f):
    lam = np.linalg.eigvalsh(voigt_stress_to_tensor(stress, dimension))
    return np.sum(f(lam))


def _spectral_trace_vjp(ans, stress, dimension, f, df, d2f):
    if d2f is None:
        # First derivatives only: a numpy gradient, so differentiating it again
        # raises instead of silently treating it as constant.
        lam, N = np.linalg.eigh(voigt_stress_to_tensor(stress, dimension))
        grad = _tensor_gradient_to_voigt((N * df(lam)) @ N.T, dimension)
        return lambda g: g * grad
    # Differentiable gradient: ∂F/∂v = shear-doubled voigt(f'(σ)).
    factors = anp.array([1.0] * dimension.ndi + [2.0] * dimension.nshr)
    return lambda g: g * factors * _matrix_function(stress, dimension, df, d2f)


defvjp(_spectral_trace, _spectral_trace_vjp)


def spectral_trace(
    stress,
    dimension: StressDimension,
    f: Callable[[np.ndarray], np.ndarray],
    df: Callable[[np.ndarray], np.ndarray],
    d2f: "Callable[[np.ndarray], np.ndarray] | None" = None,
):
    """F(σ) = Σ_i f(σ_i) over the principal stresses, autograd-differentiable.

    Parameters
    ----------
    stress : array, shape (ntens,)
        Stress Voigt vector (physical shear).
    dimension : StressDimension
        Layout of ``stress``.
    f, df : callable
        Elementwise numpy functions f and f'.  f should be at least C¹.
    d2f : callable, optional
        f''.  When given, the gradient is a :func:`matrix_function` of f' and
        the Hessian is available (e.g. the Jacobian of a residual that
        contains ∂F/∂σ).  Without it, only first derivatives exist.
    """
    return _spectral_trace(stress, dimension, f, df, d2f)


# ---------------------------------------------------------------------------
# Macaulay splits (with the zero-eigenvalue convention)
# ---------------------------------------------------------------------------

def _heaviside(c):
    return lambda lam: np.where(lam > 0.0, 1.0, np.where(lam < 0.0, 0.0, c))


def _pos(lam):
    return np.maximum(lam, 0.0)


def _neg(lam):
    return np.minimum(lam, 0.0)


def positive_part(stress, dimension: StressDimension, c: float = DEFAULT_HEAVISIDE_AT_ZERO):
    """σ₊ = Σ⟨σ_i⟩ n_i⊗n_i (stress Voigt).  ∂σ₊/∂σ = P₊ with H(0) = c."""
    return matrix_function(stress, dimension, _pos, _heaviside(c))


def negative_part(stress, dimension: StressDimension, c: float = DEFAULT_HEAVISIDE_AT_ZERO):
    """σ₋ = Σ min(σ_i, 0) n_i⊗n_i.  P₋ = 𝕀 − P₊ for the same c."""
    h = _heaviside(c)
    return matrix_function(stress, dimension, _neg, lambda lam: 1.0 - h(lam))


def positive_part_sq(stress, dimension: StressDimension, c: float = DEFAULT_HEAVISIDE_AT_ZERO):
    """Σ⟨σ_i⟩₊² (C¹; Hessian = 2 P₊ in Voigt form, using H(0) = c)."""
    h = _heaviside(c)
    return spectral_trace(stress, dimension, lambda l: _pos(l) ** 2,
                          lambda l: 2.0 * _pos(l), lambda l: 2.0 * h(l))


def negative_part_sq(stress, dimension: StressDimension, c: float = DEFAULT_HEAVISIDE_AT_ZERO):
    """Σ min(σ_i, 0)² (C¹; Hessian = 2 P₋, using H(0) = c)."""
    h = _heaviside(c)
    return spectral_trace(stress, dimension, lambda l: _neg(l) ** 2,
                          lambda l: 2.0 * _neg(l), lambda l: 2.0 * (1.0 - h(l)))


# Scalar brackets for ⟨tr σ⟩² with the same convention.  The derivative of
# ⟨x⟩² is 2⟨x⟩, whose derivative is 2H_c(x) — composing anp.maximum would give
# autograd's own tie-breaking at 0 instead of c.

@primitive
def _pos_scalar(x, c):
    return np.maximum(x, 0.0)


@primitive
def _heaviside_scalar(x, c):
    return np.where(x > 0.0, 1.0, np.where(x < 0.0, 0.0, c))


defvjp(_pos_scalar, lambda ans, x, c: lambda g: g * _heaviside_scalar(x, c))
defvjp(_heaviside_scalar, lambda ans, x, c: lambda g: 0.0 * g)


@primitive
def _macaulay_sq(x, c):
    return np.maximum(x, 0.0) ** 2


defvjp(_macaulay_sq, lambda ans, x, c: lambda g: g * 2.0 * _pos_scalar(x, c))


def macaulay_sq(x, c: float = DEFAULT_HEAVISIDE_AT_ZERO):
    """⟨x⟩₊² for a scalar; second derivative 2H(x) with H(0) = c."""
    return _macaulay_sq(x, c)


def neg_macaulay_sq(x, c: float = DEFAULT_HEAVISIDE_AT_ZERO):
    """⟨x⟩₋² = min(x, 0)²; second derivative 2(1 − H(x)) with H(0) = c."""
    return _macaulay_sq(-x, 1.0 - c)
