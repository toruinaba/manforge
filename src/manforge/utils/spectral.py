"""Spectral (principal-value) functions of a Voigt stress, with a robust gradient.

Tension/compression splits such as Σ⟨σ_i⟩₊² need the principal values σ_i.
Differentiating through ``anp.linalg.eigh`` is the obvious route and the wrong
one: its derivative carries 1/(λ_i − λ_j), which is NaN whenever two principal
values coincide — and that is the *common* case, not an edge case.  Uniaxial
stress (σ2 = σ3 = 0), plane stress, pure shear and the zero state all have
repeated principal values.

A symmetric spectral function F(σ) = Σ f(λ_i) is nevertheless smooth there
whenever f is, and its gradient is the matrix function

    ∂F/∂σ = Σ f'(λ_i) n_i ⊗ n_i = f'(σ)

which is unique even inside a degenerate eigenspace (f' takes one value on
it, so the arbitrary choice of eigenvectors cancels).  :func:`spectral_trace`
is an autograd primitive whose VJP evaluates exactly that, so it never
divides by an eigenvalue gap.

Only first derivatives are provided.  That covers every residual and tangent
the framework builds (the NR Jacobian and the consistent tangent differentiate
the residual once).  Differentiating a quantity that already contains the
gradient — e.g. using ∂F/∂σ as a flow direction — would need the
Daleckii–Krein second derivative and fails loudly here rather than silently.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from autograd.extend import defvjp, primitive

from manforge.core.dimension import StressDimension

# Tensor index pairs for the stored shear components, in Voigt order
# [12, 13, 23]; dimensions with fewer shears keep the leading ones.
_SHEAR_PAIRS = ((0, 1), (0, 2), (1, 2))


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


def _tensor_gradient_to_voigt(G: np.ndarray, dimension: StressDimension) -> np.ndarray:
    """∂F/∂T (symmetric) → ∂F/∂v for the stored Voigt components.

    A stored shear v_k sits at both T[a, b] and T[b, a], so its derivative is
    G[a, b] + G[b, a] = 2 G[a, b].
    """
    out = np.empty(dimension.ntens)
    for i in range(dimension.ndi):
        out[i] = G[i, i]
    for k in range(dimension.nshr):
        a, b = _SHEAR_PAIRS[k]
        out[dimension.ndi + k] = 2.0 * G[a, b]
    return out


def principal_values(stress: np.ndarray, dimension: StressDimension) -> np.ndarray:
    """Principal stresses in ascending order (numpy; not differentiable)."""
    return np.linalg.eigvalsh(voigt_stress_to_tensor(stress, dimension))


@primitive
def _spectral_trace(stress, dimension, f, df):
    lam = np.linalg.eigvalsh(voigt_stress_to_tensor(stress, dimension))
    return np.sum(f(lam))


def _spectral_trace_vjp(ans, stress, dimension, f, df):
    lam, N = np.linalg.eigh(voigt_stress_to_tensor(stress, dimension))
    G = (N * df(lam)) @ N.T          # Σ f'(λ_i) n_i ⊗ n_i
    grad = _tensor_gradient_to_voigt(G, dimension)
    return lambda g: g * grad


defvjp(_spectral_trace, _spectral_trace_vjp)


def spectral_trace(
    stress,
    dimension: StressDimension,
    f: Callable[[np.ndarray], np.ndarray],
    df: Callable[[np.ndarray], np.ndarray],
):
    """F(σ) = Σ_i f(σ_i) over the principal stresses, autograd-differentiable.

    Parameters
    ----------
    stress : array, shape (ntens,)
        Stress Voigt vector (physical shear).
    dimension : StressDimension
        Layout of ``stress``.
    f, df : callable
        Elementwise numpy functions f and f'.  ``df`` must be the exact
        derivative and f should be at least C¹ for a continuous Jacobian.

    Notes
    -----
    See the module docstring for why the gradient is computed as the matrix
    function f'(σ) instead of through ``eigh``.
    """
    return _spectral_trace(stress, dimension, f, df)


def _pos_sq(lam):
    return np.maximum(lam, 0.0) ** 2


def _d_pos_sq(lam):
    return 2.0 * np.maximum(lam, 0.0)


def _neg_sq(lam):
    return np.minimum(lam, 0.0) ** 2


def _d_neg_sq(lam):
    return 2.0 * np.minimum(lam, 0.0)


def positive_part_sq(stress, dimension: StressDimension):
    """Σ⟨σ_i⟩₊² — the tensile principal part (C¹ in σ)."""
    return spectral_trace(stress, dimension, _pos_sq, _d_pos_sq)


def negative_part_sq(stress, dimension: StressDimension):
    """Σ⟨σ_i⟩₋² = Σ min(σ_i, 0)² — the compressive principal part (C¹ in σ)."""
    return spectral_trace(stress, dimension, _neg_sq, _d_neg_sq)
