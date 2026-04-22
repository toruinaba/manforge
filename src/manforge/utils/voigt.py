"""Voigt notation utilities.

Conventions used throughout this package:
  Stress Voigt : σ = [σ11, σ22, σ33, σ12, σ13, σ23]  (6-component for 3D)
  Strain Voigt : ε = [ε11, ε22, ε33, γ12, γ13, γ23]  (engineering shear γ=2ε)
  Mandel       : shear components scaled by √2 so that the vector dot-product
                 equals the tensor double-contraction A:B = a_mandel · b_mandel
                   σ_mandel = [σ11, σ22, σ33, √2·σ12, √2·σ13, √2·σ23]
"""

import math

import autograd.numpy as anp
import numpy as np

_SQRT2 = math.sqrt(2.0)

# √2 factor applied to shear indices (3, 4, 5) for Mandel notation — 3D default
_MANDEL_FACTORS = anp.array([1.0, 1.0, 1.0, _SQRT2, _SQRT2, _SQRT2])

# Index pairs for the 6 independent components of a symmetric 3×3 tensor
# Order: (0,0), (1,1), (2,2), (0,1), (0,2), (1,2)
_I = np.array([0, 1, 2, 0, 0, 1])
_J = np.array([0, 1, 2, 1, 2, 2])


def to_voigt(tensor_3x3: anp.ndarray) -> anp.ndarray:
    """Convert a symmetric 3×3 tensor to a 6-component Voigt vector."""
    return tensor_3x3[_I, _J]


def from_voigt(v: anp.ndarray) -> anp.ndarray:
    """Convert a 6-component Voigt vector to a symmetric 3×3 tensor."""
    T = np.zeros((3, 3))
    T[_I, _J] = np.asarray(v)
    T[_J, _I] = np.asarray(v)
    return anp.array(T)


def to_mandel(v_voigt: anp.ndarray, ss=None) -> anp.ndarray:
    """Convert a Voigt vector to Mandel notation."""
    if ss is None:
        return v_voigt * _MANDEL_FACTORS
    return v_voigt * ss.mandel_factors_np


def from_mandel(v_mandel: anp.ndarray, ss=None) -> anp.ndarray:
    """Convert a Mandel vector back to Voigt notation."""
    if ss is None:
        return v_mandel / _MANDEL_FACTORS
    return v_mandel / ss.mandel_factors_np
