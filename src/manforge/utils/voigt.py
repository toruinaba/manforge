"""Voigt notation utilities.

Conventions used throughout this package:
  Stress Voigt : σ = [σ11, σ22, σ33, σ12, σ13, σ23]  (6-component)
  Strain Voigt : ε = [ε11, ε22, ε33, γ12, γ13, γ23]  (engineering shear γ=2ε)
  Mandel       : shear components scaled by √2 so that the vector dot-product
                 equals the tensor double-contraction A:B = a_mandel · b_mandel
                   σ_mandel = [σ11, σ22, σ33, √2·σ12, √2·σ13, √2·σ23]
"""

import jax.numpy as jnp

# √2 factor applied to shear indices (3, 4, 5) for Mandel notation
_MANDEL_FACTORS = jnp.array([1.0, 1.0, 1.0,
                              jnp.sqrt(2.0), jnp.sqrt(2.0), jnp.sqrt(2.0)])

# Index pairs for the 6 independent components of a symmetric 3×3 tensor
# Order: (0,0), (1,1), (2,2), (0,1), (0,2), (1,2)
_I = jnp.array([0, 1, 2, 0, 0, 1])
_J = jnp.array([0, 1, 2, 1, 2, 2])


def to_voigt(tensor_3x3: jnp.ndarray) -> jnp.ndarray:
    """Convert a symmetric 3×3 tensor to a 6-component Voigt vector.

    Parameters
    ----------
    tensor_3x3 : jnp.ndarray, shape (3, 3)
        Symmetric second-order tensor.

    Returns
    -------
    jnp.ndarray, shape (6,)
        Voigt representation [A11, A22, A33, A12, A13, A23].
    """
    return tensor_3x3[_I, _J]


def from_voigt(v: jnp.ndarray) -> jnp.ndarray:
    """Convert a 6-component Voigt vector to a symmetric 3×3 tensor.

    Parameters
    ----------
    v : jnp.ndarray, shape (6,)
        Voigt vector [A11, A22, A33, A12, A13, A23].

    Returns
    -------
    jnp.ndarray, shape (3, 3)
        Symmetric 3×3 tensor.
    """
    T = jnp.zeros((3, 3))
    T = T.at[_I, _J].set(v)
    T = T.at[_J, _I].set(v)
    return T


def to_mandel(v_voigt: jnp.ndarray) -> jnp.ndarray:
    """Convert a Voigt vector to Mandel notation.

    Multiplies shear components (indices 3, 4, 5) by √2 so that the Euclidean
    inner product of two Mandel vectors equals the tensor double-contraction.

    Parameters
    ----------
    v_voigt : jnp.ndarray, shape (6,)
        Voigt vector [A11, A22, A33, A12, A13, A23].

    Returns
    -------
    jnp.ndarray, shape (6,)
        Mandel vector [A11, A22, A33, √2·A12, √2·A13, √2·A23].
    """
    return v_voigt * _MANDEL_FACTORS


def from_mandel(v_mandel: jnp.ndarray) -> jnp.ndarray:
    """Convert a Mandel vector back to Voigt notation.

    Parameters
    ----------
    v_mandel : jnp.ndarray, shape (6,)
        Mandel vector [A11, A22, A33, √2·A12, √2·A13, √2·A23].

    Returns
    -------
    jnp.ndarray, shape (6,)
        Voigt vector [A11, A22, A33, A12, A13, A23].
    """
    return v_mandel / _MANDEL_FACTORS
