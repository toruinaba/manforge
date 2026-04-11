"""Fourth-order tensor operations in Voigt (ntens×ntens matrix) representation."""

import jax.numpy as jnp


def ddot42(A: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Double contraction of a fourth-order tensor with a second-order tensor.

    Parameters
    ----------
    A : jnp.ndarray, shape (ntens, ntens)
        Fourth-order tensor in Voigt notation.
    b : jnp.ndarray, shape (ntens,)
        Second-order tensor in Voigt notation.

    Returns
    -------
    jnp.ndarray, shape (ntens,)
        Result A : b in Voigt notation.
    """
    return A @ b


def ddot44(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """Double contraction of two fourth-order tensors.

    Parameters
    ----------
    A : jnp.ndarray, shape (ntens, ntens)
    B : jnp.ndarray, shape (ntens, ntens)

    Returns
    -------
    jnp.ndarray, shape (ntens, ntens)
        Result A : B in Voigt notation.
    """
    return A @ B


def symmetrize4(A: jnp.ndarray) -> jnp.ndarray:
    """Symmetrize a fourth-order tensor (major symmetry).

    Parameters
    ----------
    A : jnp.ndarray, shape (ntens, ntens)

    Returns
    -------
    jnp.ndarray, shape (ntens, ntens)
        (A + A^T) / 2
    """
    return (A + A.T) * 0.5
