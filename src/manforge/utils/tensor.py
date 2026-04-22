"""Fourth-order tensor operations in Voigt (ntens×ntens matrix) representation."""

import autograd.numpy as anp


def ddot42(A: anp.ndarray, b: anp.ndarray) -> anp.ndarray:
    """Double contraction of a fourth-order tensor with a second-order tensor.

    Parameters
    ----------
    A : anp.ndarray, shape (ntens, ntens)
        Fourth-order tensor in Voigt notation.
    b : anp.ndarray, shape (ntens,)
        Second-order tensor in Voigt notation.

    Returns
    -------
    anp.ndarray, shape (ntens,)
        Result A : b in Voigt notation.
    """
    return A @ b


def ddot44(A: anp.ndarray, B: anp.ndarray) -> anp.ndarray:
    """Double contraction of two fourth-order tensors.

    Parameters
    ----------
    A : anp.ndarray, shape (ntens, ntens)
    B : anp.ndarray, shape (ntens, ntens)

    Returns
    -------
    anp.ndarray, shape (ntens, ntens)
        Result A : B in Voigt notation.
    """
    return A @ B


def symmetrize4(A: anp.ndarray) -> anp.ndarray:
    """Symmetrize a fourth-order tensor (major symmetry).

    Parameters
    ----------
    A : anp.ndarray, shape (ntens, ntens)

    Returns
    -------
    anp.ndarray, shape (ntens, ntens)
        (A + A^T) / 2
    """
    return (A + A.T) * 0.5
