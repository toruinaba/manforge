"""Stress/strain operators in Voigt notation.

All functions accept and return stress in Voigt order:
  σ = [σ11, σ22, σ33, σ12, σ13, σ23]

Shear components are stored as physical stress components (NOT engineering
strain convention).  When inner-product equivalence with tensor double-
contraction is required (e.g. for norms), the Mandel transformation
(multiply shear by √2) is applied internally via ``utils.voigt``.
"""

import jax.numpy as jnp

from manforge.utils.voigt import to_mandel

__all__ = [
    "identity_voigt",
    "hydrostatic",
    "dev",
    "vonmises",
    "norm_mandel",
    "I_dev_voigt",
    "I_vol_voigt",
]


def identity_voigt() -> jnp.ndarray:
    """Second-order identity tensor δ_{ij} in Voigt notation.

    Returns
    -------
    jnp.ndarray, shape (6,)
        [1, 1, 1, 0, 0, 0]
    """
    return jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def hydrostatic(stress: jnp.ndarray) -> jnp.ndarray:
    """Hydrostatic pressure (mean normal stress).

    Parameters
    ----------
    stress : jnp.ndarray, shape (6,)
        Stress in Voigt notation.

    Returns
    -------
    jnp.ndarray, scalar
        p = (σ11 + σ22 + σ33) / 3
    """
    return (stress[0] + stress[1] + stress[2]) / 3.0


def dev(stress: jnp.ndarray) -> jnp.ndarray:
    """Deviatoric stress tensor.

    Parameters
    ----------
    stress : jnp.ndarray, shape (6,)
        Stress in Voigt notation.

    Returns
    -------
    jnp.ndarray, shape (6,)
        s = σ - p·δ  where p = hydrostatic(σ)
    """
    p = hydrostatic(stress)
    return stress - p * identity_voigt()


def norm_mandel(t_voigt: jnp.ndarray) -> jnp.ndarray:
    """Mandel norm of a symmetric second-order tensor.

    Converts to Mandel notation first so that the Euclidean norm equals
    the tensor norm  ‖T‖ = √(T:T).

    Parameters
    ----------
    t_voigt : jnp.ndarray, shape (6,)
        Tensor in Voigt notation.

    Returns
    -------
    jnp.ndarray, scalar
        √(T:T)  =  ‖t_mandel‖₂
    """
    t_m = to_mandel(t_voigt)
    return jnp.sqrt(jnp.dot(t_m, t_m))


def vonmises(stress: jnp.ndarray) -> jnp.ndarray:
    """Von Mises equivalent stress.

    σ_vm = √(3/2 · s:s)  where s = dev(σ)

    Uses Mandel norm internally:
      s:s  = ‖s_mandel‖²
      σ_vm = √(3/2) · ‖s_mandel‖

    Parameters
    ----------
    stress : jnp.ndarray, shape (6,)
        Stress in Voigt notation.

    Returns
    -------
    jnp.ndarray, scalar
        Von Mises equivalent stress.
    """
    s = dev(stress)
    return jnp.sqrt(1.5) * norm_mandel(s)


def I_vol_voigt() -> jnp.ndarray:
    """Volumetric projection tensor P_vol in Voigt notation.

    P_vol = (1/3) δ⊗δ

    Returns
    -------
    jnp.ndarray, shape (6, 6)
    """
    delta = identity_voigt()
    return jnp.outer(delta, delta) / 3.0


def I_dev_voigt() -> jnp.ndarray:
    """Deviatoric projection tensor P_dev in Voigt notation.

    P_dev = I_sym - P_vol

    where I_sym is the 6×6 symmetric identity (maps Voigt vector to itself
    for the normal components; shear components unchanged).

    Returns
    -------
    jnp.ndarray, shape (6, 6)
    """
    # I_sym in Voigt: identity matrix for all 6 components
    I_sym = jnp.eye(6)
    return I_sym - I_vol_voigt()
