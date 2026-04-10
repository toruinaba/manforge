"""Stress/strain operators in Voigt notation.

All functions accept and return stress in Voigt order.  For 3D solids:
  σ = [σ11, σ22, σ33, σ12, σ13, σ23]

Shear components are stored as physical stress components (NOT engineering
strain convention).  When inner-product equivalence with tensor double-
contraction is required (e.g. for norms), the Mandel transformation
(multiply shear by √2) is applied internally via ``utils.voigt``.

Each function accepts an optional ``ss`` (:class:`~manforge.core.stress_state.StressState`)
argument so that it works for any element dimensionality.  When ``ss`` is
omitted the 6-component 3D convention is used (backward-compatible).
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


def identity_voigt(ss=None) -> jnp.ndarray:
    """Second-order identity tensor δ_{ij} in Voigt notation.

    Parameters
    ----------
    ss : StressState, optional
        Dimensionality descriptor.  Defaults to ``SOLID_3D`` (6-component).

    Returns
    -------
    jnp.ndarray, shape (ntens,)
        [1, ..., 1, 0, ..., 0]  (ones for direct components, zeros for shear).
    """
    if ss is None:
        return jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    return ss.identity_jnp


def hydrostatic(stress: jnp.ndarray, ss=None) -> jnp.ndarray:
    """Hydrostatic pressure (mean normal stress).

    Parameters
    ----------
    stress : jnp.ndarray, shape (ntens,)
        Stress in Voigt notation.
    ss : StressState, optional
        Dimensionality descriptor.  Defaults to ``SOLID_3D``.

    Returns
    -------
    jnp.ndarray, scalar
        p = (sum of stored direct components) / ndi_phys

    Notes
    -----
    For plane stress (ndi=2, ndi_phys=3): divides by 3 because sigma_33=0
    is enforced externally and contributes zero to the trace.
    """
    if ss is None:
        return (stress[0] + stress[1] + stress[2]) / 3.0
    return jnp.sum(stress[: ss.ndi]) / ss.ndi_phys


def dev(stress: jnp.ndarray, ss=None) -> jnp.ndarray:
    """Deviatoric stress tensor.

    Parameters
    ----------
    stress : jnp.ndarray, shape (ntens,)
        Stress in Voigt notation.
    ss : StressState, optional
        Dimensionality descriptor.  Defaults to ``SOLID_3D``.

    Returns
    -------
    jnp.ndarray, shape (ntens,)
        s = σ - p·δ  where p = hydrostatic(σ)
    """
    p = hydrostatic(stress, ss)
    return stress - p * identity_voigt(ss)


def norm_mandel(t_voigt: jnp.ndarray, ss=None) -> jnp.ndarray:
    """Mandel norm of a symmetric second-order tensor.

    Converts to Mandel notation first so that the Euclidean norm equals
    the tensor norm  ‖T‖ = √(T:T).

    Parameters
    ----------
    t_voigt : jnp.ndarray, shape (ntens,)
        Tensor in Voigt notation.
    ss : StressState, optional
        Dimensionality descriptor.  Defaults to ``SOLID_3D``.

    Returns
    -------
    jnp.ndarray, scalar
        √(T:T)  =  ‖t_mandel‖₂
    """
    t_m = to_mandel(t_voigt, ss)
    return jnp.sqrt(jnp.dot(t_m, t_m))


def vonmises(stress: jnp.ndarray, ss=None) -> jnp.ndarray:
    """Von Mises equivalent stress.

    σ_vm = √(3/2 · s:s)  where s = dev(σ)

    Uses Mandel norm internally:
      s:s  = ‖s_mandel‖²
      σ_vm = √(3/2) · ‖s_mandel‖

    Parameters
    ----------
    stress : jnp.ndarray, shape (ntens,)
        Stress in Voigt notation.
    ss : StressState, optional
        Dimensionality descriptor.  Defaults to ``SOLID_3D``.

    Returns
    -------
    jnp.ndarray, scalar
        Von Mises equivalent stress.
    """
    s = dev(stress, ss)
    return jnp.sqrt(1.5) * norm_mandel(s, ss)


def I_vol_voigt(ss=None) -> jnp.ndarray:
    """Volumetric projection tensor P_vol in Voigt notation.

    P_vol = (1/ndi_phys) δ⊗δ

    Parameters
    ----------
    ss : StressState, optional
        Dimensionality descriptor.  Defaults to ``SOLID_3D``.

    Returns
    -------
    jnp.ndarray, shape (ntens, ntens)
    """
    delta = identity_voigt(ss)
    ndi_phys = 3 if ss is None else ss.ndi_phys
    return jnp.outer(delta, delta) / ndi_phys


def I_dev_voigt(ss=None) -> jnp.ndarray:
    """Deviatoric projection tensor P_dev in Voigt notation.

    P_dev = I_sym - P_vol

    where I_sym is the ntens×ntens symmetric identity (maps Voigt vector to
    itself for the normal components; shear components unchanged).

    Parameters
    ----------
    ss : StressState, optional
        Dimensionality descriptor.  Defaults to ``SOLID_3D``.

    Returns
    -------
    jnp.ndarray, shape (ntens, ntens)
    """
    ntens = 6 if ss is None else ss.ntens
    I_sym = jnp.eye(ntens)
    return I_sym - I_vol_voigt(ss)
