"""Stress/strain operators in Voigt notation.

All functions accept and return stress in Voigt order.  For 3D solids:
  σ = [σ11, σ22, σ33, σ12, σ13, σ23]

Shear components are stored as physical stress components (NOT engineering
strain convention).  When inner-product equivalence with tensor double-
contraction is required (e.g. for norms), the Mandel transformation
(multiply shear by √2) is applied internally via ``utils.voigt``.

Each function accepts an optional ``ss`` (:class:`~manforge.core.dimension.StressState`)
argument so that it works for any element dimensionality.  When ``ss`` is
omitted the 6-component 3D convention is used (backward-compatible).
"""

import autograd.numpy as anp

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


def identity_voigt(ss=None) -> anp.ndarray:
    """Second-order identity tensor δ_{ij} in Voigt notation."""
    if ss is None:
        return anp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    return ss.identity_np


def hydrostatic(stress: anp.ndarray, ss=None) -> anp.ndarray:
    """Hydrostatic pressure (mean normal stress)."""
    if ss is None:
        return (stress[0] + stress[1] + stress[2]) / 3.0
    return anp.sum(stress[: ss.ndi]) / ss.ndi_phys


def dev(stress: anp.ndarray, ss=None) -> anp.ndarray:
    """Deviatoric stress tensor."""
    p = hydrostatic(stress, ss)
    return stress - p * identity_voigt(ss)


def norm_mandel(t_voigt: anp.ndarray, ss=None) -> anp.ndarray:
    """Mandel norm of a symmetric second-order tensor."""
    t_m = to_mandel(t_voigt, ss)
    return anp.sqrt(anp.dot(t_m, t_m))


def vonmises(stress: anp.ndarray, ss=None) -> anp.ndarray:
    """Von Mises equivalent stress."""
    s = dev(stress, ss)
    p = hydrostatic(stress, ss)
    ndi_phys = 3 if ss is None else ss.ndi_phys
    ndi_stored = 3 if ss is None else ss.ndi
    n_missing = ndi_phys - ndi_stored
    s_m = to_mandel(s, ss)
    sq_norm = anp.dot(s_m, s_m) + n_missing * p ** 2
    return anp.sqrt(1.5 * sq_norm)


def I_vol_voigt(ss=None) -> anp.ndarray:
    """Volumetric projection tensor P_vol in Voigt notation."""
    delta = identity_voigt(ss)
    ndi_phys = 3 if ss is None else ss.ndi_phys
    return anp.outer(delta, delta) / ndi_phys


def I_dev_voigt(ss=None) -> anp.ndarray:
    """Deviatoric projection tensor P_dev in Voigt notation."""
    ntens = 6 if ss is None else ss.ntens
    I_sym = anp.eye(ntens)
    return I_sym - I_vol_voigt(ss)
