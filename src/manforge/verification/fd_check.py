"""Finite-difference verification of the consistent tangent.

Provides :func:`check_tangent`, a reusable utility that compares the AD
consistent tangent (returned by :func:`~manforge.core.return_mapping.return_mapping`)
against a central-difference approximation.  Useful for validating custom
:class:`~manforge.core.material.MaterialModel` implementations.
"""

from dataclasses import dataclass

import jax.numpy as jnp

from manforge.core.return_mapping import return_mapping


@dataclass
class TangentCheckResult:
    """Result of a tangent consistency check.

    Attributes
    ----------
    passed : bool
        ``True`` if ``max_rel_err < tol``.
    max_rel_err : float
        Maximum element-wise relative error between AD and FD tangents.
    ddsdde_ad : jnp.ndarray
        Consistent tangent from automatic differentiation, shape (ntens, ntens).
    ddsdde_fd : jnp.ndarray
        Consistent tangent from finite differences, shape (ntens, ntens).
    rel_err_matrix : jnp.ndarray
        Element-wise relative error matrix, shape (ntens, ntens).
        Useful for diagnosing which components of DDSDDE are incorrect.
    """

    passed: bool
    max_rel_err: float
    ddsdde_ad: jnp.ndarray
    ddsdde_fd: jnp.ndarray
    rel_err_matrix: jnp.ndarray


def _fd_tangent(
    model,
    strain_inc: jnp.ndarray,
    stress_n: jnp.ndarray,
    state_n: dict,
    params: dict,
    eps: float = 1e-7,
    method: str = "auto",
) -> jnp.ndarray:
    """Compute DDSDDE by central finite differences.

    Parameters
    ----------
    model : MaterialModel
        Constitutive model instance.
    strain_inc : jnp.ndarray, shape (ntens,)
        Strain increment Δε.
    stress_n : jnp.ndarray, shape (ntens,)
        Stress at the beginning of the increment σ_n.
    state_n : dict
        Internal state at the beginning of the increment.
    params : dict
        Material parameters.
    eps : float
        Perturbation size for central differences.
    method : str, optional
        Passed to :func:`~manforge.core.return_mapping.return_mapping`
        for the perturbed stress solves (default ``"auto"``).

    Returns
    -------
    jnp.ndarray, shape (ntens, ntens)
        Finite-difference approximation of dσ_{n+1}/dΔε.
    """
    ntens = model.ntens
    ddsdde_fd = jnp.zeros((ntens, ntens))

    for j in range(ntens):
        e_j = jnp.zeros(ntens).at[j].set(1.0)

        s_fwd, _, _ = return_mapping(
            model, strain_inc + eps * e_j, stress_n, state_n, params, method=method
        )
        s_bwd, _, _ = return_mapping(
            model, strain_inc - eps * e_j, stress_n, state_n, params, method=method
        )
        col = (s_fwd - s_bwd) / (2.0 * eps)
        ddsdde_fd = ddsdde_fd.at[:, j].set(col)

    return ddsdde_fd


def check_tangent(
    model,
    stress: jnp.ndarray,
    state: dict,
    params: dict,
    strain_inc: jnp.ndarray,
    eps: float = 1e-7,
    tol: float = 1e-5,
    denom_offset: float = 1e-2,
    method: str = "auto",
) -> TangentCheckResult:
    """Compare AD consistent tangent against central-difference approximation.

    Computes DDSDDE two ways and checks whether the maximum element-wise
    relative error is within tolerance.  Useful for validating that a
    custom :class:`~manforge.core.material.MaterialModel`'s
    ``yield_function`` and ``hardening_increment`` are correctly
    differentiable.

    Parameters
    ----------
    model : MaterialModel
        Constitutive model instance.
    stress : jnp.ndarray, shape (ntens,)
        Stress at the beginning of the increment σ_n.
    state : dict
        Internal state at the beginning of the increment.
    params : dict
        Material parameters.
    strain_inc : jnp.ndarray, shape (ntens,)
        Strain increment Δε.
    eps : float, optional
        Perturbation size for central differences (default 1e-7).
    tol : float, optional
        Pass/fail tolerance on max relative error (default 1e-5).
    denom_offset : float, optional
        Small offset added to ``|ddsdde_fd|`` before dividing to avoid
        near-zero division on small tangent entries (default 1e-2).
        Units match those of the stress/stiffness values.  Adjust when
        working with unit systems that produce very different magnitude
        stiffness entries (e.g. Pa instead of MPa).
    method : str, optional
        Passed to :func:`~manforge.core.return_mapping.return_mapping`
        for both the direct tangent call and the FD stress solves
        (default ``"auto"``).  Use ``"analytical"`` to verify a model's
        closed-form tangent against finite differences.

    Returns
    -------
    TangentCheckResult
        Structured result with pass/fail flag, error metrics, and both
        tangent matrices for diagnostic inspection.

    Notes
    -----
    The check may report spuriously large errors near the elastic-plastic
    transition boundary, where small perturbations in ``strain_inc`` can
    straddle the yield surface, causing one FD step to be elastic and the
    other plastic.  Use strain increments that are clearly inside one
    regime to avoid this.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import manforge
    >>> from manforge.models.j2_isotropic import J2Isotropic3D
    >>> from manforge.verification.fd_check import check_tangent
    >>> model = J2Isotropic3D()
    >>> params = {"E": 210000.0, "nu": 0.3, "sigma_y0": 250.0, "H": 1000.0}
    >>> result = check_tangent(
    ...     model,
    ...     jnp.zeros(6),
    ...     model.initial_state(),
    ...     params,
    ...     jnp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ... )
    >>> result.passed
    True
    """
    _, _, ddsdde_ad = return_mapping(model, strain_inc, stress, state, params, method=method)
    ddsdde_fd = _fd_tangent(model, strain_inc, stress, state, params, eps=eps, method=method)

    rel_err_matrix = jnp.abs(ddsdde_ad - ddsdde_fd) / (
        jnp.abs(ddsdde_fd) + denom_offset
    )
    max_rel_err = float(jnp.max(rel_err_matrix))

    return TangentCheckResult(
        passed=max_rel_err < tol,
        max_rel_err=max_rel_err,
        ddsdde_ad=ddsdde_ad,
        ddsdde_fd=ddsdde_fd,
        rel_err_matrix=rel_err_matrix,
    )
