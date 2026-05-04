"""Finite-difference verification of the consistent tangent."""

from dataclasses import dataclass

import numpy as np


@dataclass
class TangentCheckResult:
    """Result of a tangent consistency check."""

    passed: bool
    max_rel_err: float
    ddsdde_ad: np.ndarray
    ddsdde_fd: np.ndarray
    rel_err_matrix: np.ndarray


def _fd_tangent(integrator, strain_inc, stress_n, state_n, eps=1e-7):
    """Compute DDSDDE by central finite differences."""
    ntens = integrator.ntens
    ddsdde_fd = np.zeros((ntens, ntens))
    strain_inc = np.array(strain_inc)
    stress_n = np.array(stress_n)

    for j in range(ntens):
        e_j = np.zeros(ntens)
        e_j[j] = 1.0
        col = (
            np.array(integrator.stress_update(strain_inc + eps * e_j, stress_n, state_n).stress)
            - np.array(integrator.stress_update(strain_inc - eps * e_j, stress_n, state_n).stress)
        ) / (2.0 * eps)
        ddsdde_fd[:, j] = col

    return ddsdde_fd


def check_tangent(integrator, stress, state, strain_inc, eps=1e-7, tol=1e-5,
                  denom_offset=1e-2) -> TangentCheckResult:
    """Compare AD consistent tangent against central-difference approximation.

    Parameters
    ----------
    integrator : StressIntegrator
        Constitutive integrator — use
        :class:`~manforge.simulation.integrator.PythonIntegrator` (auto),
        :class:`~manforge.simulation.integrator.PythonNumericalIntegrator`, or
        :class:`~manforge.simulation.integrator.PythonAnalyticalIntegrator`.
    stress : array-like, shape (ntens,)
        Stress state at which to evaluate the tangent.
    state : dict
        Internal state at which to evaluate the tangent.
    strain_inc : array-like, shape (ntens,)
        Strain increment for the evaluation point.
    eps : float, optional
        Finite-difference step size (default 1e-7).
    tol : float, optional
        Relative-error tolerance for the pass/fail check (default 1e-5).
    denom_offset : float, optional
        Offset added to the denominator to avoid division by zero (default 1e-2).
    """
    ddsdde_ad = np.array(integrator.stress_update(strain_inc, stress, state).ddsdde)
    ddsdde_fd = _fd_tangent(integrator, strain_inc, stress, state, eps=eps)

    rel_err_matrix = np.abs(ddsdde_ad - ddsdde_fd) / (np.abs(ddsdde_fd) + denom_offset)
    max_rel_err = float(np.max(rel_err_matrix))

    return TangentCheckResult(
        passed=max_rel_err < tol,
        max_rel_err=max_rel_err,
        ddsdde_ad=ddsdde_ad,
        ddsdde_fd=ddsdde_fd,
        rel_err_matrix=rel_err_matrix,
    )
