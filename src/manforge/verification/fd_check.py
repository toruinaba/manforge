"""Finite-difference verification of the consistent tangent."""

from dataclasses import dataclass

import numpy as np

from manforge.core.stress_update import stress_update


@dataclass
class TangentCheckResult:
    """Result of a tangent consistency check."""

    passed: bool
    max_rel_err: float
    ddsdde_ad: np.ndarray
    ddsdde_fd: np.ndarray
    rel_err_matrix: np.ndarray


def _fd_tangent(model, strain_inc, stress_n, state_n, eps=1e-7, method="auto"):
    """Compute DDSDDE by central finite differences."""
    ntens = model.ntens
    ddsdde_fd = np.zeros((ntens, ntens))
    strain_inc = np.array(strain_inc)
    stress_n = np.array(stress_n)

    for j in range(ntens):
        e_j = np.zeros(ntens)
        e_j[j] = 1.0
        col = (
            np.array(stress_update(model, strain_inc + eps * e_j, stress_n, state_n, method=method).stress)
            - np.array(stress_update(model, strain_inc - eps * e_j, stress_n, state_n, method=method).stress)
        ) / (2.0 * eps)
        ddsdde_fd[:, j] = col

    return ddsdde_fd


def check_tangent(model, stress, state, strain_inc, eps=1e-7, tol=1e-5,
                  denom_offset=1e-2, method="auto") -> TangentCheckResult:
    """Compare AD consistent tangent against central-difference approximation."""
    ddsdde_ad = np.array(stress_update(model, strain_inc, stress, state, method=method).ddsdde)
    ddsdde_fd = _fd_tangent(model, strain_inc, stress, state, eps=eps, method=method)

    rel_err_matrix = np.abs(ddsdde_ad - ddsdde_fd) / (np.abs(ddsdde_fd) + denom_offset)
    max_rel_err = float(np.max(rel_err_matrix))

    return TangentCheckResult(
        passed=max_rel_err < tol,
        max_rel_err=max_rel_err,
        ddsdde_ad=ddsdde_ad,
        ddsdde_fd=ddsdde_fd,
        rel_err_matrix=rel_err_matrix,
    )
