"""Finite-difference verification of the consistent tangent."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manforge._typing import FloatArray, StateDict, Stiffness


@dataclass
class TangentCheckResult:
    """Result of a tangent consistency check."""

    passed: bool
    max_rel_err: float
    ddsdde_ad: Stiffness
    ddsdde_fd: Stiffness
    rel_err_matrix: FloatArray


class TangentChecker:
    """Compare AD consistent tangent against central-difference approximation.

    Parallel class-based API to :class:`~manforge.verification.JacobianChecker`.

    Parameters
    ----------
    integrator : StressIntegrator
        Constitutive integrator — use
        :class:`~manforge.simulation.integrator.PythonIntegrator` (auto),
        :class:`~manforge.simulation.integrator.PythonNumericalIntegrator`, or
        :class:`~manforge.simulation.integrator.PythonAnalyticalIntegrator`.
    eps : float, optional
        Finite-difference step size (default 1e-7).
    tol : float, optional
        Relative-error tolerance for the pass/fail check (default 1e-5).
    denom_offset : float, optional
        Offset added to the denominator to avoid division by zero (default 1e-2).

    Examples
    --------
    >>> checker = TangentChecker(PythonIntegrator(model))
    >>> result = checker.check(stress_n, state_n, strain_inc)
    >>> assert result.passed
    """

    def __init__(self, integrator: object, *, eps: float = 1e-7, tol: float = 1e-5,
                 denom_offset: float = 1e-2) -> None:
        self.integrator = integrator
        self.eps = eps
        self.tol = tol
        self.denom_offset = denom_offset

    def check(self, stress: FloatArray, state: StateDict, strain_inc: FloatArray) -> TangentCheckResult:
        """Run the FD vs AD tangent comparison.

        Parameters
        ----------
        stress : array-like, shape (ntens,)
            Stress state at which to evaluate the tangent.
        state : dict
            Internal state at which to evaluate the tangent.
        strain_inc : array-like, shape (ntens,)
            Strain increment for the evaluation point.
        """
        ddsdde_ad = np.array(self.integrator.stress_update(strain_inc, stress, state).ddsdde)
        ddsdde_fd = self._fd_tangent(strain_inc, stress, state)
        rel_err_matrix = np.abs(ddsdde_ad - ddsdde_fd) / (np.abs(ddsdde_fd) + self.denom_offset)
        max_rel_err = float(np.max(rel_err_matrix))
        return TangentCheckResult(
            passed=max_rel_err < self.tol,
            max_rel_err=max_rel_err,
            ddsdde_ad=ddsdde_ad,
            ddsdde_fd=ddsdde_fd,
            rel_err_matrix=rel_err_matrix,
        )

    def _fd_tangent(self, strain_inc: FloatArray, stress_n: FloatArray, state_n: StateDict) -> Stiffness:
        ntens = self.integrator.ntens
        ddsdde_fd = np.zeros((ntens, ntens))
        strain_inc = np.array(strain_inc)
        stress_n = np.array(stress_n)
        for j in range(ntens):
            e_j = np.zeros(ntens)
            e_j[j] = 1.0
            col = (
                np.array(self.integrator.stress_update(
                    strain_inc + self.eps * e_j, stress_n, state_n).stress)
                - np.array(self.integrator.stress_update(
                    strain_inc - self.eps * e_j, stress_n, state_n).stress)
            ) / (2.0 * self.eps)
            ddsdde_fd[:, j] = col
        return ddsdde_fd


def check_tangent(
    integrator: object,
    stress: FloatArray,
    state: StateDict,
    strain_inc: FloatArray,
    eps: float = 1e-7,
    tol: float = 1e-5,
    denom_offset: float = 1e-2,
) -> TangentCheckResult:
    """Compare AD consistent tangent against central-difference approximation.

    Backward-compatible function form — delegates to :class:`TangentChecker`.

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
    return TangentChecker(integrator, eps=eps, tol=tol, denom_offset=denom_offset).check(
        stress, state, strain_inc
    )
