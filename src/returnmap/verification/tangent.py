"""Finite-difference check of the consistent tangent.

:func:`check_tangent` compares the tangent an integrator returns for one
stress update with central differences of the stress it returns — the
definition of the consistent tangent, ``dσ_{n+1} / dΔε``, taken literally.
It needs nothing but ``stress_update``, so it applies to any integrator: the
autodiff tangent, a hand-written ``user_defined_tangent``, or a Fortran UMAT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from returnmap._typing import FloatArray, StateDict, Stiffness
from returnmap.verification._measure import max_abs, rel_err

if TYPE_CHECKING:
    from returnmap.simulation.integrator import StressIntegrator

#: A tangent this many orders below the elastic stiffness is below what
#: central differences resolve: the differenced stresses carry round-off of
#: about 1e-16 of their size, which a strain step of 1e-7 turns into a
#: derivative of about 1e-12 of the stiffness.  A tangent that is zero — a
#: yield plateau, perfect plasticity in one dimension — therefore comes back
#: from the differences as noise of that size, and has nothing of its own to
#: be relative to.  Such a tangent is measured against the elastic stiffness.
ZERO_TANGENT = 1e-8


@dataclass
class TangentReport:
    """Result of :func:`check_tangent`.

    Attributes
    ----------
    ddsdde : ndarray, shape (ntens, ntens)
        The tangent the integrator returned.
    ddsdde_fd : ndarray, shape (ntens, ntens)
        Central differences of the integrator's stress — the reference.
    tol : float
        The tolerance the tangent was judged against.
    elastic_scale : float or None
        Largest entry of the elastic tangent at the start of the step — what
        a vanishing tangent is measured against (see :attr:`max_rel_err`).
        ``None`` if the integrator could not give one.
    """

    ddsdde: Stiffness
    ddsdde_fd: Stiffness
    tol: float
    elastic_scale: float | None = None

    @property
    def _scale(self) -> float:
        """What the difference is relative to."""
        own = max_abs(self.ddsdde_fd)
        if self.elastic_scale is not None and own < ZERO_TANGENT * self.elastic_scale:
            return self.elastic_scale
        return own

    @property
    def max_rel_err(self) -> float:
        """``max|ddsdde − ddsdde_fd| / max|ddsdde_fd|``.

        One case has no ``max|ddsdde_fd|`` to be relative to: a tangent that
        is zero (a yield plateau).  Central differences return noise there,
        and noise against noise is a 100 % error.  A reference tangent eight
        orders below the elastic stiffness is taken to be that case, and the
        difference is measured against the elastic stiffness instead.
        """
        with np.errstate(invalid="ignore"):
            diff = max_abs(
                np.asarray(self.ddsdde, dtype=float) - np.asarray(self.ddsdde_fd, dtype=float)
            )
        return rel_err(diff, self._scale)

    @property
    def rel_err_matrix(self) -> FloatArray:
        """Each entry's difference relative to ``max|ddsdde_fd|``.

        Its largest entry is :attr:`max_rel_err`; the position says which
        component of the tangent is off.
        """
        scale = self._scale
        diff = np.abs(
            np.asarray(self.ddsdde, dtype=float) - np.asarray(self.ddsdde_fd, dtype=float)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            return diff / scale

    @property
    def passed(self) -> bool:
        """The tangent is within ``tol`` of the central differences."""
        return self.max_rel_err <= self.tol

    def summary(self) -> str:
        """Two lines: the verdict, then the error and the entry it occurs at."""
        head = "tangent check: " + (
            "agrees with central differences" if self.passed
            else "differs from central differences"
        )
        m = self.rel_err_matrix
        where = ""
        if np.all(np.isfinite(m)):
            i, j = np.unravel_index(int(np.argmax(m)), m.shape)
            where = f", worst at [{i}][{j}]"
        mark = "ok  " if self.passed else "FAIL"
        return f"{head}\n  {mark} ddsdde: {self.max_rel_err:.3e} (tol {self.tol:.0e}{where})"

    def __str__(self) -> str:
        return self.summary()


def check_tangent(
    integrator: "StressIntegrator",
    strain_inc: FloatArray,
    stress_n: FloatArray,
    state_n: StateDict,
    *,
    eps: float = 1e-7,
    tol: float = 1e-5,
) -> TangentReport:
    """Compare an integrator's consistent tangent with central differences.

    The arguments after ``integrator`` are those of ``stress_update``, in the
    same order: the check is of the update ``stress_update(strain_inc,
    stress_n, state_n)``.

    Parameters
    ----------
    integrator : StressIntegrator
        Any integrator — :class:`~returnmap.simulation.PythonIntegrator`,
        :class:`~returnmap.simulation.PythonAnalyticalIntegrator` (checks a
        hand-written tangent), :class:`~returnmap.simulation.FortranIntegrator`.
    strain_inc : array-like, shape (ntens,)
        Strain increment of the step.
    stress_n : array-like, shape (ntens,)
        Stress at the start of the step.
    state_n : dict
        Internal state at the start of the step.
    eps : float, default 1e-7
        Finite-difference step in strain.
    tol : float, default 1e-5
        Pass threshold on ``max|ddsdde − ddsdde_fd| / max|ddsdde_fd|``.

    Returns
    -------
    TangentReport
        ``report.passed``, ``report.summary()``, and both matrices.

    Notes
    -----
    The step ``eps`` trades truncation error against round-off, and the
    tolerance has to leave room for both: the default pair suits a tangent
    of ordinary smoothness.  Across a yield point, or a branch a model
    switches on, the stress is not differentiable and central differences
    straddling it are not a reference.

    Examples
    --------
    ::

        report = check_tangent(PythonIntegrator(model), strain_inc, stress_n, state_n)
        assert report.passed, report.summary()
    """
    strain_inc = np.array(strain_inc, dtype=float)
    stress_n = np.array(stress_n, dtype=float)
    ntens = integrator.ntens

    ddsdde = np.array(integrator.stress_update(strain_inc, stress_n, state_n).ddsdde)
    ddsdde_fd = np.zeros((ntens, ntens))
    for j in range(ntens):
        e_j = np.zeros(ntens)
        e_j[j] = 1.0
        plus = integrator.stress_update(strain_inc + eps * e_j, stress_n, state_n).stress
        minus = integrator.stress_update(strain_inc - eps * e_j, stress_n, state_n).stress
        ddsdde_fd[:, j] = (np.array(plus) - np.array(minus)) / (2.0 * eps)
    try:
        elastic_scale: float | None = max_abs(integrator.elastic_tangent(stress_n, state_n))
    except Exception:
        # Not every integrator can say (a UMAT without an elastic-stiffness
        # hook); the tangent is then measured against itself only.
        elastic_scale = None
    return TangentReport(ddsdde=ddsdde, ddsdde_fd=ddsdde_fd, tol=tol, elastic_scale=elastic_scale)


__all__ = ["TangentReport", "check_tangent"]
