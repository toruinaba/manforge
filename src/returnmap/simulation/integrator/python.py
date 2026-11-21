"""Concrete Python integrator classes."""

from __future__ import annotations

from typing import ClassVar, Literal

from returnmap.simulation.integrator.base import _PythonIntegratorBase


class PythonIntegrator(_PythonIntegratorBase):
    """Wraps a MaterialModel using the ``"auto"`` solver strategy.

    Uses ``model.user_defined_return_mapping`` when available, otherwise
    falls back to numerical Newton-Raphson.

    Parameters
    ----------
    model : MaterialModel
    max_iter : int, optional
        Maximum Newton-Raphson iterations (default 50).
    tol : float, optional
        NR convergence tolerance (default 1e-10).
    raise_on_nonconverged : bool, optional
        Raise ``RuntimeError`` if NR does not converge (default ``True``).
    nr_jacobian : {"exact", "broyden"}, optional
        ``"exact"`` (default) re-evaluates the autodiff Jacobian at every NR
        iteration.  ``"broyden"`` evaluates it once per step and applies
        rank-1 updates, falling back to an exact re-evaluation when the
        residual stops decreasing; several times faster on models without a
        hand-written solver, at the cost of one or two more iterations.  The
        consistent tangent is exact either way.
    nr_restart : bool, optional
        ``True`` (default): a step the numerical Newton-Raphson fails on
        from the trial point (Δλ = 0) is solved once more, from a plastic
        estimate of Δλ; ``StressUpdateResult.n_restarts`` says so.
        ``False``: no second attempt — the solver as it was before 0.3.2,
        for comparing the two.  Steps the first attempt solves are identical
        either way, and ``user_defined_return_mapping`` is never restarted.
    """

    _method: ClassVar[Literal["auto", "numerical_newton", "user_defined"]] = "auto"


class PythonNumericalIntegrator(_PythonIntegratorBase):
    """Wraps a MaterialModel always using numerical Newton-Raphson.

    Parameters
    ----------
    model : MaterialModel
    max_iter : int, optional
        Maximum Newton-Raphson iterations (default 50).
    tol : float, optional
        NR convergence tolerance (default 1e-10).
    raise_on_nonconverged : bool, optional
        Raise ``RuntimeError`` if NR does not converge (default ``True``).
    nr_jacobian : {"exact", "broyden"}, optional
        See :class:`PythonIntegrator`.
    nr_restart : bool, optional
        Whether a step that fails from the trial point is solved once more
        from a plastic estimate of Δλ (default ``True``).  See
        :class:`PythonIntegrator`.
    """

    _method: ClassVar[Literal["auto", "numerical_newton", "user_defined"]] = "numerical_newton"


class PythonAnalyticalIntegrator(_PythonIntegratorBase):
    """Wraps a MaterialModel always using the user-defined analytical solver.

    Requires ``model.user_defined_return_mapping`` to be implemented.

    Parameters
    ----------
    model : MaterialModel
    max_iter : int, optional
        Maximum Newton-Raphson iterations (default 50, ignored for analytical solver).
    tol : float, optional
        NR convergence tolerance (default 1e-10, ignored for analytical solver).
    raise_on_nonconverged : bool, optional
        Raise ``RuntimeError`` if NR does not converge (default ``True``).
    """

    _method: ClassVar[Literal["auto", "numerical_newton", "user_defined"]] = "user_defined"
