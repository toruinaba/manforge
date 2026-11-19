"""Concrete Python integrator classes."""

from __future__ import annotations

from typing import ClassVar, Literal

from manforge.simulation.integrator.base import _PythonIntegratorBase


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
