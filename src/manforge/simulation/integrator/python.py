"""Concrete Python integrator classes."""

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
    """

    _method = "auto"


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

    _method = "numerical_newton"


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

    _method = "user_defined"
