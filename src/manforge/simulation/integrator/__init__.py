"""StressIntegrator protocol and adapter implementations.

Four adapters implement the :class:`StressIntegrator` protocol:

:class:`PythonIntegrator`
    Wraps a :class:`~manforge.core.material.MaterialModel` and uses the
    ``"auto"`` solver strategy (user-defined hook if present, else
    numerical Newton-Raphson).

:class:`PythonNumericalIntegrator`
    Same as ``PythonIntegrator`` but always uses ``"numerical_newton"``.

:class:`PythonAnalyticalIntegrator`
    Same as ``PythonIntegrator`` but always uses ``"user_defined"``
    (requires ``model.user_defined_return_mapping`` to be implemented).

:class:`FortranIntegrator`
    Wraps a :class:`~manforge.verification.FortranModule` and the four hook
    functions that map between Python state dicts and Fortran argument lists.

Submodules: :mod:`base`, :mod:`python`, :mod:`fortran`.
"""

from manforge.simulation.integrator.base import (
    StressIntegrator,
    _PythonIntegratorBase,
)
from manforge.simulation.integrator.python import (
    PythonIntegrator,
    PythonNumericalIntegrator,
    PythonAnalyticalIntegrator,
)
from manforge.simulation.integrator.fortran import FortranIntegrator

__all__ = [
    "StressIntegrator",
    "_PythonIntegratorBase",
    "PythonIntegrator",
    "PythonNumericalIntegrator",
    "PythonAnalyticalIntegrator",
    "FortranIntegrator",
]
