"""StressIntegrator protocol and adapter implementations.

:class:`StressIntegrator` is a ``typing.Protocol`` — the contract drivers
check at construction (``dimension``, ``ntens``, ``raise_on_nonconverged``,
``initial_state``, ``increment``, ``elastic_tangent``, ``stress_update``).
Five adapters fulfil it:

:class:`PythonIntegrator`
    Wraps a :class:`~returnmap.core.MaterialModel` and uses the
    ``"auto"`` solver strategy (user-defined hook if present, else
    numerical Newton-Raphson).

:class:`PythonNumericalIntegrator`
    Same as ``PythonIntegrator`` but always uses ``"numerical_newton"``.

:class:`PythonAnalyticalIntegrator`
    Same as ``PythonIntegrator`` but always uses ``"user_defined"``
    (requires ``model.user_defined_return_mapping`` to be implemented).

:class:`FortranIntegrator`
    Wraps a :class:`~returnmap.simulation.FortranModule` and the four hook
    functions that map between Python state dicts and Fortran argument lists.

:class:`SubSteppingIntegrator`
    Wraps any of the above and retries a failed increment in 2, 4, 8, ...
    substeps.  Opt-in; an increment that already converged is passed through
    untouched.

Submodules: :mod:`base`, :mod:`python`, :mod:`fortran`, :mod:`substepping`.
"""

from returnmap.simulation.integrator.base import StressIntegrator
from returnmap.simulation.integrator.python import (
    PythonIntegrator,
    PythonNumericalIntegrator,
    PythonAnalyticalIntegrator,
)
from returnmap.simulation.integrator.fortran import FortranIntegrator
from returnmap.simulation.integrator.fortran_module import FortranModule
from returnmap.simulation.integrator.substepping import SubSteppingIntegrator

__all__ = [
    "StressIntegrator",
    "PythonIntegrator",
    "PythonNumericalIntegrator",
    "PythonAnalyticalIntegrator",
    "FortranIntegrator",
    "FortranModule",
    "SubSteppingIntegrator",
]
