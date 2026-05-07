"""Fortran subroutine binding registry.

Provides a decorator and helpers that record the correspondence between
Python methods and Fortran subroutines.  The decorator attaches a function
attribute only — runtime behaviour is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class FortranBinding:
    """Metadata linking a Python method to a Fortran subroutine.

    Parameters
    ----------
    subroutine:
        Exact subroutine name as passed to :meth:`~manforge.simulation.integrator.FortranModule.call`.
    test:
        Pytest node id of the test that verifies this binding
        (e.g. ``"tests/fortran/test_j2_bindings.py::test_check_bindings_elastic_stiffness"``).
        Not executed at runtime — recorded as a machine-readable pointer.
    notes:
        Free-form annotation (e.g. sign convention differences).
    """

    subroutine: str
    test: str | None = None
    notes: str = ""


def verified_against_fortran(
    subroutine: str,
    *,
    test: str | None = None,
    notes: str = "",
) -> Callable:
    """Attach a :class:`FortranBinding` to a method.  Runtime behaviour unchanged.

    Usage::

        @verified_against_fortran(
            "j2_isotropic_3d_elastic_stiffness",
            test="tests/fortran/test_j2_bindings.py::test_check_bindings_elastic_stiffness",
        )
        def elastic_stiffness(self):
            ...
    """
    binding = FortranBinding(subroutine=subroutine, test=test, notes=notes)

    def decorator(fn: Callable) -> Callable:
        fn.__fortran_binding__ = binding
        return fn

    return decorator


def collect_bindings(cls) -> dict[str, FortranBinding]:
    """Collect methods decorated with :func:`verified_against_fortran`.

    Traverses ``cls.__mro__`` from base to derived so that overrides in
    subclasses shadow the parent's binding.  ``object`` and dunder names
    are excluded.

    Parameters
    ----------
    cls:
        The class to inspect (typically a concrete :class:`~manforge.core.material.MaterialModel` subclass).

    Returns
    -------
    dict[str, FortranBinding]
        Mapping from method name to its :class:`FortranBinding`.
    """
    bindings: dict[str, FortranBinding] = {}
    for klass in reversed(cls.__mro__):
        if klass is object:
            continue
        for name, obj in klass.__dict__.items():
            if name.startswith("__"):
                continue
            if hasattr(obj, "__fortran_binding__"):
                bindings[name] = obj.__fortran_binding__
    return bindings
