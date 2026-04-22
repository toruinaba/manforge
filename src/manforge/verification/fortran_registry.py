"""Fortran subroutine binding registry.

Provides a decorator and helpers that record the correspondence between
Python methods and Fortran subroutines.  The decorator attaches a function
attribute only — runtime behaviour is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class FortranBinding:
    """Metadata linking a Python method to a Fortran subroutine.

    Parameters
    ----------
    subroutine:
        Exact subroutine name as passed to :meth:`FortranUMAT.call`.
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
        The class to inspect (typically a concrete :class:`MaterialModel` subclass).

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


def check_bindings(
    model,
    fortran,
    cases: dict[str, tuple[tuple, tuple]],
    *,
    rtol: float = 1e-10,
) -> dict[str, tuple[bool, float]]:
    """Compare registered Python methods against their Fortran counterparts.

    Only methods listed in *cases* are checked.  Methods that return
    non-array types (e.g. ``dict``) are not suitable for this helper — test
    them individually.

    Parameters
    ----------
    model:
        An instantiated :class:`MaterialModel`.
    fortran:
        A :class:`FortranUMAT` instance (module must be importable).
    cases:
        ``{method_name: (py_args, fortran_args)}``.
        *py_args* is passed to ``getattr(model, method_name)(*py_args)``.
        *fortran_args* is passed to ``fortran.call(subroutine, *fortran_args)``.
    rtol:
        Relative tolerance threshold.  A result is ``ok`` when
        ``max_rel_err < rtol``.

    Returns
    -------
    dict[str, tuple[bool, float]]
        ``{method_name: (ok, max_rel_err)}``.

    Raises
    ------
    KeyError
        If a method in *cases* is not in ``model._fortran_bindings``.
    """
    bindings: dict[str, FortranBinding] = getattr(model, "_fortran_bindings", {})
    results: dict[str, tuple[bool, float]] = {}

    for method_name, (py_args, fortran_args) in cases.items():
        binding = bindings[method_name]

        py_out = getattr(model, method_name)(*py_args)
        f_out = fortran.call(binding.subroutine, *fortran_args)

        py_arr = np.asarray(py_out, dtype=float).ravel()
        f_arr = np.asarray(f_out, dtype=float).ravel()

        max_rel_err = float(np.max(np.abs(py_arr - f_arr) / (np.abs(f_arr) + 1e-14)))
        results[method_name] = (max_rel_err < rtol, max_rel_err)

    return results
