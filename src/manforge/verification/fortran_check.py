"""Fortran binding runtime checker.

Provides :func:`check_bindings` which compares registered Python methods
against their Fortran counterparts using a :class:`~manforge.simulation.integrator.FortranModule`
instance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from manforge.core.material.fortran_binding import FortranBinding

if TYPE_CHECKING:
    from manforge.core.material import MaterialModel
    from manforge.simulation.integrator import FortranModule


def check_bindings(
    model: "MaterialModel",
    fortran: "FortranModule",
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
        An instantiated :class:`~manforge.core.material.MaterialModel`.
    fortran:
        A :class:`~manforge.simulation.integrator.FortranModule` instance (module must be importable).
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

        if py_arr.size != f_arr.size:
            raise ValueError(
                f"check_bindings: shape mismatch for '{method_name}': "
                f"Python returned size {py_arr.size}, Fortran returned size {f_arr.size}"
            )

        max_rel_err = float(np.max(np.abs(py_arr - f_arr) / (np.abs(f_arr) + 1e-14)))
        results[method_name] = (max_rel_err < rtol, max_rel_err)

    return results
