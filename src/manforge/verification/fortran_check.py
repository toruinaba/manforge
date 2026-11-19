"""Compare the model methods bound to Fortran subroutines with those subroutines.

A model records which of its methods has a Fortran counterpart with
:func:`~manforge.core.verified_against_fortran`.  :func:`check_bindings`
calls both sides of each binding with the arguments given and compares what
they return.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from manforge.core.material.fortran_binding import FortranBinding
from manforge.verification._measure import array_rel_err

if TYPE_CHECKING:
    from manforge.core.material import MaterialModel
    from manforge.simulation.integrator import FortranModule


@dataclass
class BindingCheck:
    """One bound method against its Fortran subroutine.

    Attributes
    ----------
    method : str
        Name of the model method.
    subroutine : str
        Name of the Fortran subroutine it is bound to.
    rel_err : float
        ``max|fortran − python| / max|python|`` over the returned array.
    tol : float
        The tolerance it is held to.
    """

    method: str
    subroutine: str
    rel_err: float
    tol: float

    @property
    def passed(self) -> bool:
        return self.rel_err <= self.tol


@dataclass
class BindingReport:
    """Result of :func:`check_bindings`.

    Attributes
    ----------
    checks : dict[str, BindingCheck]
        One entry per method in ``cases``, keyed by method name.
    not_checked : list[str]
        Bound methods of the model that ``cases`` did not cover.
    """

    checks: dict[str, BindingCheck]
    not_checked: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Every checked method agrees.  Says nothing about the ones not checked."""
        return all(c.passed for c in self.checks.values())

    @property
    def failed(self) -> list[BindingCheck]:
        """Methods outside tolerance, worst first."""
        bad = [c for c in self.checks.values() if not c.passed]
        return sorted(bad, key=lambda c: -c.rel_err)

    def summary(self) -> str:
        """Multi-line text: the verdict, then one line per checked method."""
        n_bad = len(self.failed)
        lines = [
            f"bindings: {len(self.checks)} checked — "
            + ("all agree" if self.passed else f"{n_bad} differ")
        ]
        for c in self.checks.values():
            mark = "ok  " if c.passed else "FAIL"
            lines.append(
                f"  {mark} {c.method} ↔ {c.subroutine}: {c.rel_err:.3e} (tol {c.tol:.0e})"
            )
        if self.not_checked:
            lines.append("  not checked: " + ", ".join(self.not_checked))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def check_bindings(
    model: "MaterialModel",
    fortran: "FortranModule",
    cases: dict[str, tuple[tuple, tuple]],
    *,
    tol: float = 1e-10,
) -> BindingReport:
    """Compare bound Python methods against their Fortran counterparts.

    Only methods listed in ``cases`` are checked; the model's other bound
    methods are listed under ``report.not_checked``.  Methods that return
    something other than an array (a ``dict``, say) are not suitable for
    this helper — test them individually.

    Parameters
    ----------
    model : MaterialModel
        The model whose bindings are checked.
    fortran : FortranModule
        The compiled module holding the subroutines.
    cases : dict
        ``{method_name: (py_args, fortran_args)}``.  ``py_args`` is passed to
        ``getattr(model, method_name)(*py_args)``, ``fortran_args`` to
        ``fortran.call(subroutine, *fortran_args)``.
    tol : float, default 1e-10
        Pass threshold on each method's
        ``max|fortran − python| / max|python|``.

    Returns
    -------
    BindingReport
        ``report.passed``, and ``report.summary()`` for the per-method table.

    Raises
    ------
    KeyError
        If a method in ``cases`` is not bound on the model.
    ValueError
        If ``cases`` is empty — nothing would have been checked — or the two
        sides return arrays of different sizes.
    """
    if not cases:
        raise ValueError("cases is empty — no binding was checked")
    bindings: dict[str, FortranBinding] = getattr(model, "_fortran_bindings", {})
    checks: dict[str, BindingCheck] = {}

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
        checks[method_name] = BindingCheck(
            method=method_name, subroutine=binding.subroutine,
            rel_err=array_rel_err(f_arr, py_arr), tol=tol,
        )

    not_checked = [name for name in bindings if name not in cases]
    return BindingReport(checks=checks, not_checked=not_checked)


__all__ = ["BindingCheck", "BindingReport", "check_bindings"]
