"""Check the solutions of a hand-written return mapping against the framework residual.

:func:`check_solution` substitutes what a hand-written return mapping
(``user_defined_return_mapping``) reports — stress, Δλ, state — into the
residual the autodiff path solves for the same step, and reports row by row
whether it is a root.  The residual itself is never declared by hand: only
the roots are compared.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from manforge.core.result import StressUpdateResult
from manforge.simulation.jacobian import increment_of, residual_at
from manforge.verification._measure import as_error

if TYPE_CHECKING:
    from manforge.core.material import MaterialModel


# check_solution allows a row this many times the residual norm the solver
# reported: a row can carry the whole of that norm, and the two residuals are
# evaluated by different code, so the bound itself needs room.
_SLACK_FACTOR = 10.0


@dataclass
class SolutionRowCheck:
    """One line of a :class:`SolutionReport`.

    Attributes
    ----------
    name : str
        A residual row (``"stress"``, the Δλ row, an implicit state) or an
        explicit state.
    kind : {"residual", "state"}
        ``"residual"``: the framework residual row evaluated at the reported
        solution.  ``"state"``: an explicit state as reported, against the
        framework's ``update_state`` at the same point.
    worst_err : float
        Largest error over the checked steps (see
        :func:`check_solution` for the measure).
    worst_at : int
        Index, in the checked sequence, of the step where it occurred.
    tol : float
        Tolerance the line is held to.
    """

    name: str
    kind: str
    worst_err: float
    worst_at: int
    tol: float

    @property
    def passed(self) -> bool:
        return self.worst_err <= self.tol


@dataclass
class SolutionReport:
    """Result of :func:`check_solution`.

    Attributes
    ----------
    rows : dict[str, SolutionRowCheck]
        One entry per residual row and per exercised explicit state.
    not_exercised : list[str]
        Explicit states that never moved in the checked steps, so their
        update was not tested by this history.
    n_points : int
        Number of plastic steps checked.
    n_substepped : int
        Plastic steps left out because the integrator divided the increment
        (``n_substeps > 1``): such a result is the end of a chain of smaller
        steps, not a root of the residual of the undivided one.
    """

    rows: dict[str, SolutionRowCheck]
    not_exercised: list[str] = field(default_factory=list)
    n_points: int = 0
    n_substepped: int = 0

    @property
    def passed(self) -> bool:
        """Every row and every exercised state within tolerance."""
        return all(r.passed for r in self.rows.values())

    @property
    def failed(self) -> list[SolutionRowCheck]:
        """Lines outside tolerance, worst first."""
        bad = [r for r in self.rows.values() if not r.passed]
        return sorted(bad, key=lambda r: -r.worst_err)

    def summary(self) -> str:
        """Multi-line text: one line per residual row, then the explicit states."""
        n_bad = len(self.failed)
        lines = [
            f"solution check: {self.n_points} plastic steps — "
            + ("every row satisfied" if self.passed
               else f"{n_bad} {'row' if n_bad == 1 else 'rows'} off")
        ]
        for r in self.rows.values():
            mark = "ok  " if r.passed else "FAIL"
            label = f"[{r.name}]" if r.kind == "residual" else f"{r.name} (explicit state)"
            lines.append(
                f"  {mark} {label}: {r.worst_err:.3e} "
                f"(tol {r.tol:.0e}, worst at point {r.worst_at})"
            )
        if self.not_exercised:
            lines.append("  not exercised: " + ", ".join(self.not_exercised))
        if self.n_substepped:
            lines.append(f"  left out: {self.n_substepped} sub-stepped steps")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def check_solution(
    model: "MaterialModel",
    results: Iterable[StressUpdateResult],
    *,
    tol: float = 1e-8,
) -> SolutionReport:
    """Substitute reported solutions into the framework residual, row by row.

    For a model with a hand-written return mapping
    (``user_defined_return_mapping`` — a closed form, or a hand NR on its
    own residual), run the history through
    :class:`~manforge.simulation.PythonAnalyticalIntegrator` and pass the
    step results here.  Each plastic step is judged on its own: is the
    reported solution a root of the residual the autodiff path solves for
    that step, starting from the same state?

    A hand residual row that differs from the framework's leaves exactly
    that row unsatisfied, so the report says *which* row is wrong.
    Comparing two trajectories only says that the stress drifted apart,
    and after the first disagreement every later step differs.  Only roots
    are compared: a hand residual scaled differently, with the same
    solution, passes.

    The measure, per plastic step:

    * **Residual rows** — ``max|R_row| / (scale_row + slack / tol)``.
      ``scale_row`` is the size of the terms the row is made of,
      ``max (|∂R/∂x| · |x|)`` over the row (a componentwise backward
      error, unchanged if the row is rescaled).  ``slack`` is ten times
      the residual norm the solver itself reported for the step
      (``residual_history[-1]``), so a solution is not faulted for the
      tolerance its own solver stopped at — and an error smaller than
      that cannot be told from it.  It is zero unless the solver iterated
      (a history of at least two entries): a closed form has no
      tolerance to allow for.
    * **Explicit states** — the reported value against the framework's
      ``update_state`` at the reported point, relative to the largest
      magnitude the state reaches in the checked steps.  A state that
      equals its value at the start of the step in every checked step,
      as reported and as derived, is listed under ``not_exercised``
      instead: its update was never used.

    What is not checked.  A sub-stepped result (``n_substeps > 1``) is
    left out and counted in ``n_substepped``: it is the end of a chain of
    smaller steps, not a root of the undivided increment's residual.

    Parameters
    ----------
    model : MaterialModel
        The model whose framework residual the solutions are substituted into.
    results : iterable of StressUpdateResult
        Typically ``driver.run(load).step_results`` of a Python
        integrator.  Only plastic steps are used.
    tol : float, default 1e-8
        Tolerance on every line of the report.

    Returns
    -------
    SolutionReport
        ``report.passed``, and ``report.summary()`` for the per-row table.

    Raises
    ------
    ValueError
        If nothing would have been checked (no plastic step, or every one
        sub-stepped); if a result carries no increment (a Fortran UMAT
        result — compare that with
        :func:`~manforge.verification.crosscheck`); or if a plastic
        result did not converge (an iterate is not a solution).
    """
    res_rows: dict[str, list[tuple[int, float, float, float]]] = {}
    exp_rows: dict[str, list[tuple[int, float, float, float]]] = {}
    n_points = 0
    n_substepped = 0
    for i, result in enumerate(results):
        inc = increment_of(result)
        if result.is_plastic is not True:
            continue
        if not result.converged:
            raise ValueError(
                f"result {i} did not converge: it is a Newton iterate, not "
                "a solution, and cannot be checked as one"
            )
        if result.n_substeps > 1:
            n_substepped += 1
            continue
        point = residual_at(model, result, inc)
        layout = point.layout
        n_points += 1
        scale = np.abs(point.jacobian) @ np.abs(point.x)
        history = result.residual_history
        slack = _SLACK_FACTOR * float(history[-1]) if len(history) >= 2 else 0.0
        for unknown in ("stress", "dlambda", *layout.implicit_keys):
            sl = layout.slot_slice(unknown)
            res_rows.setdefault(layout.residual_name_for(unknown), []).append((
                i,
                float(np.max(np.abs(point.residual[sl]))),
                float(np.max(scale[sl])),
                slack,
            ))
        for key in layout.explicit_keys:
            reported = np.asarray(result.state[key], dtype=float)
            derived = np.asarray(point.state[key], dtype=float)
            start = np.asarray(inc.state_n[key], dtype=float)
            exp_rows.setdefault(key, []).append((
                i,
                float(np.max(np.abs(reported - derived))),
                max(float(np.max(np.abs(reported))), float(np.max(np.abs(derived)))),
                max(float(np.max(np.abs(reported - start))),
                    float(np.max(np.abs(derived - start)))),
            ))
    if n_points == 0:
        raise ValueError(
            "every plastic step in results was sub-stepped — no solution was checked"
            if n_substepped else
            "no plastic step in results — no solution was checked"
        )

    rows: dict[str, SolutionRowCheck] = {}
    for name, samples in res_rows.items():
        # A row whose terms all vanish at some step (a mechanism not yet
        # active) is measured against a small fraction of its own history
        # there, not against round-off.
        floor = np.sqrt(np.finfo(float).eps) * max(s for _, _, s, _ in samples)
        worst, at = 0.0, samples[0][0]
        for i, r, s, slack in samples:
            denom = max(s, floor) + slack / tol
            err = as_error(r / denom if denom > 0.0 else (0.0 if r == 0.0 else np.inf))
            if err > worst:
                worst, at = err, i
        rows[name] = SolutionRowCheck(name, "residual", worst, at, tol)

    not_exercised = []
    for key, samples in exp_rows.items():
        # Judged from the state itself, so no unit enters: it either moved
        # from where the step started, or its update was never used.
        if all(moved == 0.0 for _, _, _, moved in samples):
            not_exercised.append(key)
            continue
        magnitude = max(m for _, _, m, _ in samples)
        worst, at = 0.0, samples[0][0]
        for i, diff, _, _ in samples:
            err = as_error(diff / magnitude if magnitude > 0.0 else np.inf)
            if err > worst:
                worst, at = err, i
        rows[key] = SolutionRowCheck(key, "state", worst, at, tol)

    return SolutionReport(
        rows=rows, not_exercised=not_exercised,
        n_points=n_points, n_substepped=n_substepped,
    )


__all__ = ["SolutionRowCheck", "SolutionReport", "check_solution"]
