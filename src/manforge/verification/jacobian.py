"""Jacobian block inspection and comparison for return-mapping residual systems.

:class:`JacobianChecker` is the working tool for deriving a Jacobian by hand:

* :meth:`~JacobianChecker.compute` — the autodiff Jacobian of the framework
  residual in named blocks, the reference a derivation is checked against;
* :meth:`~JacobianChecker.hand` — the same matrix assembled from the blocks the
  model declares with :func:`~manforge.core.hand_block`;
* :meth:`~JacobianChecker.check_hand_blocks` — every declared block against the
  reference over a whole history, with the worst deviation per block and the
  list of blocks not derived yet;
* :meth:`~JacobianChecker.check_solution` — the solutions of a hand-written
  return mapping substituted into the framework residual, row by row.

The matrices themselves come from :mod:`manforge.simulation.jacobian`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from manforge.core.material.hand_block import hand_block_record
from manforge.core.result import StressUpdateResult
from manforge.core.state import Increment
from manforge.simulation.jacobian import (
    JacobianBlocks,
    hand_jacobian,
    hand_jacobian_blocks,
    residual_at,
    residual_jacobian,
)


@dataclass
class JacobianComparisonResult:
    """Result of comparing Jacobian blocks from two StressUpdateResults.

    Attributes
    ----------
    passed : bool
        ``True`` if every block is within ``tol``.
    blocks : dict[str, float]
        Block label → maximum relative error.
    max_rel_err : float
        Maximum relative error across all blocks.
    """

    passed: bool
    blocks: dict[str, float]
    max_rel_err: float


@dataclass
class HandBlockCheck:
    """One declared hand block against the autodiff reference.

    Attributes
    ----------
    row, col : str
        The block ``∂R_row / ∂col``.
    method : str
        Name of the model method that computes it.
    worst_err : float
        Largest deviation over the checked points:
        ``max|hand − ad| / max(max|ad|, 1)`` per point.
    worst_at : int
        Index, in the checked sequence, of the point where it occurred.
    tol : float
        What the block is held to: the declared ``approx`` for a deliberate
        approximation, otherwise the tolerance of the check.
    approx : bool
        Whether the block is declared an approximation.
    note : str
        The declared reason for the approximation.
    """

    row: str
    col: str
    method: str
    worst_err: float
    worst_at: int
    tol: float
    approx: bool
    note: str

    @property
    def passed(self) -> bool:
        return self.worst_err <= self.tol


@dataclass
class HandBlockReport:
    """Result of :meth:`JacobianChecker.check_hand_blocks`.

    Attributes
    ----------
    blocks : dict[(row, col), HandBlockCheck]
        Every block the model declares.
    undeclared : list[(row, col)]
        Blocks of the Jacobian with no hand derivation yet.
    n_points : int
        Number of plastic steps the blocks were evaluated at.
    """

    blocks: dict[tuple[str, str], HandBlockCheck]
    undeclared: list[tuple[str, str]] = field(default_factory=list)
    n_points: int = 0

    @property
    def passed(self) -> bool:
        """Every declared block within its tolerance.  Says nothing about undeclared ones."""
        return all(b.passed for b in self.blocks.values())

    @property
    def failed(self) -> list[HandBlockCheck]:
        """Blocks outside their tolerance, worst first."""
        bad = [b for b in self.blocks.values() if not b.passed]
        return sorted(bad, key=lambda b: -b.worst_err)

    @property
    def n_declared(self) -> int:
        return len(self.blocks)

    @property
    def n_total(self) -> int:
        return len(self.blocks) + len(self.undeclared)

    def summary(self) -> str:
        """Multi-line text: coverage, then one line per declared block."""
        lines = [
            f"hand blocks: {self.n_declared}/{self.n_total} derived, "
            f"checked at {self.n_points} plastic steps — "
            + ("all within tolerance" if self.passed else f"{len(self.failed)} outside tolerance")
        ]
        for b in sorted(self.blocks.values(), key=lambda b: -b.worst_err):
            mark = "ok  " if b.passed else "FAIL"
            tail = f"  (approx: {b.note})" if b.approx else ""
            lines.append(
                f"  {mark} [{b.row}][{b.col}] {b.method}: "
                f"{b.worst_err:.3e} (tol {b.tol:.0e}, worst at point {b.worst_at}){tail}"
            )
        if self.undeclared:
            lines.append(
                "  not derived: " + ", ".join(f"[{r}][{c}]" for r, c in self.undeclared)
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _block_rel_err(ref: np.ndarray, cand: np.ndarray) -> float:
    """``max|cand − ref| / max|ref|`` of one Jacobian block.

    Whole-block, not component-wise: an entry that is zero in exact
    arithmetic and 1e-17 in floating point is round-off, not a 100 % error.
    """
    diff = float(np.max(np.abs(cand - ref)))
    scale = float(np.max(np.abs(ref)))
    if np.isnan(diff):
        return float("inf")
    if scale > 0.0:
        return diff / scale
    return 0.0 if diff == 0.0 else float("inf")


def _as_error(value: float) -> float:
    """NaN is the worst error, not the absence of one.

    ``nan > x`` is False for every x, so a NaN left as is would lose every
    "is this worse than the worst so far" comparison and vanish.
    """
    return float("inf") if np.isnan(value) else float(value)


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
        :meth:`JacobianChecker.check_solution` for the measure).
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
    """Result of :meth:`JacobianChecker.check_solution`.

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


class JacobianChecker:
    """Compute and compare residual Jacobians for a material model.

    The Jacobian is that of the framework residual — the one the autodiff
    Newton-Raphson solves — evaluated at a converged point.  Its named blocks
    are the reference a hand-derived Jacobian is checked against.

    A result carries the increment it was computed for (state at the start of
    the step, Δε, trial state), so nothing else has to be passed alongside it,
    and the Jacobian cannot be evaluated on a residual other than the one the
    solver used.

    Parameters
    ----------
    model : MaterialModel
    tol : float, default 1e-8
        Pass threshold of the checks, as a relative error: used by
        :meth:`compare`, and by :meth:`check_hand_blocks` for blocks meant to
        be exact.

    Examples
    --------
    ::

        result = PythonIntegrator(model).stress_update(deps, stress_n, state_n)
        jac = JacobianChecker(model).compute(result)
        print(jac.part["dlambda"]["stress"])

        # Every hand-derived block against autodiff, over a history
        report = JacobianChecker(model).check_hand_blocks(driver.run(load).step_results)
        assert report.passed, report.summary()
    """

    def __init__(self, model, *, tol: float = 1e-8):
        self.model = model
        self.tol = tol

    @staticmethod
    def _increment(result: StressUpdateResult) -> Increment:
        if result.increment is None:
            raise ValueError(
                "this result carries no increment (it comes from an integrator "
                "that reports no trial state, such as FortranIntegrator), so "
                "the residual it would be evaluated on is not known.  Use "
                "compare(python_result, this_result)."
            )
        return result.increment

    def compute(self, result: StressUpdateResult) -> JacobianBlocks:
        """Autodiff Jacobian at the converged point of ``result``, in named blocks.

        Parameters
        ----------
        result : StressUpdateResult
            Result of a stress update by a Python integrator.  On an elastic
            step the point is ``(σ_trial, Δλ = 0, elastic state)``.

        Returns
        -------
        JacobianBlocks

        Raises
        ------
        ValueError
            If ``result`` carries no increment (a Fortran UMAT result).  Use
            :meth:`compare` to evaluate at such a point with the increment of
            a Python result of the same step.
        """
        return residual_jacobian(self.model, result, self._increment(result))

    def hand(self, result: StressUpdateResult) -> JacobianBlocks:
        """The Jacobian at ``result`` with the model's hand-derived blocks in place.

        Every block declared with :func:`~manforge.core.hand_block` is
        taken from its method; the rest come from autodiff, so a derivation in
        progress still gives a complete matrix.  ``.source[row][col]`` says
        which is which.
        """
        return hand_jacobian(self.model, result, self._increment(result))

    def check_hand_blocks(
        self, results: Iterable[StressUpdateResult], *, tol: float | None = None,
    ) -> HandBlockReport:
        """Check every declared hand block against autodiff, over many steps.

        Errors are collected across all blocks and all steps rather than
        stopping at the first mismatch — a single wrong block otherwise hides
        behind whichever one happens to fail first.

        The history matters as much as the check.  A transposed outer product
        or a ×2 shear-convention error is invisible under uniaxial or
        pure-shear loading; drive direct and shear components together.

        Parameters
        ----------
        results : iterable of StressUpdateResult
            Typically ``driver.run(load).step_results``.  Only plastic steps
            are used; the rest are skipped.
        tol : float or None
            Tolerance for blocks meant to be exact; ``None`` (default) uses
            the checker's own ``tol``.  A block declared with ``approx=`` is
            held to that value instead.

        Returns
        -------
        HandBlockReport
            ``report.passed``, and ``report.summary()`` for the per-block table.

        Raises
        ------
        ValueError
            If the model declares no hand block, or ``results`` holds no
            plastic step — either way nothing would have been checked.
        """
        model = self.model
        if tol is None:
            tol = self.tol
        declared = type(model)._hand_blocks
        if not declared:
            raise ValueError(
                f"{type(model).__name__} declares no hand-derived block "
                "(decorate the methods with manforge.core.hand_block)"
            )
        worst: dict[tuple[str, str], tuple[float, int]] = {}
        rows: tuple[str, ...] = ()
        cols: tuple[str, ...] = ()
        n_points = 0
        for i, result in enumerate(results):
            if result.is_plastic is not True:
                continue
            inc = self._increment(result)
            reference = residual_jacobian(model, result, inc)
            rows, cols = reference.row_names(), reference.col_names()
            hand = hand_jacobian_blocks(model, result.state, result.dlambda, inc)
            n_points += 1
            for key, got in hand.items():
                ref = np.asarray(reference.part[key[0]][key[1]], dtype=float)
                err = _as_error(
                    float(np.max(np.abs(got - ref))) / max(float(np.max(np.abs(ref))), 1.0))
                if key not in worst or err > worst[key][0]:
                    worst[key] = (err, i)
        if n_points == 0:
            raise ValueError("no plastic step in results — no hand block was evaluated")

        checks = {}
        for key, name in declared.items():
            record = hand_block_record(type(model), name)
            err, at = worst[key]
            checks[key] = HandBlockCheck(
                row=key[0], col=key[1], method=name, worst_err=err, worst_at=at,
                tol=tol if record.approx is None else record.approx,
                approx=record.approx is not None, note=record.note,
            )
        undeclared = [(r, c) for r in rows for c in cols if (r, c) not in declared]
        return HandBlockReport(blocks=checks, undeclared=undeclared, n_points=n_points)

    def check_solution(
        self, results: Iterable[StressUpdateResult], *, tol: float | None = None,
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
        results : iterable of StressUpdateResult
            Typically ``driver.run(load).step_results`` of a Python
            integrator.  Only plastic steps are used.
        tol : float or None
            Tolerance; ``None`` (default) uses the checker's own ``tol``.

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
        model = self.model
        if tol is None:
            tol = self.tol

        res_rows: dict[str, list[tuple[int, float, float, float]]] = {}
        exp_rows: dict[str, list[tuple[int, float, float, float]]] = {}
        n_points = 0
        n_substepped = 0
        for i, result in enumerate(results):
            inc = self._increment(result)
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
                err = _as_error(r / denom if denom > 0.0 else (0.0 if r == 0.0 else np.inf))
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
                err = _as_error(diff / magnitude if magnitude > 0.0 else np.inf)
                if err > worst:
                    worst, at = err, i
            rows[key] = SolutionRowCheck(key, "state", worst, at, tol)

        return SolutionReport(
            rows=rows, not_exercised=not_exercised,
            n_points=n_points, n_substepped=n_substepped,
        )

    def compare(
        self,
        result_a: StressUpdateResult,
        result_b: StressUpdateResult,
    ) -> JacobianComparisonResult:
        """Compare the residual Jacobian at the converged points of two results.

        Both are evaluated on the residual of ``result_a``'s increment, so
        ``result_b`` may come from an integrator that reports none (Fortran).
        A Fortran UMAT reports no Δλ either; ``result_b``'s point then takes
        ``result_a``'s, so the comparison is over stress and state.  Each
        block is measured as ``max|b − a| / max|a|`` over the block.
        A difference localises *where* two solutions of the same step disagree;
        it does not compare a hand-derived Jacobian with the autodiff one — that
        is :meth:`check_hand_blocks`.

        Parameters
        ----------
        result_a : StressUpdateResult
            Result of a Python integrator (e.g. ``PythonNumericalIntegrator``).
        result_b : StressUpdateResult
            Result of the same step by another integrator.

        Returns
        -------
        JacobianComparisonResult
        """
        if result_a.increment is None:
            raise ValueError(
                "result_a must come from a Python integrator: its increment "
                "defines the residual both points are evaluated on"
            )
        jac_a = residual_jacobian(self.model, result_a, result_a.increment)
        jac_b = residual_jacobian(
            self.model, result_b, result_a.increment,
            dlambda_if_missing=result_a.dlambda,
        )

        block_errs: dict[str, float] = {}
        blocks_b = {label: arr for label, arr in jac_b.iter_blocks()}

        for label, arr_a in jac_a.iter_blocks():
            arr_a = np.asarray(arr_a, dtype=float)
            arr_b = np.asarray(blocks_b.get(label, np.zeros_like(arr_a)), dtype=float)
            block_errs[label] = _block_rel_err(arr_a, arr_b)

        max_err = max(block_errs.values()) if block_errs else 0.0

        return JacobianComparisonResult(
            passed=max_err <= self.tol,
            blocks=block_errs,
            max_rel_err=max_err,
        )
