"""Jacobian block inspection and comparison for return-mapping residual systems.

:class:`JacobianChecker` is the working tool for deriving a Jacobian by hand:

* :meth:`~JacobianChecker.compute` — the autodiff Jacobian of the framework
  residual in named blocks, the reference a derivation is checked against;
* :meth:`~JacobianChecker.hand` — the same matrix assembled from the blocks the
  model declares with :func:`~manforge.core.material.hand_block`;
* :meth:`~JacobianChecker.check_hand_blocks` — every declared block against the
  reference over a whole history, with the worst deviation per block and the
  list of blocks not derived yet.

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
    residual_jacobian,
)
from manforge.verification.comparator_base import _array_rel_err


@dataclass
class JacobianComparisonResult:
    """Result of comparing Jacobian blocks from two StressUpdateResults.

    Attributes
    ----------
    passed : bool
        ``True`` if every block is within ``rtol``.
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
    rtol : float, default 1e-8
        Relative tolerance used by :meth:`compare`.

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

    def __init__(self, model, *, rtol: float = 1e-8):
        self.model = model
        self.rtol = rtol

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

        Every block declared with :func:`~manforge.core.material.hand_block` is
        taken from its method; the rest come from autodiff, so a derivation in
        progress still gives a complete matrix.  ``.source[row][col]`` says
        which is which.
        """
        return hand_jacobian(self.model, result, self._increment(result))

    def check_hand_blocks(
        self, results: Iterable[StressUpdateResult], *, tol: float = 1e-8,
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
        tol : float, default 1e-8
            Tolerance for blocks meant to be exact.  A block declared with
            ``approx=`` is held to that value instead.

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
        declared = type(model)._hand_blocks
        if not declared:
            raise ValueError(
                f"{type(model).__name__} declares no hand-derived block "
                "(decorate the methods with manforge.core.material.hand_block)"
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
                err = float(np.max(np.abs(got - ref))) / max(float(np.max(np.abs(ref))), 1.0)
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

    def compare(
        self,
        result_a: StressUpdateResult,
        result_b: StressUpdateResult,
    ) -> JacobianComparisonResult:
        """Compare the residual Jacobian at the converged points of two results.

        Both are evaluated on the residual of ``result_a``'s increment, so
        ``result_b`` may come from an integrator that reports none (Fortran).
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
        jac_b = residual_jacobian(self.model, result_b, result_a.increment)

        block_errs: dict[str, float] = {}
        blocks_b = {label: arr for label, arr in jac_b.iter_blocks()}

        for label, arr_a in jac_a.iter_blocks():
            arr_a = np.asarray(arr_a, dtype=float)
            arr_b = np.asarray(blocks_b.get(label, np.zeros_like(arr_a)), dtype=float)
            block_errs[label] = _array_rel_err(arr_a, arr_b)

        max_err = max(block_errs.values()) if block_errs else 0.0

        return JacobianComparisonResult(
            passed=max_err <= self.rtol,
            blocks=block_errs,
            max_rel_err=max_err,
        )
