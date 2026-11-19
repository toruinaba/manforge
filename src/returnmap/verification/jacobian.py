"""Checks on the residual Jacobian of a return mapping.

* :func:`check_hand_blocks` — every block a model declares with
  :func:`~returnmap.core.hand_block` against the autodiff Jacobian of the
  framework residual, over a whole history: the worst deviation per block and
  the list of blocks not derived yet.
* :func:`compare_jacobians` — the Jacobian at the points two integrators
  reached for the same step, block by block, to localise where two solutions
  disagree.

The matrices themselves are not obtained here:
``integrator.jacobian(result)`` is the autodiff Jacobian and
:func:`~returnmap.simulation.hand_jacobian` the hand-derived one, both as
:class:`~returnmap.simulation.JacobianBlocks`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from returnmap.core.material.hand_block import hand_block_record
from returnmap.core.result import StressUpdateResult
from returnmap.simulation.jacobian import (
    JacobianBlocks,
    hand_jacobian_blocks,
    increment_of,
    point_of,
    residual_jacobian,
)
from returnmap.verification._measure import NOISE_FLOOR, as_error, max_abs, rel_err

if TYPE_CHECKING:
    from returnmap.core.material import MaterialModel


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
        ``max|hand − ad| / max|ad|``, both over the block and over the
        checked points (see :func:`check_hand_blocks` for a block that is
        zero).
    worst_at : int
        Index, in the checked sequence, of the point with the largest
        difference.
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
    """Result of :func:`check_hand_blocks`.

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


@dataclass
class JacobianComparisonReport:
    """Result of :func:`compare_jacobians`.

    Attributes
    ----------
    blocks : dict[str, float]
        Block label (``"row::col"``) → ``max|b − a| / max|a|`` over the
        block (a block that is zero at ``a`` is treated as in
        :func:`check_hand_blocks`).
    tol : float
        The tolerance the blocks were judged against.
    """

    blocks: dict[str, float]
    tol: float

    @property
    def max_rel_err(self) -> float:
        """The largest block error."""
        return max(self.blocks.values(), default=0.0)

    @property
    def passed(self) -> bool:
        """Every block within ``tol``."""
        return all(err <= self.tol for err in self.blocks.values())

    @property
    def failed(self) -> list[str]:
        """Labels of the blocks outside ``tol``, worst first."""
        bad = [k for k, err in self.blocks.items() if not err <= self.tol]
        return sorted(bad, key=lambda k: -self.blocks[k])

    def summary(self) -> str:
        """Multi-line text: the verdict, then the blocks, worst first."""
        n_bad = len(self.failed)
        lines = [
            f"jacobian comparison: {len(self.blocks)} blocks — "
            + ("all agree" if self.passed else f"{n_bad} differ")
        ]
        for label, err in sorted(self.blocks.items(), key=lambda kv: -kv[1]):
            mark = "ok  " if err <= self.tol else "FAIL"
            lines.append(f"  {mark} {label}: {err:.3e} (tol {self.tol:.0e})")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _row_and_column_sizes(jac: JacobianBlocks, x) -> tuple[dict[str, float], dict[str, float]]:
    """The size of each residual row's terms, and of each unknown, at one point.

    A row ``R_r = Σ_c J_rc x_c + …`` is made of terms of size ``Σ |J_rc| |x_c|``;
    that sum is taken as the size of the row (the scale ``check_solution``
    measures a residual against).  Both are in the units of the model, and
    their quotient ``row / column`` has the unit of the block ``∂R_r/∂x_c``.
    """
    layout = jac.layout
    x = np.abs(np.asarray(x, dtype=float))
    terms = np.abs(np.asarray(jac.full, dtype=float)) @ x
    rows, cols = {}, {}
    for unknown in jac.col_names():
        sl = layout.slot_slice(unknown)
        rows[layout.residual_name_for(unknown)] = max_abs(terms[sl])
        cols[unknown] = max_abs(x[sl])
    return rows, cols


def _block_scale(own: float, row: float, col: float) -> float:
    """What a block's error is relative to — see the Notes of :func:`check_hand_blocks`.

    ``own`` is the block's largest entry, ``row`` the size of its residual
    row's terms and ``col`` the size of the unknown it multiplies.  The
    block's own size, unless that is round-off next to ``row / col``.
    """
    if not (col > 0.0 and np.isfinite(row) and np.isfinite(col)):
        return own
    implied = row / col
    return own if own > NOISE_FLOOR * implied else implied


def check_hand_blocks(
    model: "MaterialModel",
    results: Iterable[StressUpdateResult],
    *,
    tol: float = 1e-8,
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
    model : MaterialModel
        The model whose ``hand_block`` methods are checked.
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
        plastic step — either way nothing would have been checked.  Also if
        a result carries no increment (a Fortran UMAT result).

    Notes
    -----
    The error of a block is the largest ``|hand − ad|`` over its entries and
    over the checked steps, relative to the largest ``|ad|`` over the same —
    ``max|hand − ad| / max|ad|``.  A sign error reads 2, a dropped block 1.
    Taken over the whole history, a block that vanishes at some step — a
    mechanism not active there — is measured against its size in the rest of
    the history, not against round-off.

    One case has no ``max|ad|`` to be relative to: a block that is zero in
    exact arithmetic.  Autodiff returns round-off there (1e-13, say) and a
    hand derivation a literal 0, which relative to the block's own entries
    is a 100 % error.  Such a block is recognised by its size next to what
    its row and column imply — the terms the residual row is made of
    (``Σ |∂R/∂x| · |x|``, the scale :func:`check_solution` uses) over the
    unknown the block multiplies, both at their largest over the checked
    steps.  A block that stays twelve orders below that is round-off, and is
    measured against that implied size instead: a hand block of 0 passes,
    one that is not negligible next to its row does not.

    Both sizes carry the unit of the block, so the error has none.
    """
    declared = type(model)._hand_blocks
    if not declared:
        raise ValueError(
            f"{type(model).__name__} declares no hand-derived block "
            "(decorate the methods with returnmap.core.hand_block)"
        )
    # Per block: the largest difference, the point it occurred at, and the
    # largest magnitude of the reference.  Per row and per unknown: its size.
    diff: dict[tuple[str, str], float] = {}
    at: dict[tuple[str, str], int] = {}
    own: dict[tuple[str, str], float] = {}
    row_size: dict[str, float] = {}
    col_size: dict[str, float] = {}
    rows: tuple[str, ...] = ()
    cols: tuple[str, ...] = ()
    n_points = 0
    for i, result in enumerate(results):
        if result.is_plastic is not True:
            continue
        inc = increment_of(result)
        reference = residual_jacobian(model, result, inc)
        rows, cols = reference.row_names(), reference.col_names()
        hand = hand_jacobian_blocks(model, result.state, result.dlambda, inc)
        n_points += 1
        r_size, c_size = _row_and_column_sizes(reference, point_of(reference.layout, result))
        for name, value in r_size.items():
            row_size[name] = float(np.fmax(row_size.get(name, 0.0), value))
        for name, value in c_size.items():
            col_size[name] = float(np.fmax(col_size.get(name, 0.0), value))
        for key, got in hand.items():
            ref = np.asarray(reference.part[key[0]][key[1]], dtype=float)
            d = as_error(max_abs(got - ref))
            if key not in diff or d > diff[key]:
                diff[key], at[key] = d, i
            own[key] = float(np.fmax(own.get(key, 0.0), max_abs(ref)))
    if n_points == 0:
        raise ValueError("no plastic step in results — no hand block was evaluated")

    checks = {}
    for key, name in declared.items():
        record = hand_block_record(type(model), name)
        checks[key] = HandBlockCheck(
            row=key[0], col=key[1], method=name,
            worst_err=rel_err(
                diff[key], _block_scale(own[key], row_size[key[0]], col_size[key[1]])),
            worst_at=at[key],
            tol=tol if record.approx is None else record.approx,
            approx=record.approx is not None, note=record.note,
        )
    undeclared = [(r, c) for r in rows for c in cols if (r, c) not in declared]
    return HandBlockReport(blocks=checks, undeclared=undeclared, n_points=n_points)


def compare_jacobians(
    model: "MaterialModel",
    result_a: StressUpdateResult,
    result_b: StressUpdateResult,
    *,
    tol: float = 1e-8,
) -> JacobianComparisonReport:
    """Compare the residual Jacobian at the converged points of two results.

    Both are evaluated on the residual of ``result_a``'s increment, so
    ``result_b`` may come from an integrator that reports none (Fortran).
    A Fortran UMAT reports no Δλ either; ``result_b``'s point then takes
    ``result_a``'s, so the comparison is over stress and state.

    A difference localises *where* two solutions of the same step disagree;
    it does not compare a hand-derived Jacobian with the autodiff one — that
    is :func:`check_hand_blocks`.

    Parameters
    ----------
    model : MaterialModel
        The model both results belong to.
    result_a : StressUpdateResult
        The reference: the result of a Python integrator.
    result_b : StressUpdateResult
        The candidate: the same step by another integrator.
    tol : float, default 1e-8
        Pass threshold on each block's ``max|b − a| / max|a|``.

    Returns
    -------
    JacobianComparisonReport

    Raises
    ------
    ValueError
        If ``result_a`` carries no increment.
    """
    if result_a.increment is None:
        raise ValueError(
            "result_a must come from a Python integrator: its increment "
            "defines the residual both points are evaluated on"
        )
    jac_a = residual_jacobian(model, result_a, result_a.increment)
    jac_b = residual_jacobian(
        model, result_b, result_a.increment,
        dlambda_if_missing=result_a.dlambda,
    )
    row_size, col_size = _row_and_column_sizes(jac_a, point_of(jac_a.layout, result_a))
    block_errs = {}
    for row in jac_a.row_names():
        for col in jac_a.col_names():
            a = np.asarray(jac_a.part[row][col], dtype=float)
            b = np.asarray(jac_b.part[row][col], dtype=float)
            block_errs[f"{row}::{col}"] = rel_err(
                as_error(max_abs(b - a)),
                _block_scale(max_abs(a), row_size[row], col_size[col]),
            )
    return JacobianComparisonReport(blocks=block_errs, tol=tol)


__all__ = [
    "HandBlockCheck",
    "HandBlockReport",
    "JacobianComparisonReport",
    "check_hand_blocks",
    "compare_jacobians",
]
