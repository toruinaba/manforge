"""Jacobian block decomposition utilities for return-mapping residual systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import autograd
import autograd.numpy as anp
import numpy as np

from manforge.core.result import StressUpdateResult
from manforge.core.state import Increment
from manforge.simulation._residual import build_residual
from manforge.simulation._layout import ResidualLayout
from manforge.verification.comparator_base import _array_rel_err
from manforge._typing import FloatArray


@dataclass
class JacobianBlocks:
    """Named blocks of the return-mapping residual Jacobian at the converged point.

    The residual / unknown vector layout follows :class:`~manforge.simulation._layout.ResidualLayout`:
    ``[σ (ntens) | Δλ (1) | q_implicit_non_stress (declaration order)]``.

    Blocks are accessed via ``part[row_residual_name][col_state_name]``.
    Row names are residual labels (``effective_residual_name`` from the field
    declaration, or ``model.dlambda_residual_name`` for the Δλ row).  Column
    names are state names (``"stress"``, ``"dlambda"``, and any implicit
    non-stress keys).

    **Block shape rule**::

        part[row][col].shape == layout.slot_shape(row_state) + layout.slot_shape(col)

    where ``row_state`` is the state name whose residual label equals ``row``
    (``layout.residual_name_for(row_state) == row``).

    Shape notation below mirrors the ``Implicit(shape=...)`` declaration:

    - ``NTENS``  — resolves to ``(ntens,)`` at runtime
    - ``SCALAR`` — resolves to ``()``  at runtime (0-d ndarray)

    When ``residual_name`` is not set, row and column names are identical
    (the default symmetric case):

    - ``part["stress"]["stress"]``    — ∂R_σ / ∂σ,   shape ``(NTENS, NTENS)`` ≡ ``(ntens, ntens)``
    - ``part["stress"]["dlambda"]``   — ∂R_σ / ∂Δλ,  shape ``(NTENS,)``       ≡ ``(ntens,)``
    - ``part["dlambda"]["stress"]``   — ∂R_Δλ / ∂σ,  shape ``(NTENS,)``       ≡ ``(ntens,)``
    - ``part["dlambda"]["dlambda"]``  — ∂R_Δλ / ∂Δλ, shape ``SCALAR``         ≡ ``()``
    - ``part["alpha"]["stress"]``     — ∂R_α / ∂σ,   ``(NTENS, NTENS)`` if α declared ``NTENS``
    - ``part["ep"]["alpha"]``         — ∂R_ep / ∂α,  ``(NTENS,)`` if ep is ``SCALAR``, α is ``NTENS``

    With opt-in residual names (e.g. ``Implicit(residual_name="R_alpha")``,
    ``dlambda_residual_name="R_yield"``):

    - ``part["R_alpha"]["stress"]``   — ∂R_α / ∂σ
    - ``part["R_yield"]["stress"]``   — ∂R_Δλ / ∂σ

    Attributes
    ----------
    layout : ResidualLayout
        Layout descriptor used to compute this Jacobian.
    part : dict[str, dict[str, ndarray]]
        ``part[residual_row_name][state_col_name]`` → block array.
    full : ndarray, shape (n_unknown, n_unknown)
        Full Jacobian matrix.
    """

    layout: ResidualLayout
    part: dict[str, dict[str, FloatArray]]
    full: FloatArray

    def row_names(self) -> tuple[str, ...]:
        """Residual-row labels in canonical order."""
        return self.layout.residual_names()

    def col_names(self) -> tuple[str, ...]:
        """State column names in canonical order: ``("stress", "dlambda", *implicit_keys)``."""
        return ("stress", "dlambda", *self.layout.implicit_keys)

    def iter_blocks(self) -> Iterator[tuple[str, anp.ndarray]]:
        """Iterate over all blocks as ``("row::col", array)`` pairs.

        Labels use the form ``"<residual_name>::<state_name>"``.
        Useful for :meth:`JacobianChecker.compare`.
        """
        col_names = self.col_names()
        for row in self.row_names():
            for col in col_names:
                yield f"{row}::{col}", np.asarray(self.part[row][col])


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

        # Manual crosscheck after a failed step
        cmp = checker.compare(result_a, result_b)
        if not cmp.passed:
            print(cmp.blocks)
    """

    def __init__(self, model, *, rtol: float = 1e-8):
        self.model = model
        self.rtol = rtol

    def compute(self, result: StressUpdateResult) -> JacobianBlocks:
        """Residual Jacobian at the converged point of ``result``, in named blocks.

        Parameters
        ----------
        result : StressUpdateResult
            Result of a stress update by a Python integrator.  On an elastic
            step the point is ``(σ_trial, Δλ = 0, trial state)``.

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
        if result.increment is None:
            raise ValueError(
                "this result carries no increment (it comes from an integrator "
                "that reports no trial state, such as FortranIntegrator), so "
                "the residual it would be evaluated on is not known.  Use "
                "compare(python_result, this_result)."
            )
        return self._at(result, result.increment)

    def _at(self, result: StressUpdateResult, inc: Increment) -> JacobianBlocks:
        """Jacobian of the residual of ``inc`` at the converged point of ``result``."""
        residual_fn, layout = build_residual(self.model, inc)

        state = result.state
        q_imp = {k: state[k] for k in layout.implicit_keys}
        x_conv = layout.pack(result.stress, result.dlambda, q_imp)  # type: ignore[arg-type]

        J = autograd.jacobian(residual_fn)(anp.array(x_conv))  # type: ignore[arg-type]

        col_names = ("stress", "dlambda", *layout.implicit_keys)
        part: dict[str, dict[str, FloatArray]] = {}
        for row_state in col_names:
            row = layout.residual_name_for(row_state)
            sl_row = layout.slot_slice(row_state)
            shp_row = layout.slot_shape(row_state)
            part[row] = {}
            for col in col_names:
                sl_col = layout.slot_slice(col)
                shp_col = layout.slot_shape(col)
                block = J[sl_row, sl_col]
                part[row][col] = block.reshape(shp_row + shp_col)

        return JacobianBlocks(layout=layout, part=part, full=J)

    def compare(
        self,
        result_a: StressUpdateResult,
        result_b: StressUpdateResult,
    ) -> JacobianComparisonResult:
        """Compare the residual Jacobian at the converged points of two results.

        Both are evaluated on the residual of ``result_a``'s increment, so
        ``result_b`` may come from an integrator that reports none (Fortran).
        A difference localises *where* two solutions of the same step disagree;
        it does not compare a hand-derived Jacobian with the autodiff one.

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
        jac_a = self._at(result_a, result_a.increment)
        jac_b = self._at(result_b, result_a.increment)

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
