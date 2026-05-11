"""Jacobian block decomposition utilities for return-mapping residual systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import autograd
import autograd.numpy as anp
import numpy as np

from manforge.simulation._residual import build_residual
from manforge.simulation._layout import ResidualLayout
from manforge.verification.comparator_base import _array_rel_err
from manforge._typing import FloatArray, StressVec, StateDict


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

    Wraps the Jacobian computation and comparison operations for a fixed model,
    mirroring the class-based interface of ``CrosscheckStrainDriver`` /
    ``CrosscheckStressDriver``.

    Parameters
    ----------
    model : MaterialModel
    rtol : float, default 1e-8
        Relative tolerance used by :meth:`compare`.

    Examples
    --------
    ::

        checker = JacobianChecker(model)
        jac = checker.compute(result, state_n)
        print(jac.part["dlambda"]["stress"])

        # Manual crosscheck after a failed step
        cmp = checker.compare(result_a, result_b, state_n)
        if not cmp.passed:
            print(cmp.blocks)
    """

    def __init__(self, model, *, rtol: float = 1e-8):
        self.model = model
        self.rtol = rtol

    def compute(self, result: object, state_n: StateDict, *, stress_trial: "StressVec | None" = None) -> JacobianBlocks:
        """Compute the residual Jacobian at the converged point and decompose into blocks.

        Parameters
        ----------
        result : StressUpdateResult or ReturnMappingResult
            Converged result from a stress integration step.
        state_n : dict
            State at the beginning of the increment (must include ``"stress"``).
        stress_trial : array-like, optional
            Required when *result* is a
            :class:`~manforge.core.result.ReturnMappingResult`.

        Returns
        -------
        JacobianBlocks
        """
        from manforge.core.result import StressUpdateResult

        if isinstance(result, StressUpdateResult):
            if result.return_mapping is None:
                stress = result.stress_trial
                dlambda = anp.array(0.0)
                stress_trial = result.stress_trial
                state = result.state
            else:
                stress = result.return_mapping.stress
                dlambda = result.return_mapping.dlambda
                stress_trial = result.stress_trial
                state = result.return_mapping.state
        else:
            stress = result.stress
            dlambda = result.dlambda
            state = result.state
            if stress_trial is None:
                raise ValueError(
                    "stress_trial must be provided when passing a ReturnMappingResult "
                    "to compute(). Use stress_update() instead, or pass "
                    "stress_trial=... explicitly."
                )

        assert stress_trial is not None
        residual_fn, layout = build_residual(self.model, stress_trial, state_n)

        q_imp = {k: state[k] for k in layout.implicit_keys}
        x_conv = layout.pack(stress, dlambda, q_imp)  # type: ignore[arg-type]

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
        self, result_a: object, result_b: object, state_n: StateDict
    ) -> JacobianComparisonResult:
        """Compare Jacobian blocks from two StressUpdateResults.

        Parameters
        ----------
        result_a : StressUpdateResult
            First result (e.g. from ``PythonNumericalIntegrator``).
        result_b : StressUpdateResult
            Second result (e.g. from ``PythonAnalyticalIntegrator``).
        state_n : dict
            Initial state at the start of the step (before the increment).

        Returns
        -------
        JacobianComparisonResult
        """
        jac_a = self.compute(result_a, state_n)
        jac_b = self.compute(result_b, state_n)

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
