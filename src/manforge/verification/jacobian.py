"""Jacobian block decomposition utilities for return-mapping residual systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import autograd
import autograd.numpy as anp
import numpy as np

from manforge.simulation._residual import build_residual
from manforge.simulation._layout import ResidualLayout


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

    When ``residual_name`` is not set, row and column names are identical
    (the default symmetric case):

    - ``part["stress"]["stress"]``    — ∂R_σ / ∂σ,   shape ``(ntens, ntens)``
    - ``part["stress"]["dlambda"]``   — ∂R_σ / ∂Δλ,  shape ``(ntens,)``
    - ``part["dlambda"]["stress"]``   — ∂R_Δλ / ∂σ,  shape ``(ntens,)``
    - ``part["dlambda"]["dlambda"]``  — ∂R_Δλ / ∂Δλ, shape ``()``
    - ``part["alpha"]["stress"]``     — ∂R_α / ∂σ (when alpha implicit)
    - ``part["ep"]["alpha"]``         — ∂R_ep / ∂α

    With opt-in residual names (e.g. ``Implicit(residual_name="R_alpha")``,
    ``dlambda_residual_name="R_yield"``):

    - ``part["R_alpha"]["stress"]``   — ∂R_α / ∂σ
    - ``part["R_yield"]["stress"]``   — ∂R_Δλ / ∂σ

    ``part[row][col].shape == layout.slot_shape(col_state) + layout.slot_shape(col)``
    where ``col_state`` is the state name corresponding to ``row``.

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
    part: dict
    full: anp.ndarray

    def row_names(self) -> tuple:
        """Residual-row labels in canonical order."""
        return self.layout.residual_names()

    def col_names(self) -> tuple:
        """State column names in canonical order: ``("stress", "dlambda", *implicit_keys)``."""
        return ("stress", "dlambda", *self.layout.implicit_keys)

    def iter_blocks(self) -> Iterator[tuple[str, anp.ndarray]]:
        """Iterate over all blocks as ``("row::col", array)`` pairs.

        Labels use the form ``"<residual_name>::<state_name>"``.
        Useful for :func:`compare_jacobians`.
        """
        col_names = self.col_names()
        for row in self.row_names():
            for col in col_names:
                yield f"{row}::{col}", np.asarray(self.part[row][col])


def ad_jacobian_blocks(
    model, result, state_n: dict, *, stress_trial=None
) -> JacobianBlocks:
    """Compute the residual Jacobian at the converged point and decompose into blocks.

    Parameters
    ----------
    model : MaterialModel
    result : StressUpdateResult or ReturnMappingResult
        Converged result from a stress integration step.
    state_n : dict
        State at the beginning of the increment (must include ``"stress"``).
    stress_trial : array-like, optional
        Required when *result* is a :class:`~manforge.core.result.ReturnMappingResult`.

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
                "to ad_jacobian_blocks(). Use stress_update() instead, or pass "
                "stress_trial=... explicitly."
            )

    residual_fn, layout = build_residual(model, stress_trial, state_n)

    q_imp = {k: state[k] for k in layout.implicit_keys}
    x_conv = layout.pack(stress, dlambda, q_imp)

    J = autograd.jacobian(residual_fn)(anp.array(x_conv))

    col_names = ("stress", "dlambda", *layout.implicit_keys)
    part: dict = {}
    for col_state in col_names:
        row = layout.residual_name_for(col_state)
        sl_row = layout.slot_slice(col_state)
        part[row] = {}
        for col in col_names:
            sl_col = layout.slot_slice(col)
            block = J[sl_row, sl_col]
            shp_row = layout.slot_shape(col_state)
            shp_col = layout.slot_shape(col)
            part[row][col] = block.reshape(shp_row + shp_col)

    return JacobianBlocks(layout=layout, part=part, full=J)
