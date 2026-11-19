"""The residual Jacobian of one stress update, in named blocks.

Two ways of obtaining the same matrix:

* :func:`residual_jacobian` — autodiff of the framework residual, the one the
  numerical Newton-Raphson solves.  This is the reference.
* :func:`hand_jacobian_blocks` — the blocks a model declares with
  :func:`~manforge.core.material.hand_block`, evaluated by calling them.

Both are keyed ``part[row][col]`` with the block shape rule of
:class:`JacobianBlocks`, so a hand derivation can be laid next to the
reference block by block.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

import autograd
import autograd.numpy as anp
import numpy as np

from manforge.core.material.hand_block import hand_block_record
from manforge.core.state import Increment, State
from manforge.simulation._layout import ResidualLayout
from manforge.simulation._residual import build_residual
from manforge._typing import FloatArray

if TYPE_CHECKING:
    from manforge.core.material import MaterialModel
    from manforge.core.result import StressUpdateResult


@dataclass
class JacobianBlocks:
    """Named blocks of the return-mapping residual Jacobian at one point.

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
    source : dict[str, dict[str, str]]
        Where each block came from: ``"ad"`` (autodiff of the framework
        residual) or ``"hand"`` (a method declared with ``hand_block``).
    """

    layout: ResidualLayout
    part: dict[str, dict[str, FloatArray]]
    full: FloatArray
    source: dict[str, dict[str, str]] = field(default_factory=dict)

    def row_names(self) -> tuple[str, ...]:
        """Residual-row labels in canonical order."""
        return self.layout.residual_names()

    def col_names(self) -> tuple[str, ...]:
        """State column names in canonical order: ``("stress", "dlambda", *implicit_keys)``."""
        return ("stress", "dlambda", *self.layout.implicit_keys)

    def iter_blocks(self) -> Iterator[tuple[str, anp.ndarray]]:
        """Iterate over all blocks as ``("row::col", array)`` pairs.

        Labels use the form ``"<residual_name>::<state_name>"``.
        """
        col_names = self.col_names()
        for row in self.row_names():
            for col in col_names:
                yield f"{row}::{col}", np.asarray(self.part[row][col])


def _unknowns(layout: ResidualLayout) -> tuple[str, ...]:
    return ("stress", "dlambda", *layout.implicit_keys)


def _split(layout: ResidualLayout, J: FloatArray, source: str) -> JacobianBlocks:
    unknowns = _unknowns(layout)
    part: dict[str, dict[str, FloatArray]] = {}
    src: dict[str, dict[str, str]] = {}
    for row_state in unknowns:
        row = layout.residual_name_for(row_state)
        part[row], src[row] = {}, {}
        for col in unknowns:
            block = J[layout.slot_slice(row_state), layout.slot_slice(col)]
            part[row][col] = block.reshape(layout.slot_shape(row_state) + layout.slot_shape(col))
            src[row][col] = source
    return JacobianBlocks(layout=layout, part=part, full=J, source=src)


def residual_jacobian(
    model: "MaterialModel", result: "StressUpdateResult", inc: Increment,
) -> JacobianBlocks:
    """Autodiff Jacobian of the residual of ``inc`` at the converged point of ``result``.

    On an elastic step the point is ``(σ_trial, Δλ = 0, elastic state)``.
    """
    residual_fn, layout = build_residual(model, inc)
    state = result.state
    x_conv = layout.pack(
        result.stress, result.dlambda, {k: state[k] for k in layout.implicit_keys}
    )  # type: ignore[arg-type]
    J = autograd.jacobian(residual_fn)(anp.array(x_conv))  # type: ignore[arg-type]
    return _split(layout, J, "ad")


def hand_jacobian_blocks(
    model: "MaterialModel", state, dlambda, inc: Increment,
) -> dict[tuple[str, str], FloatArray]:
    """Evaluate the blocks the model declares with ``hand_block``.

    Each declared method is called with the arguments it names (see
    :mod:`manforge.core.material.hand_block`) and its value is brought to the
    block shape of :class:`JacobianBlocks`.

    Parameters
    ----------
    model : MaterialModel
    state : State or dict
        The point the blocks are evaluated at — the full state, stress included.
    dlambda : scalar
        Plastic multiplier increment at that point.
    inc : Increment
        The increment; supplies ``state_n`` to blocks that ask for it.

    Returns
    -------
    dict[(row, col), ndarray]
        Declared blocks only.

    Raises
    ------
    ValueError
        If a block's value does not have as many entries as its slot.
    """
    layout = ResidualLayout.from_model(model)
    row_state = {layout.residual_name_for(u): u for u in _unknowns(layout)}
    if not isinstance(state, State):
        state = State(dict(state.items()), tuple(model.state_names))
    available = {"state": state, "dlambda": dlambda, "state_n": inc.state_n, "inc": inc}

    out: dict[tuple[str, str], FloatArray] = {}
    for (row, col), name in type(model)._hand_blocks.items():
        fn = getattr(model, name)
        kwargs = {p: available[p] for p in inspect.signature(fn).parameters}
        value = np.asarray(fn(**kwargs), dtype=float)
        shape = layout.slot_shape(row_state[row]) + layout.slot_shape(col)
        if value.size != int(np.prod(shape)):
            raise ValueError(
                f"{type(model).__name__}.{name}: hand block [{row}][{col}] has "
                f"shape {value.shape}, expected {shape}"
            )
        out[(row, col)] = value.reshape(shape)
    return out


def hand_jacobian(
    model: "MaterialModel", result: "StressUpdateResult", inc: Increment,
) -> JacobianBlocks:
    """The Jacobian with every declared block taken from the hand derivation.

    Blocks the model has not declared are filled from the autodiff reference
    (``source[row][col] == "ad"``), so a derivation in progress still yields a
    complete matrix.
    """
    reference = residual_jacobian(model, result, inc)
    layout = reference.layout
    row_state = {layout.residual_name_for(u): u for u in _unknowns(layout)}
    full = np.array(reference.full, dtype=float)
    blocks = hand_jacobian_blocks(model, result.state, result.dlambda, inc)
    for (row, col), value in blocks.items():
        r, c = layout.slot_slice(row_state[row]), layout.slot_slice(col)
        full[r, c] = value.reshape(full[r, c].shape)
    out = _split(layout, full, "ad")
    for row, col in blocks:
        out.source[row][col] = "hand"
    return out


__all__ = [
    "JacobianBlocks",
    "residual_jacobian",
    "hand_jacobian_blocks",
    "hand_jacobian",
    "hand_block_record",
]
