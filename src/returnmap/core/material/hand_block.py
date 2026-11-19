"""Declaring hand-derived Jacobian blocks on a material model.

The deliverable of a model is a hand-written UMAT, so the residual Jacobian
is eventually derived by hand, block by block, and ported to Fortran.  Those
derivations live on the model as ordinary methods; :func:`hand_block` records
*which* block each one is::

    class MyModel(MaterialModel):
        stress = Implicit(shape=NTENS)
        alpha  = Implicit(shape=NTENS)

        @hand_block("alpha", "stress")            # ∂R_alpha / ∂σ
        def dRa_ds(self, state, dlambda):
            ...

With the blocks named, the framework can lay them out next to the autodiff
Jacobian of the same residual (:func:`~returnmap.simulation.hand_jacobian`,
:func:`~returnmap.verification.check_hand_blocks`) without a hand-maintained
table of "which method is which block", and can say which blocks are still
missing.

The decorator only attaches a record; the method itself is untouched and can
still be called directly, and stacked with ``verified_against_fortran``.

Row and column names are those of ``JacobianBlocks.part``: a row is a residual
name (``"stress"``, the model's ``dlambda_residual_name``, an implicit field's
``residual_name`` or its own name), a column is an unknown (``"stress"``,
``"dlambda"``, an implicit field).

Arguments
---------
A hand-derived block is user code that mirrors a Fortran routine, so it keeps
its own argument list.  The framework fills the arguments by *name*, from this
vocabulary:

``state``
    The point the block is evaluated at: the full state, stress included.
``dlambda``
    Plastic multiplier increment at that point.
``state_n``
    State at the beginning of the increment.
``inc``
    The whole :class:`~returnmap.core.Increment`.

Any other parameter name is rejected when the class is created.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, TypeVar

_F = TypeVar("_F", bound=Callable)

#: Parameter names a hand-derived block may take (besides ``self``).
HAND_BLOCK_ARGS = ("state", "dlambda", "state_n", "inc")


@dataclass(frozen=True)
class HandBlock:
    """Record attached to a method by :func:`hand_block`.

    Attributes
    ----------
    row, col : str
        The block is ``∂R_row / ∂col``.
    approx : float or None
        ``None`` for a block that is meant to equal the autodiff one.
        Otherwise the largest relative deviation from it that is *by design*
        — in the measure of :func:`~returnmap.verification.check_hand_blocks`,
        the largest difference over the block relative to the block's
        largest entry — with ``note`` saying why.
    note : str
        Why the block deviates.  Required when ``approx`` is given.
    """

    row: str
    col: str
    approx: float | None = None
    note: str = ""


def hand_block(
    row: str, col: str, *, approx: float | None = None, note: str = "",
) -> Callable[[_F], _F]:
    """Declare a method as the hand-derived Jacobian block ``∂R_row / ∂col``.

    Parameters
    ----------
    row, col : str
        Residual-row name and unknown name, as in ``JacobianBlocks.part``.
    approx : float, optional
        Declare the block a deliberate approximation, agreeing with the
        autodiff block only to this relative deviation.  A hand-written
        Newton-Raphson that updates explicit states between iterations
        differentiates the residual with those states frozen, so some of its
        blocks differ from the framework residual's by construction.
    note : str, optional
        The reason for ``approx``.  Required with it: a tolerance without its
        reason is indistinguishable from a bug that was waved through.
    """
    if approx is not None:
        if not approx > 0.0:
            raise ValueError(f"hand_block({row!r}, {col!r}): approx must be > 0, got {approx!r}")
        if not note.strip():
            raise ValueError(
                f"hand_block({row!r}, {col!r}): approx={approx!r} needs a note "
                "saying why the block deviates from the autodiff one"
            )
    record = HandBlock(row=row, col=col, approx=approx, note=note)

    def decorator(fn: _F) -> _F:
        fn.__hand_block__ = record  # type: ignore[attr-defined]
        return fn

    return decorator


def collect_hand_blocks(cls: type) -> dict[tuple[str, str], str]:
    """``(row, col) -> method name`` for every block declared on *cls*.

    Traverses the MRO from base to derived, so a subclass that re-declares a
    block shadows its parent's.  Two *different* methods of the resolved class
    claiming the same block is an error.
    """
    by_name: dict[str, HandBlock] = {}
    for klass in reversed(cls.__mro__):
        if klass is object:
            continue
        for name, obj in klass.__dict__.items():
            if isinstance(obj, (staticmethod, classmethod)) and isinstance(
                getattr(obj.__func__, "__hand_block__", None), HandBlock
            ):
                raise TypeError(
                    f"{klass.__name__}.{name}: a hand block must be an instance "
                    f"method, not a {type(obj).__name__} (its arguments are "
                    "filled by name: state, dlambda, state_n, inc)"
                )
            record = getattr(obj, "__hand_block__", None)
            if isinstance(record, HandBlock):
                # A later override without the decorator leaves this entry in
                # place: it is still the same block, by name.
                by_name[name] = record

    blocks: dict[tuple[str, str], str] = {}
    for name, record in by_name.items():
        key = (record.row, record.col)
        if key in blocks:
            raise TypeError(
                f"{cls.__name__}: hand block [{record.row}][{record.col}] is "
                f"declared by both {blocks[key]}() and {name}()"
            )
        blocks[key] = name
    return blocks


def hand_block_record(cls: type, method_name: str) -> HandBlock:
    """The :class:`HandBlock` record of ``cls.<method_name>`` (searched up the MRO)."""
    for klass in cls.__mro__:
        obj = klass.__dict__.get(method_name)
        record = getattr(obj, "__hand_block__", None)
        if isinstance(record, HandBlock):
            return record
    raise KeyError(f"{cls.__name__}.{method_name} is not declared with hand_block")


def validate_hand_blocks(
    cls: type, blocks: dict[tuple[str, str], str],
    row_names: tuple[str, ...], col_names: tuple[str, ...],
) -> None:
    """Reject blocks naming an unknown row/column or taking an unknown argument."""
    for (row, col), name in blocks.items():
        if row not in row_names:
            raise TypeError(
                f"{cls.__name__}.{name}: hand block row {row!r} is not a residual "
                f"row of this model {list(row_names)}"
            )
        if col not in col_names:
            raise TypeError(
                f"{cls.__name__}.{name}: hand block column {col!r} is not an "
                f"unknown of this model {list(col_names)}"
            )
        params = tuple(inspect.signature(getattr(cls, name)).parameters)[1:]
        unknown = [p for p in params if p not in HAND_BLOCK_ARGS]
        if unknown:
            raise TypeError(
                f"{cls.__name__}.{name}: hand block takes {unknown}, which the "
                f"framework cannot supply.  Arguments are filled by name from "
                f"{list(HAND_BLOCK_ARGS)}."
            )
