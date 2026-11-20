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

Deliberate approximations
-------------------------
A hand derivation is sometimes not meant to equal the autodiff one, and the
model says so where it says what the block is — never by loosening a
tolerance in a test, where the reason is lost.  There are two ways:

``approx=0.05``
    The block agrees with autodiff only to this relative deviation, at every
    step.  For a term dropped from the derivation everywhere.
``approx_when="method_name"``
    The block is exact except on the steps where the named method of the
    model returns true.  For a derivation that treats a switch as constant:
    exact while the switch is off or on, and not on the step where it is
    in between.  Those steps are left out of the comparison and counted in
    the report; every other step is still held to the exact tolerance.  With
    ``approx=`` as well, they are held to that deviation instead of left out.

The condition method takes its arguments by name from the vocabulary above
and returns a bool.  A hand-written tangent is declared the same way with
:func:`hand_tangent`.
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
    approx_when : str or None
        Name of the model method that says on which steps the block is an
        approximation.  ``None``: on every step (if ``approx`` is given) or
        on none.
    note : str
        Why the block deviates.  Required when ``approx`` or ``approx_when``
        is given.
    """

    row: str
    col: str
    approx: float | None = None
    approx_when: str | None = None
    note: str = ""


def _check_approximation(
    what: str, approx: float | None, approx_when: str | None, note: str,
) -> None:
    """Shared by hand_block and hand_tangent: a declared approximation has a reason."""
    if approx is not None and not approx > 0.0:
        raise ValueError(f"{what}: approx must be > 0, got {approx!r}")
    if approx_when is not None and not (isinstance(approx_when, str) and approx_when):
        raise ValueError(
            f"{what}: approx_when is the name of a model method, got {approx_when!r}"
        )
    if (approx is not None or approx_when is not None) and not note.strip():
        raise ValueError(
            f"{what}: a declared approximation needs a note saying why the "
            "hand derivation deviates from the autodiff one"
        )


def hand_block(
    row: str, col: str, *, approx: float | None = None,
    approx_when: str | None = None, note: str = "",
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
    approx_when : str, optional
        Name of a method of the model that returns true on the steps where
        the block is an approximation; elsewhere it is held to the exact
        tolerance.  See the module docstring.
    note : str, optional
        The reason for the approximation.  Required with ``approx`` or
        ``approx_when``: a tolerance without its reason is indistinguishable
        from a bug that was waved through.
    """
    _check_approximation(f"hand_block({row!r}, {col!r})", approx, approx_when, note)
    record = HandBlock(row=row, col=col, approx=approx, approx_when=approx_when, note=note)

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
        validate_condition(cls, hand_block_record(cls, name).approx_when, f"{cls.__name__}.{name}")


def validate_condition(cls: type, condition: str | None, declared_on: str) -> None:
    """Reject an ``approx_when`` naming no method, or one with unknown arguments."""
    if condition is None:
        return
    method = getattr(cls, condition, None)
    if not callable(method):
        raise TypeError(
            f"{declared_on}: approx_when={condition!r} names no method of {cls.__name__}"
        )
    params = tuple(inspect.signature(method).parameters)[1:]
    unknown = [p for p in params if p not in HAND_BLOCK_ARGS]
    if unknown:
        raise TypeError(
            f"{cls.__name__}.{condition}: an approx_when condition takes {unknown}, "
            f"which the framework cannot supply.  Arguments are filled by name "
            f"from {list(HAND_BLOCK_ARGS)}."
        )


def condition_holds(model, condition: str | None, *, state, dlambda, inc) -> bool:
    """Whether the ``approx_when`` condition named ``condition`` holds at a step.

    ``False`` when no condition is declared.
    """
    if condition is None:
        return False
    method = getattr(model, condition)
    available = {"state": state, "dlambda": dlambda, "state_n": inc.state_n, "inc": inc}
    kwargs = {p: available[p] for p in inspect.signature(method).parameters}
    return bool(method(**kwargs))


# ---------------------------------------------------------------------------
# A hand-written tangent
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HandTangent:
    """Record attached to ``user_defined_tangent`` by :func:`hand_tangent`.

    Attributes
    ----------
    approx : float or None
        Largest relative deviation from the exact tangent that is by design.
    approx_when : str or None
        Name of the model method that says on which steps the tangent is an
        approximation.
    note : str
        Why the tangent deviates.
    """

    approx: float | None = None
    approx_when: str | None = None
    note: str = ""


def hand_tangent(
    *, approx: float | None = None, approx_when: str | None = None, note: str = "",
) -> Callable[[_F], _F]:
    """Declare how exact a hand-written ``user_defined_tangent`` is meant to be.

    A hand-written tangent without this declaration is taken to be exact: it
    is checked against central differences, and against the autodiff
    tangent, at the tolerance of the check.  Decorate it when it is not::

        @hand_tangent(
            approx_when="gate_in_transition",
            note="the stagnation gate is differentiated as a constant",
        )
        def user_defined_tangent(self, new, dlambda, inc):
            ...

    Parameters
    ----------
    approx : float, optional
        The tangent agrees with the exact one only to this relative
        deviation (``max|difference| / max|reference|``).
    approx_when : str, optional
        Name of a method of the model that returns true on the steps where
        the tangent is an approximation.  Those steps are left out of the
        comparison and counted in the report — or held to ``approx``, if
        that is given too — and every other step is held to the exact
        tolerance.  The method takes its arguments by name from ``state``,
        ``dlambda``, ``state_n``, ``inc``.
    note : str
        The reason.  Required.

    Notes
    -----
    Read by :func:`~returnmap.verification.check_model`.  The single-step
    checks (:func:`~returnmap.verification.check_tangent`,
    :func:`~returnmap.verification.crosscheck`) take an integrator, not a
    model, and judge by the tolerance they are given.
    """
    if approx is None and approx_when is None:
        raise ValueError(
            "hand_tangent: give approx= or approx_when= — an exact tangent "
            "needs no declaration"
        )
    _check_approximation("hand_tangent", approx, approx_when, note)
    record = HandTangent(approx=approx, approx_when=approx_when, note=note)

    def decorator(fn: _F) -> _F:
        fn.__hand_tangent__ = record  # type: ignore[attr-defined]
        return fn

    return decorator


def hand_tangent_record(cls: type) -> HandTangent | None:
    """The declaration on ``cls.user_defined_tangent``, or ``None`` if it has none.

    Read off the method the class resolves to: an override without the
    decorator is a different derivation and makes no claim.
    """
    record = getattr(getattr(cls, "user_defined_tangent", None), "__hand_tangent__", None)
    return record if isinstance(record, HandTangent) else None
