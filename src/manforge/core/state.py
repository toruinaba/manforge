"""StateField descriptors and State wrapper for material model state variables.

Usage in a model class::

    from manforge.core import Implicit, Explicit, NTENS

    class OWKinematic3D(MaterialModel3D):
        stress = Implicit(shape=NTENS, doc="Cauchy stress")   # NR unknown
        alpha  = Implicit(shape=NTENS, doc="backstress tensor")
        ep     = Implicit(shape=(),    doc="equivalent plastic strain")

The ``Implicit`` / ``Explicit`` descriptors replace the old list-based
``state_names`` / ``implicit_state_names`` declarations.  ``MaterialModel.
__init_subclass__`` collects them via MRO traversal and auto-derives
``cls.state_names`` and ``cls.implicit_state_names`` for internal use.

stress field
------------
σ (stress) is automatically declared as ``Explicit(shape=NTENS, name="stress")``
by ``MaterialModel.__init_subclass__`` if the user does not declare it.  When
declared as ``Implicit``, σ is included as an NR unknown.  The ``implicit_stress``
flag has been removed — use ``stress = Implicit(shape=NTENS)`` instead.

User methods receive state arguments as :class:`State` dict-wrappers.
Access state variables with bracket notation: ``state["stress"]``, ``state["alpha"]``.

Returning residuals / updates
------------------------------
``StateField.__call__`` wraps a computed value as ``StateResidual`` (implicit)
or ``StateUpdate`` (explicit), giving a concise return idiom::

    def state_residual(self, state_new, dlambda, state_n, state_trial):
        ...
        return [self.stress(R_stress), self.alpha(R_alpha), self.ep(R_ep)]

    def update_state(self, dlambda, state_n, state_trial):
        ...
        return [self.stress(sig_new), self.alpha(alpha_new), self.ep(ep_new)]

The framework validates the returned list at the call boundary via
``_validate_state_items``.

``State`` is a thin dict-wrapper that carries field-ordering metadata.
It is created at ``_wrap_state`` call boundaries so that autograd traces
through the raw dict values unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# NTENS sentinel
# ---------------------------------------------------------------------------

class _NtensSentinel:
    """Placeholder shape resolved to ``(model.ntens,)`` at StateField.resolve_shape()."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "NTENS"

    def __reduce__(self):
        return (_get_ntens_sentinel, ())


def _get_ntens_sentinel():
    return NTENS


NTENS = _NtensSentinel()


# ---------------------------------------------------------------------------
# StateField descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StateField:
    """Descriptor for a single material-model state variable.

    Parameters
    ----------
    kind : str
        ``"implicit"`` — treated as an NR unknown (goes into `state_residual`).
        ``"explicit"`` — updated in closed form (goes into `update_state`).
    shape : _NtensSentinel, tuple, or int
        Array shape.  Use ``NTENS`` (from ``manforge.core``) to denote a vector
        of length ``model.ntens`` (resolved at construction time).  Use ``()``
        for a scalar state.  An ``int`` ``n`` is equivalent to ``(n,)``.
        Any other tuple is used directly (e.g. ``(6, 6)``).
        The string ``"ntens"`` is no longer accepted — use ``NTENS``.
    default : callable or None
        Factory ``(model) -> array`` for the initial value.  ``None`` → zeros
        of the resolved shape.
    doc : str
        Short description of the physical meaning.
    name : str
        Set automatically by ``__set_name__`` when declared as a class
        attribute.  Do not pass this manually.
    """

    kind: str                           # "implicit" | "explicit"
    shape: Any                          # NTENS | tuple | int
    default: Callable | None = field(default=None, compare=False)
    doc: str = field(default="", compare=False)
    name: str = field(default="", compare=False)

    def __post_init__(self):
        if self.kind not in ("implicit", "explicit"):
            raise ValueError(f"StateField kind must be 'implicit' or 'explicit', got {self.kind!r}")
        if isinstance(self.shape, str):
            raise TypeError(
                f"StateField shape {self.shape!r} is not valid. "
                "Use the NTENS sentinel instead:\n"
                "    from manforge.core import NTENS\n"
                "    alpha = Implicit(shape=NTENS)"
            )

    def __set_name__(self, owner, attr_name: str):
        object.__setattr__(self, "name", attr_name)

    def __call__(self, value) -> "StateResidual | StateUpdate":
        """Wrap *value* as :class:`StateResidual` (implicit) or :class:`StateUpdate` (explicit).

        Usage inside ``state_residual`` / ``update_state``::

            return [self.alpha(R_alpha), self.ep(R_ep)]   # state_residual
            return [self.alpha(alpha_new), self.ep(ep_new)]  # update_state

        Parameters
        ----------
        value : array-like
            The residual value (implicit) or the updated state value (explicit).
            ArrayBox values produced by autograd are passed through unchanged.

        Returns
        -------
        StateResidual or StateUpdate
        """
        if not self.name:
            raise RuntimeError(
                "StateField.__call__ used before the field was declared as a class "
                "attribute (name not set via __set_name__). "
                "Ensure this field is declared directly on the model class."
            )
        if self.kind == "implicit":
            return StateResidual(name=self.name, value=value)
        return StateUpdate(name=self.name, value=value)

    def resolve_shape(self, ntens: int) -> tuple:
        """Return the concrete shape with NTENS substituted."""
        if self.shape is NTENS:
            return (ntens,)
        if isinstance(self.shape, int):
            return (self.shape,)
        return tuple(self.shape)

    def initial_value(self, model) -> Any:
        """Compute the initial (zero) value for this field."""
        import autograd.numpy as anp
        if self.default is not None:
            return self.default(model)
        shp = self.resolve_shape(model.ntens)
        if shp == ():
            return anp.array(0.0)
        return anp.zeros(shp)


def Implicit(shape=(), default=None, doc="") -> StateField:
    """Create an implicit StateField (state variable solved as NR unknown)."""
    return StateField(kind="implicit", shape=shape, default=default, doc=doc)


def Explicit(shape=(), default=None, doc="") -> StateField:
    """Create an explicit StateField (state variable updated in closed form)."""
    return StateField(kind="explicit", shape=shape, default=default, doc=doc)


# ---------------------------------------------------------------------------
# StateResidual / StateUpdate wrapper types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StateResidual:
    """Per-field residual value for an implicit state variable.

    Produced by calling an implicit :class:`StateField`::

        return [self.alpha(R_alpha), self.ep(R_ep)]

    Attributes
    ----------
    name : str
        Field name (set from the StateField declaration).
    value : array-like
        Residual value.  May be an autograd ArrayBox during NR tracing.
    """

    name: str
    value: Any


@dataclass(frozen=True)
class StateUpdate:
    """Per-field new value for an explicit state variable.

    Produced by calling an explicit :class:`StateField`::

        return [self.alpha(alpha_new), self.ep(ep_new)]

    Attributes
    ----------
    name : str
        Field name (set from the StateField declaration).
    value : array-like
        Updated state value.  May be an autograd ArrayBox during NR tracing.
    """

    name: str
    value: Any


# ---------------------------------------------------------------------------
# DlambdaResidual / DlambdaField — optional Δλ row override
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DlambdaResidual:
    """Optional override for the Δλ row of the NR residual.

    Produced by ``self.dlambda(R)`` inside ``state_residual``.  If present in
    the returned list, the framework uses this as the Δλ row instead of the
    default ``model.yield_function(state) = 0``.

    Use this to implement viscoplasticity (Perzyna), non-standard consistency
    conditions, or any model where the equation determining Δλ differs from the
    yield surface.
    """

    value: Any


class DlambdaField:
    """Framework-provided pseudo-field for the Δλ NR unknown.

    Not a :class:`StateField` — does not appear in ``state_fields`` /
    ``state_names`` / ``implicit_state_names``.  A single stateless singleton
    ``DLAMBDA_FIELD`` is attached to :class:`~manforge.core.material.MaterialModel`
    as the class attribute ``dlambda`` so that users can write
    ``self.dlambda(R_dl)`` inside ``state_residual``.

    Example::

        def state_residual(self, state_new, dlambda, state_n, state_trial):
            f = self.yield_function(state_new)
            R_dl = f - self.eta * dlambda / self.dt   # Perzyna
            return [self.dlambda(R_dl), self.alpha(R_alpha)]
    """

    __slots__ = ()

    def __call__(self, value) -> DlambdaResidual:
        return DlambdaResidual(value=value)


DLAMBDA_FIELD = DlambdaField()


# ---------------------------------------------------------------------------
# Boundary validator
# ---------------------------------------------------------------------------

def _validate_state_items(
    returned,
    expected_names: set,
    expected_cls,
    method_name: str,
    model_name: str,
) -> dict:
    """Validate a list of StateResidual / StateUpdate and return a name→value dict.

    Called at the framework boundary (residual.py / solver.py) after
    ``state_residual`` / ``update_state`` returns.

    Parameters
    ----------
    returned : object
        The value returned by the user method.
    expected_names : set of str
        The field names expected in the returned list.
    expected_cls : type
        ``StateResidual`` for state_residual, ``StateUpdate`` for update_state.
    method_name : str
        ``"state_residual"`` or ``"update_state"`` (used in error messages).
    model_name : str
        Class name of the model (used in error messages).

    Returns
    -------
    dict[str, array-like]
        Mapping from field name to value, suitable for ``_flatten_state``.

    Raises
    ------
    TypeError
        If *returned* is not a list or contains items of the wrong type.
    ValueError
        If the set of names is not exactly *expected_names*.
    """
    if not isinstance(returned, list):
        raise TypeError(
            f"{model_name}.{method_name} must return a list of "
            f"{expected_cls.__name__} (use `self.<field>(value)`), "
            f"got {type(returned).__name__}"
        )
    out: dict = {}
    for item in returned:
        if not isinstance(item, expected_cls):
            raise TypeError(
                f"{model_name}.{method_name}: every item must be "
                f"{expected_cls.__name__} (use `self.<field>(value)`), "
                f"got {type(item).__name__}"
            )
        if item.name in out:
            raise ValueError(
                f"{model_name}.{method_name}: duplicate entry for {item.name!r}"
            )
        out[item.name] = item.value
    actual = set(out.keys())
    if actual != expected_names:
        parts = []
        missing = expected_names - actual
        extra = actual - expected_names
        if missing:
            parts.append(f"missing: {sorted(missing)}")
        if extra:
            parts.append(f"unexpected: {sorted(extra)}")
        raise ValueError(f"{model_name}.{method_name}: {'; '.join(parts)}")
    return out


# ---------------------------------------------------------------------------
# Field collection helper (used by __init_subclass__)
# ---------------------------------------------------------------------------

def collect_state_fields(cls) -> dict[str, StateField]:
    """Collect StateField descriptors from *cls* and its MRO (base→derived).

    Traverses the MRO from the root downward so that subclass declarations
    override parent ones (e.g. a fixture that overrides ``Explicit`` with
    ``Implicit`` to test the vector-NR path).

    Returns
    -------
    dict[str, StateField]
        Ordered by first-seen name in base→derived order; name collision
        resolves to the *most derived* class's value.
    """
    mro_fields: dict[str, StateField] = {}
    for base in reversed(cls.__mro__):
        for attr_name, value in base.__dict__.items():
            if isinstance(value, StateField):
                mro_fields[attr_name] = value
    return mro_fields


# ---------------------------------------------------------------------------
# State wrapper
# ---------------------------------------------------------------------------

class State:
    """Dict-wrapper carrying field-ordering metadata for state variables.

    Wraps the internal ``dict[str, array]`` and preserves declaration-order
    field names in ``_fields``.  The internal dict is preserved unchanged so
    autograd traces through the raw array values without modification.

    Access state variables with bracket notation: ``state["stress"]``,
    ``state["alpha"]``.  Attribute-style dot access is not supported.

    ``State`` objects are created at ``_wrap_state`` call boundaries and
    are never mutated by the framework.
    """

    __slots__ = ("_data", "_fields")

    def __init__(self, data: dict, fields: tuple):
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_fields", fields)

    def __getitem__(self, key: str):
        return self._data[key]

    def __contains__(self, key: str):
        return key in self._data

    def __iter__(self):
        return iter(self._fields)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def with_stress(self, new_stress) -> "State":
        """Return a new State with only the ``stress`` entry replaced."""
        data = dict(self._data)
        data["stress"] = new_stress
        return State(data, self._fields)

    def as_dict(self) -> dict:
        """Return the underlying dict (no copy — values are shared)."""
        return self._data

    def __repr__(self):
        return f"State({self._data!r})"


# ---------------------------------------------------------------------------
# make_state factory helper
# ---------------------------------------------------------------------------

def _make(required_keys: set, factory_name: str, kwargs: dict) -> dict:
    """Validate and assemble a state dict.

    Raises ``TypeError`` listing all missing and unexpected keys so users
    get actionable error messages at the call site (before any autograd trace).
    """
    given = set(kwargs.keys())
    missing = required_keys - given
    extra = given - required_keys
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing keys: {sorted(missing)}")
        if extra:
            parts.append(f"unexpected keys: {sorted(extra)}")
        raise TypeError(f"{factory_name}() — {'; '.join(parts)}")
    return kwargs


# ---------------------------------------------------------------------------
# stress-replacement helper (used by residual.py to build ∂f/∂σ via autograd)
# ---------------------------------------------------------------------------

def _state_with_stress(state, new_stress) -> "State | dict":
    """Return a copy of *state* with only the ``stress`` entry replaced.

    Used by the framework to build ``autograd.grad(lambda s: model.yield_function(
    _state_with_stress(state, s)))(stress)`` without disturbing other state
    entries.  The returned object mirrors the type of *state*:

    - If *state* is a :class:`State`, a new ``State`` is returned.
    - If *state* is a plain ``dict``, a shallow copy with the replaced entry
      is returned.

    The replacement is a *thin wrapper* — existing entries are not copied,
    only the ``"stress"`` key is swapped.  autograd ``ArrayBox`` values in
    other entries pass through unchanged.

    Parameters
    ----------
    state : State or dict
        Current state (must contain the ``"stress"`` key).
    new_stress : array-like
        Replacement stress value (may be an autograd ArrayBox).

    Returns
    -------
    State or dict
        Same type as *state*, with ``state["stress"]`` replaced by *new_stress*.
    """
    if isinstance(state, State):
        data = dict(state.as_dict())
        data["stress"] = new_stress
        return State(data, state._fields)
    # plain dict
    out = dict(state)
    out["stress"] = new_stress
    return out
