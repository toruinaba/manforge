"""StateField descriptors and State wrapper for material model state variables.

Usage in a model class::

    from manforge.core.state import Implicit, Explicit

    class OWKinematic3D(MaterialModel3D):
        alpha = Implicit(shape="ntens", doc="backstress tensor")
        ep    = Implicit(shape=(),      doc="equivalent plastic strain")
        implicit_stress = True

The ``Implicit`` / ``Explicit`` descriptors replace the old list-based
``state_names`` / ``implicit_state_names`` declarations.  ``MaterialModel.
__init_subclass__`` collects them via MRO traversal and auto-derives
``cls.state_names`` and ``cls.implicit_state_names`` for internal use.

``State`` is a thin, immutable dict-wrapper with ``__getattr__`` access.
It is created at ``make_state`` / ``make_residual`` / ``make_update`` call
boundaries so that autograd traces through the raw dict values unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


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
    shape : str or tuple
        Array shape.  Use the string ``"ntens"`` to denote a vector of length
        ``model.ntens`` (resolved at construction time).  Use ``()`` for a
        scalar state.  Any other tuple is used directly (e.g. ``(6, 6)``).
    default : callable or None
        Factory ``(model) -> array`` for the initial value.  ``None`` → zeros
        of the resolved shape.
    doc : str
        Short description of the physical meaning.
    """

    kind: str                           # "implicit" | "explicit"
    shape: Any                          # "ntens" | tuple
    default: Callable | None = field(default=None, compare=False)
    doc: str = field(default="", compare=False)

    def __post_init__(self):
        if self.kind not in ("implicit", "explicit"):
            raise ValueError(f"StateField kind must be 'implicit' or 'explicit', got {self.kind!r}")

    def resolve_shape(self, ntens: int) -> tuple:
        """Return the concrete shape with 'ntens' substituted."""
        if self.shape == "ntens":
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
    # Collect in base→derived order, then reverse so derived wins on dict update.
    mro_fields: dict[str, StateField] = {}
    for base in reversed(cls.__mro__):
        for name, value in base.__dict__.items():
            if isinstance(value, StateField):
                mro_fields[name] = value
    return mro_fields


# ---------------------------------------------------------------------------
# State wrapper
# ---------------------------------------------------------------------------

class State:
    """Immutable dict-wrapper providing attribute-style access to state variables.

    Wraps the internal ``dict[str, array]`` so that user code can write
    ``state.alpha`` instead of ``state["alpha"]``.  The internal dict is
    preserved unchanged, so autograd traces through the raw array values
    without modification.

    ``State`` objects are created at the ``make_state`` / ``make_residual`` /
    ``make_update`` call boundaries and are never mutated.
    """

    __slots__ = ("_data", "_fields")

    def __init__(self, data: dict, fields: tuple[str, ...]):
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_fields", fields)

    # Attribute access — raises AttributeError (not KeyError) on miss.
    def __getattr__(self, name: str):
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(
                f"State has no field {name!r}. Available: {list(self._fields)}"
            )

    # Dict-like access.
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

    def as_dict(self) -> dict:
        """Return the underlying dict (no copy — values are shared)."""
        return self._data

    # Immutable — no assignment.
    def __setattr__(self, name, value):
        raise AttributeError("State is immutable")

    def __repr__(self):
        return f"State({self._data!r})"


# ---------------------------------------------------------------------------
# make_* factory helpers
# ---------------------------------------------------------------------------

def _make(required_keys: set[str], factory_name: str, kwargs: dict) -> dict:
    """Validate and assemble a state / residual / update dict.

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
