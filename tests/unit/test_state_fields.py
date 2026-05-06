"""Tests for StateField descriptors, State wrapper, and validation utilities.

Covers:
- Field collection via MRO (base→derived, subclass overrides parent)
- Shape resolution: NTENS sentinel, scalar, explicit tuple, int
- "ntens" string input raises TypeError (migration)
- StateField.__call__ produces StateResidual / StateUpdate
- _validate_state_items boundary validation
- State attribute access and immutability
- _make helper
- make_state on a real model
- Migration error: non-empty list state_names / implicit_state_names raise TypeError
"""

import numpy as np
import pytest
import autograd.numpy as anp

from manforge.core.state import (
    Implicit, Explicit, StateField, collect_state_fields, State, _make, NTENS, SCALAR,
    StateResidual, StateUpdate, _validate_state_items,
)
from manforge.core.material import MaterialModel3D, MaterialModel1D
from manforge.core.dimension import SOLID_3D, UNIAXIAL_1D


# ---------------------------------------------------------------------------
# Helpers — minimal concrete models
# ---------------------------------------------------------------------------

class _EP(MaterialModel3D):
    """Single explicit ep field."""
    param_names = []
    ep = Explicit(shape=(), doc="plastic strain")

    def yield_function(self, state):
        return anp.array(0.0)

    def update_state(self, dlambda, state_n, state_trial):
        return [self.ep(state_n["ep"] + dlambda)]


class _AlphaEP(MaterialModel3D):
    """alpha (Implicit) and ep (Explicit) mixed model."""
    param_names = []
    alpha = Implicit(shape=NTENS, doc="backstress")
    ep = Explicit(shape=(), doc="plastic strain")

    def yield_function(self, state):
        return anp.array(0.0)

    def update_state(self, dlambda, state_n, state_trial):
        return [self.ep(state_n["ep"] + dlambda)]

    def state_residual(self, state_new, dlambda, state_n, state_trial, *, stress_trial):
        return [self.alpha(state_new["alpha"] - state_n["alpha"])]


class _AllImplicit(MaterialModel3D):
    """Both alpha and ep implicit."""
    param_names = []
    alpha = Implicit(shape=NTENS)
    ep = Implicit(shape=())

    def yield_function(self, state):
        return anp.array(0.0)

    def state_residual(self, state_new, dlambda, state_n, state_trial, *, stress_trial):
        return [
            self.alpha(state_new["alpha"] - state_n["alpha"]),
            self.ep(state_new["ep"] - state_n["ep"]),
        ]


# ---------------------------------------------------------------------------
# Field collection
# ---------------------------------------------------------------------------

def test_collect_state_fields_order():
    """Fields must be collected in base→derived declaration order."""
    fields = collect_state_fields(_AlphaEP)
    # stress is auto-added first by __init_subclass__; collect_state_fields returns only declared fields
    assert "alpha" in fields and "ep" in fields


def test_collect_state_fields_kinds():
    fields = collect_state_fields(_AlphaEP)
    assert fields["alpha"].kind == "implicit"
    assert fields["ep"].kind == "explicit"


def test_state_names_derived_correctly():
    # stress is auto-prepended by __init_subclass__
    assert "stress" in _AlphaEP.state_names
    assert "alpha" in _AlphaEP.state_names
    assert "ep" in _AlphaEP.state_names
    assert "stress" in _EP.state_names
    assert "ep" in _EP.state_names
    assert "stress" in _AllImplicit.state_names
    assert "alpha" in _AllImplicit.state_names
    assert "ep" in _AllImplicit.state_names


def test_implicit_state_names_derived_correctly():
    # stress is Explicit by default; alpha is Implicit
    assert "alpha" in _AlphaEP.implicit_state_names
    assert "stress" not in _AlphaEP.implicit_state_names
    assert _EP.implicit_state_names == []
    assert "alpha" in _AllImplicit.implicit_state_names
    assert "ep" in _AllImplicit.implicit_state_names
    assert "stress" not in _AllImplicit.implicit_state_names


def test_mro_override_explicit_to_implicit():
    """Subclass can override a parent Explicit field with Implicit."""
    class _Override(_AlphaEP):
        ep = Implicit(shape=(), doc="overridden to implicit")

        def state_residual(self, state_new, dlambda, state_n, state_trial, *, stress_trial):
            return [
                self.alpha(state_new["alpha"] - state_n["alpha"]),
                self.ep(state_new["ep"] - state_n["ep"]),
            ]

    assert "alpha" in _Override.implicit_state_names
    assert "ep" in _Override.implicit_state_names
    assert "stress" not in _Override.implicit_state_names
    assert "alpha" in _Override.state_names
    assert "ep" in _Override.state_names
    assert "stress" in _Override.state_names


def test_mro_override_implicit_to_explicit():
    """Subclass can override a parent Implicit field with Explicit."""
    class _Override(_AllImplicit):
        alpha = Explicit(shape=NTENS, doc="overridden to explicit")

        def update_state(self, dlambda, state_n, state_trial):
            return [self.alpha(state_n["alpha"])]

    assert "ep" in _Override.implicit_state_names
    assert "alpha" not in _Override.implicit_state_names
    assert "alpha" in _Override.state_names
    assert "ep" in _Override.state_names


# ---------------------------------------------------------------------------
# Shape resolution
# ---------------------------------------------------------------------------

def test_ntens_sentinel_resolves_to_6_for_solid_3d():
    f = Implicit(shape=NTENS)
    assert f.resolve_shape(ntens=6) == (6,)


def test_ntens_sentinel_resolves_to_1_for_1d():
    f = Explicit(shape=NTENS)
    assert f.resolve_shape(ntens=1) == (1,)


def test_ntens_string_raises_typeerror():
    """The old 'ntens' magic string must raise TypeError with migration hint."""
    with pytest.raises(TypeError, match="NTENS sentinel"):
        Implicit(shape="ntens")


def test_scalar_shape_resolves_to_empty_tuple():
    f = Explicit(shape=())
    assert f.resolve_shape(ntens=6) == ()


def test_scalar_sentinel_resolves_to_empty_tuple():
    f = Explicit(shape=SCALAR)
    assert f.resolve_shape(ntens=6) == ()


def test_scalar_sentinel_initial_value_is_zero_d():
    model = _EP(SOLID_3D)
    f = Explicit(shape=SCALAR)
    val = f.initial_value(model)
    assert np.asarray(val).shape == ()
    assert float(val) == 0.0


def test_int_shape_resolves_to_singleton_tuple():
    f = Explicit(shape=3)
    assert f.resolve_shape(ntens=6) == (3,)


def test_tuple_shape_passed_through():
    f = Explicit(shape=(3, 3))
    assert f.resolve_shape(ntens=6) == (3, 3)


# ---------------------------------------------------------------------------
# initial_value
# ---------------------------------------------------------------------------

def test_initial_value_ntens_gives_zeros_array():
    model = _AlphaEP(SOLID_3D)
    val = _AlphaEP.state_fields["alpha"].initial_value(model)
    assert np.allclose(val, np.zeros(6))


def test_initial_value_scalar_gives_zero():
    model = _EP(SOLID_3D)
    val = _EP.state_fields["ep"].initial_value(model)
    assert float(val) == 0.0


def test_initial_state_uses_field_shapes():
    model = _AlphaEP(SOLID_3D)
    state = model.initial_state()
    assert np.asarray(state["stress"]).shape == (6,)
    assert np.asarray(state["alpha"]).shape == (6,)
    assert np.asarray(state["ep"]).shape == ()


# ---------------------------------------------------------------------------
# StateField.__call__ (kind dispatch)
# ---------------------------------------------------------------------------

def test_implicit_field_call_returns_state_residual():
    model = _AllImplicit(SOLID_3D)
    result = model.alpha(np.zeros(6))
    assert isinstance(result, StateResidual)
    assert result.name == "alpha"
    assert np.allclose(result.value, np.zeros(6))


def test_explicit_field_call_returns_state_update():
    model = _EP(SOLID_3D)
    result = model.ep(np.array(0.1))
    assert isinstance(result, StateUpdate)
    assert result.name == "ep"
    assert float(result.value) == pytest.approx(0.1)


def test_field_call_name_set_from_class_declaration():
    """__set_name__ must populate the name attribute correctly."""
    model = _AlphaEP(SOLID_3D)
    assert _AlphaEP.state_fields["alpha"].name == "alpha"
    assert _AlphaEP.state_fields["ep"].name == "ep"


def test_state_residual_is_frozen():
    r = StateResidual(name="alpha", value=np.zeros(6))
    with pytest.raises(Exception):
        r.name = "other"


def test_state_update_is_frozen():
    u = StateUpdate(name="ep", value=np.array(0.0))
    with pytest.raises(Exception):
        u.name = "other"


def test_field_call_without_set_name_raises():
    """A StateField not declared as a class attribute raises RuntimeError."""
    f = Implicit(shape=())
    with pytest.raises(RuntimeError, match="__set_name__"):
        f(np.array(0.0))


# ---------------------------------------------------------------------------
# _validate_state_items boundary validator
# ---------------------------------------------------------------------------

def test_validate_state_items_normal_residual():
    model = _AllImplicit(SOLID_3D)
    items = [model.alpha(np.ones(6)), model.ep(np.array(0.5))]
    result = _validate_state_items(items, {"alpha", "ep"}, StateResidual, "state_residual", "TestModel")
    assert set(result.keys()) == {"alpha", "ep"}
    assert np.allclose(result["alpha"], np.ones(6))


def test_validate_state_items_wrong_type_update_in_residual():
    """StateUpdate items returned where StateResidual expected → TypeError."""
    model = _EP(SOLID_3D)
    items = [model.ep(np.array(0.5))]  # StateUpdate, not StateResidual
    with pytest.raises(TypeError, match="StateResidual"):
        _validate_state_items(items, {"ep"}, StateResidual, "state_residual", "TestModel")


def test_validate_state_items_wrong_type_residual_in_update():
    """StateResidual items returned where StateUpdate expected → TypeError."""
    model = _AllImplicit(SOLID_3D)
    items = [model.alpha(np.zeros(6))]  # StateResidual, not StateUpdate
    with pytest.raises(TypeError, match="StateUpdate"):
        _validate_state_items(items, {"alpha"}, StateUpdate, "update_state", "TestModel")


def test_validate_state_items_not_list_raises():
    with pytest.raises(TypeError, match="must return a list"):
        _validate_state_items({"alpha": np.zeros(6)}, {"alpha"}, StateResidual, "state_residual", "M")


def test_validate_state_items_duplicate_raises():
    model = _AllImplicit(SOLID_3D)
    items = [model.alpha(np.zeros(6)), model.alpha(np.zeros(6))]
    with pytest.raises(ValueError, match="duplicate"):
        _validate_state_items(items, {"alpha", "ep"}, StateResidual, "state_residual", "M")


def test_validate_state_items_missing_raises():
    model = _AllImplicit(SOLID_3D)
    items = [model.alpha(np.zeros(6))]  # ep missing
    with pytest.raises(ValueError, match="missing"):
        _validate_state_items(items, {"alpha", "ep"}, StateResidual, "state_residual", "M")


def test_validate_state_items_extra_raises():
    model = _AllImplicit(SOLID_3D)
    items = [model.alpha(np.zeros(6)), model.ep(np.array(0.0))]
    with pytest.raises(ValueError, match="unexpected"):
        _validate_state_items(items, {"alpha"}, StateResidual, "state_residual", "M")


# ---------------------------------------------------------------------------
# State wrapper
# ---------------------------------------------------------------------------

def test_state_getitem():
    s = State({"alpha": np.zeros(6), "ep": np.array(1.0)}, ("alpha", "ep"))
    assert np.allclose(s["alpha"], np.zeros(6))
    assert float(s["ep"]) == 1.0


def test_state_as_dict_returns_same_object():
    data = {"ep": np.array(0.0)}
    s = State(data, ("ep",))
    assert s.as_dict() is data


def test_state_contains():
    s = State({"ep": np.array(0.0)}, ("ep",))
    assert "ep" in s
    assert "alpha" not in s


def test_state_iter_yields_field_names():
    s = State({"alpha": np.zeros(6), "ep": np.array(0.0)}, ("alpha", "ep"))
    assert list(s) == ["alpha", "ep"]


def test_state_identity_no_copy():
    arr = np.zeros(6)
    data = {"alpha": arr}
    s = State(data, ("alpha",))
    assert s["alpha"] is arr


# ---------------------------------------------------------------------------
# _make helper
# ---------------------------------------------------------------------------

def test_make_correct_keys():
    result = _make({"alpha", "ep"}, "test", {"alpha": 1, "ep": 2})
    assert result == {"alpha": 1, "ep": 2}


def test_make_missing_key_raises():
    with pytest.raises(TypeError, match="missing keys"):
        _make({"alpha", "ep"}, "test", {"alpha": 1})


def test_make_extra_key_raises():
    with pytest.raises(TypeError, match="unexpected keys"):
        _make({"ep"}, "test", {"ep": 1, "extra": 2})


def test_make_both_missing_and_extra_raises():
    with pytest.raises(TypeError) as exc:
        _make({"alpha", "ep"}, "test", {"ep": 1, "foo": 2})
    msg = str(exc.value)
    assert "missing" in msg
    assert "unexpected" in msg


# ---------------------------------------------------------------------------
# make_state on a real model
# ---------------------------------------------------------------------------

def test_make_state_all_keys_required():
    model = _AlphaEP(SOLID_3D)
    s = model.make_state(stress=np.zeros(6), alpha=np.zeros(6), ep=np.array(0.0))
    assert isinstance(s, State)
    assert np.allclose(s["alpha"], np.zeros(6))
    assert np.allclose(s["stress"], np.zeros(6))


def test_make_state_missing_raises():
    model = _AlphaEP(SOLID_3D)
    with pytest.raises(TypeError, match="missing keys"):
        model.make_state(alpha=np.zeros(6))


