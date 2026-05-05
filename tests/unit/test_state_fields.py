"""Tests for StateField descriptors, State wrapper, and make_* factories.

Covers:
- Field collection via MRO (base→derived, subclass overrides parent)
- Shape resolution: "ntens" → (ntens,), scalar, explicit tuple
- make_state / make_residual / make_update key validation
- State attribute access and immutability
- Migration error: non-empty list state_names / implicit_state_names raise TypeError
"""

import numpy as np
import pytest
import autograd.numpy as anp

from manforge.core.state import Implicit, Explicit, StateField, collect_state_fields, State, _make
from manforge.core.material import MaterialModel3D, MaterialModel1D
from manforge.core.stress_state import SOLID_3D, UNIAXIAL_1D


# ---------------------------------------------------------------------------
# Helpers — minimal concrete models
# ---------------------------------------------------------------------------

class _EP(MaterialModel3D):
    """Single explicit ep field."""
    param_names = []
    ep = Explicit(shape=(), doc="plastic strain")

    def yield_function(self, stress, state):
        return anp.array(0.0)

    def update_state(self, dlambda, stress, state):
        return {"ep": state["ep"] + dlambda}


class _AlphaEP(MaterialModel3D):
    """alpha (Implicit) and ep (Explicit) mixed model."""
    param_names = []
    alpha = Implicit(shape="ntens", doc="backstress")
    ep = Explicit(shape=(), doc="plastic strain")

    def yield_function(self, stress, state):
        return anp.array(0.0)

    def update_state(self, dlambda, stress, state):
        return {"ep": state["ep"] + dlambda}

    def state_residual(self, state_new, dlambda, stress, state_n):
        return {"alpha": state_new["alpha"] - state_n["alpha"]}


class _AllImplicit(MaterialModel3D):
    """Both alpha and ep implicit."""
    param_names = []
    alpha = Implicit(shape="ntens")
    ep = Implicit(shape=())

    def yield_function(self, stress, state):
        return anp.array(0.0)

    def state_residual(self, state_new, dlambda, stress, state_n):
        return {"alpha": state_new["alpha"] - state_n["alpha"],
                "ep": state_new["ep"] - state_n["ep"]}


# ---------------------------------------------------------------------------
# Field collection
# ---------------------------------------------------------------------------

def test_collect_state_fields_order():
    """Fields must be collected in base→derived declaration order."""
    fields = collect_state_fields(_AlphaEP)
    assert list(fields.keys()) == ["alpha", "ep"]


def test_collect_state_fields_kinds():
    fields = collect_state_fields(_AlphaEP)
    assert fields["alpha"].kind == "implicit"
    assert fields["ep"].kind == "explicit"


def test_state_names_derived_correctly():
    assert _AlphaEP.state_names == ["alpha", "ep"]
    assert _EP.state_names == ["ep"]
    assert _AllImplicit.state_names == ["alpha", "ep"]


def test_implicit_state_names_derived_correctly():
    assert _AlphaEP.implicit_state_names == ["alpha"]
    assert _EP.implicit_state_names == []
    assert _AllImplicit.implicit_state_names == ["alpha", "ep"]


def test_mro_override_explicit_to_implicit():
    """Subclass can override a parent Explicit field with Implicit."""
    class _Override(_AlphaEP):
        ep = Implicit(shape=(), doc="overridden to implicit")

        def state_residual(self, state_new, dlambda, stress, state_n):
            return {
                "alpha": state_new["alpha"] - state_n["alpha"],
                "ep": state_new["ep"] - state_n["ep"],
            }

    assert _Override.implicit_state_names == ["alpha", "ep"]
    assert _Override.state_names == ["alpha", "ep"]


def test_mro_override_implicit_to_explicit():
    """Subclass can override a parent Implicit field with Explicit."""
    class _Override(_AllImplicit):
        alpha = Explicit(shape="ntens", doc="overridden to explicit")

        def update_state(self, dlambda, stress, state):
            return {"alpha": state["alpha"]}

    assert _Override.implicit_state_names == ["ep"]
    assert _Override.state_names == ["alpha", "ep"]


# ---------------------------------------------------------------------------
# Shape resolution
# ---------------------------------------------------------------------------

def test_ntens_shape_resolves_to_6_for_solid_3d():
    f = Implicit(shape="ntens")
    assert f.resolve_shape(ntens=6) == (6,)


def test_ntens_shape_resolves_to_1_for_1d():
    f = Explicit(shape="ntens")
    assert f.resolve_shape(ntens=1) == (1,)


def test_scalar_shape_resolves_to_empty_tuple():
    f = Explicit(shape=())
    assert f.resolve_shape(ntens=6) == ()


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
    assert np.asarray(state["alpha"]).shape == (6,)
    assert np.asarray(state["ep"]).shape == ()


# ---------------------------------------------------------------------------
# State wrapper
# ---------------------------------------------------------------------------

def test_state_getattr():
    s = State({"alpha": np.zeros(6), "ep": np.array(0.0)}, ("alpha", "ep"))
    assert np.allclose(s.alpha, np.zeros(6))
    assert float(s.ep) == 0.0


def test_state_getitem():
    s = State({"ep": np.array(1.0)}, ("ep",))
    assert float(s["ep"]) == 1.0


def test_state_getattr_missing_raises_attribute_error():
    s = State({"ep": np.array(0.0)}, ("ep",))
    with pytest.raises(AttributeError, match="no field 'alpha'"):
        _ = s.alpha


def test_state_is_immutable():
    s = State({"ep": np.array(0.0)}, ("ep",))
    with pytest.raises(AttributeError):
        s.ep = np.array(1.0)


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
    assert s.alpha is arr


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
# make_state / make_residual / make_update on a real model
# ---------------------------------------------------------------------------

def test_make_state_all_keys_required():
    model = _AlphaEP(SOLID_3D)
    s = model.make_state(alpha=np.zeros(6), ep=np.array(0.0))
    assert isinstance(s, State)
    assert np.allclose(s.alpha, np.zeros(6))


def test_make_state_missing_raises():
    model = _AlphaEP(SOLID_3D)
    with pytest.raises(TypeError, match="missing keys"):
        model.make_state(alpha=np.zeros(6))


def test_make_residual_implicit_keys_only():
    model = _AlphaEP(SOLID_3D)
    r = model.make_residual(alpha=np.zeros(6))
    assert list(r.keys()) == ["alpha"]


def test_make_residual_extra_key_raises():
    model = _AlphaEP(SOLID_3D)
    with pytest.raises(TypeError, match="unexpected keys"):
        model.make_residual(alpha=np.zeros(6), ep=np.array(0.0))


def test_make_update_explicit_keys_only():
    model = _AlphaEP(SOLID_3D)
    r = model.make_update(ep=np.array(0.0))
    assert list(r.keys()) == ["ep"]


def test_make_update_extra_key_raises():
    model = _AlphaEP(SOLID_3D)
    with pytest.raises(TypeError, match="unexpected keys"):
        model.make_update(ep=np.array(0.0), alpha=np.zeros(6))


# ---------------------------------------------------------------------------
# Migration errors
# ---------------------------------------------------------------------------

def test_list_state_names_non_empty_raises():
    with pytest.raises(TypeError, match="list-based state_names has been removed"):
        class Bad(MaterialModel3D):
            param_names = []
            state_names = ["ep"]

            def yield_function(self, stress, state):
                return anp.array(0.0)

            def update_state(self, dlambda, stress, state):
                return {"ep": state["ep"] + dlambda}


def test_list_implicit_state_names_non_empty_raises():
    with pytest.raises(TypeError, match="list-based implicit_state_names has been removed"):
        class Bad(MaterialModel3D):
            param_names = []
            implicit_state_names = ["alpha"]
            alpha = Implicit(shape="ntens")

            def yield_function(self, stress, state):
                return anp.array(0.0)

            def state_residual(self, state_new, dlambda, stress, state_n):
                return {"alpha": state_new["alpha"]}


def test_empty_list_state_names_allowed():
    """state_names=[] is a no-op (backwards compat for stateless stubs)."""
    class OK(MaterialModel3D):
        param_names = []
        state_names = []

        def yield_function(self, stress, state):
            return anp.array(0.0)

    assert OK.state_names == []
