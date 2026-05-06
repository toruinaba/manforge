"""Validation tests for self.dlambda(R) API: error paths and boundary checks."""

import pytest
import autograd.numpy as anp

from manforge.core.state import (
    Implicit, Explicit, NTENS,
    DlambdaResidual, DlambdaField, DLAMBDA_FIELD,
    StateResidual, StateUpdate,
)
from manforge.core.material import MaterialModel3D, MaterialModel1D


# ---------------------------------------------------------------------------
# Minimal models for validation tests
# ---------------------------------------------------------------------------

_LAM = 115384.0
_MU = 76923.0


class _NoStateModel(MaterialModel3D):
    """No state fields beyond stress (scalar-NR path)."""
    param_names = ["sigma_y0"]

    def __init__(self, *, sigma_y0):
        super().__init__()
        self.sigma_y0 = sigma_y0

    def yield_function(self, state):
        return self._vonmises(state["stress"]) - self.sigma_y0

    def elastic_stiffness(self, state):
        return self.isotropic_C(_LAM, _MU)


class _DlambdaOnlyModel(MaterialModel3D):
    """state_residual returns only self.dlambda(R) — no other implicit states."""
    param_names = ["sigma_y0"]

    def __init__(self, *, sigma_y0):
        super().__init__()
        self.sigma_y0 = sigma_y0

    def yield_function(self, state):
        return self._vonmises(state["stress"]) - self.sigma_y0

    def elastic_stiffness(self, state):
        return self.isotropic_C(_LAM, _MU)

    def state_residual(self, state_new, dlambda, state_n, state_trial, *, stress_trial):
        return [self.dlambda(self.yield_function(state_new))]


class _AlphaModel(MaterialModel3D):
    """Vector-NR: alpha Implicit."""
    param_names = ["sigma_y0", "H_k"]
    alpha = Implicit(shape=NTENS, doc="backstress")

    def __init__(self, *, sigma_y0, H_k):
        super().__init__()
        self.sigma_y0 = sigma_y0
        self.H_k = H_k

    def yield_function(self, state):
        xi = state["stress"] - state["alpha"]
        return self._vonmises(xi) - self.sigma_y0

    def elastic_stiffness(self, state):
        return self.isotropic_C(_LAM, _MU)

    def state_residual(self, state_new, dlambda, state_n, state_trial, *, stress_trial):
        R_alpha = state_new["alpha"] - state_n["alpha"]
        return [self.alpha(R_alpha)]


class _DuplicateDlambdaModel(MaterialModel3D):
    """Intentionally returns self.dlambda(...) twice — should error."""
    param_names = ["sigma_y0"]
    alpha = Implicit(shape=NTENS, doc="backstress")

    def __init__(self, *, sigma_y0):
        super().__init__()
        self.sigma_y0 = sigma_y0

    def yield_function(self, state):
        return self._vonmises(state["stress"]) - self.sigma_y0

    def elastic_stiffness(self, state):
        return self.isotropic_C(_LAM, _MU)

    def state_residual(self, state_new, dlambda, state_n, state_trial, *, stress_trial):
        R_alpha = state_new["alpha"] - state_n["alpha"]
        return [
            self.alpha(R_alpha),
            self.dlambda(self.yield_function(state_new)),
            self.dlambda(self.yield_function(state_new)),  # duplicate
        ]


class _BadItemModel(MaterialModel3D):
    """state_residual returns a non-StateResidual / non-DlambdaResidual item."""
    param_names = ["sigma_y0"]
    alpha = Implicit(shape=NTENS, doc="backstress")

    def __init__(self, *, sigma_y0):
        super().__init__()
        self.sigma_y0 = sigma_y0

    def yield_function(self, state):
        return self._vonmises(state["stress"]) - self.sigma_y0

    def elastic_stiffness(self, state):
        return self.isotropic_C(_LAM, _MU)

    def state_residual(self, state_new, dlambda, state_n, state_trial, *, stress_trial):
        return [42]  # wrong type


class _DlambdaInUpdateState(MaterialModel3D):
    """update_state mistakenly returns self.dlambda(R) — should error."""
    param_names = ["sigma_y0"]
    ep = Explicit(shape=(), doc="ep")

    def __init__(self, *, sigma_y0):
        super().__init__()
        self.sigma_y0 = sigma_y0

    def yield_function(self, state):
        return self._vonmises(state["stress"]) - self.sigma_y0

    def elastic_stiffness(self, state):
        return self.isotropic_C(_LAM, _MU)

    def update_state(self, dlambda, state_n, state_trial):
        return [
            self.ep(state_n["ep"] + dlambda),
            self.dlambda(anp.array(0.0)),  # not allowed
        ]


# ---------------------------------------------------------------------------
# Tests: DlambdaField / DlambdaResidual basics
# ---------------------------------------------------------------------------

def test_dlambda_field_is_attached_to_model():
    model = _NoStateModel(sigma_y0=250.0)
    assert hasattr(model, "dlambda")
    assert isinstance(model.dlambda, DlambdaField)


def test_dlambda_field_call_produces_dlambda_residual():
    model = _NoStateModel(sigma_y0=250.0)
    result = model.dlambda(anp.array(1.5))
    assert isinstance(result, DlambdaResidual)
    assert float(result.value) == pytest.approx(1.5)


def test_dlambda_field_singleton():
    assert isinstance(DLAMBDA_FIELD, DlambdaField)
    model_a = _NoStateModel(sigma_y0=250.0)
    model_b = _AlphaModel(sigma_y0=250.0, H_k=1000.0)
    assert model_a.dlambda is model_b.dlambda is DLAMBDA_FIELD


def test_dlambda_not_in_state_fields():
    model = _NoStateModel(sigma_y0=250.0)
    assert "dlambda" not in model.state_fields
    assert "dlambda" not in model.state_names
    assert "dlambda" not in model.implicit_state_names


# ---------------------------------------------------------------------------
# Tests: update_state must not accept self.dlambda(R)
# ---------------------------------------------------------------------------

def test_update_state_rejects_dlambda_residual():
    """update_state returning self.dlambda(...) raises TypeError via the integrator."""
    from manforge.simulation.integrator import PythonIntegrator
    model = _DlambdaInUpdateState(sigma_y0=250.0)
    state_n = model.initial_state()
    # Uniaxial strain that triggers yielding (sigma_vm > 250 MPa at E≈200GPa)
    deps = anp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(TypeError, match="self.dlambda.*not allowed in update_state"):
        PythonIntegrator(model).stress_update(deps, anp.zeros(6), state_n)


# ---------------------------------------------------------------------------
# Tests: state_residual duplicate self.dlambda(...) raises ValueError
# ---------------------------------------------------------------------------

def test_state_residual_duplicate_dlambda_raises():
    """state_residual returning self.dlambda(...) twice raises ValueError via integrator."""
    from manforge.simulation.integrator import PythonIntegrator
    model = _DuplicateDlambdaModel(sigma_y0=250.0)
    state_n = model.initial_state()
    deps = anp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="duplicate self.dlambda"):
        PythonIntegrator(model).stress_update(deps, anp.zeros(6), state_n)


# ---------------------------------------------------------------------------
# Tests: state_residual bad item type raises TypeError
# ---------------------------------------------------------------------------

def test_state_residual_bad_item_raises():
    """state_residual returning a non-StateResidual item raises TypeError via integrator."""
    from manforge.simulation.integrator import PythonIntegrator
    model = _BadItemModel(sigma_y0=250.0)
    state_n = model.initial_state()
    deps = anp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(TypeError):
        PythonIntegrator(model).stress_update(deps, anp.zeros(6), state_n)


# ---------------------------------------------------------------------------
# Tests: state_residual dlambda extraction returns (dict, r_dl) correctly
# ---------------------------------------------------------------------------

def test_state_residual_with_dlambda_override_extracts_correctly():
    """state_residual returning only self.dlambda(R) → empty dict + non-None r_dl."""
    from manforge.core.state import _validate_state_items, StateResidual
    model = _DlambdaOnlyModel(sigma_y0=250.0)
    state_n = model.make_state(stress=anp.zeros(6))
    from manforge.simulation._residual import _wrap_state
    state_wrapped = _wrap_state(state_n, model)
    returned = model.state_residual(
        state_wrapped, anp.array(0.0), state_wrapped, state_wrapped,
        stress_trial=anp.zeros(6),
    )
    result_dict, r_dl = _validate_state_items(
        returned, set(), StateResidual, "state_residual", "_DlambdaOnlyModel",
        extract_dlambda=True,
    )
    assert result_dict == {}
    assert r_dl is not None


def test_state_residual_without_dlambda_extracts_none():
    """state_residual returning only alpha → dict with 'alpha' + None r_dl."""
    from manforge.core.state import _validate_state_items, StateResidual
    model = _AlphaModel(sigma_y0=250.0, H_k=1000.0)
    state_n = model.make_state(stress=anp.zeros(6), alpha=anp.zeros(6))
    from manforge.simulation._residual import _wrap_state
    state_wrapped = _wrap_state(state_n, model)
    returned = model.state_residual(
        state_wrapped, anp.array(0.0), state_wrapped, state_wrapped,
        stress_trial=anp.zeros(6),
    )
    result_dict, r_dl = _validate_state_items(
        returned, {"alpha"}, StateResidual, "state_residual", "_AlphaModel",
        extract_dlambda=True,
    )
    assert "alpha" in result_dict
    assert r_dl is None
