"""Unit tests for YUKinematic and its dimension-specialised subclasses."""

import numpy as np
import pytest
import numpy.testing as npt
from manforge.models.yu_kinematic import YUKinematic, YUKinematic3D, YUKinematicPS, YUKinematic1D
from manforge.core.dimension import SOLID_3D, PLANE_STRESS, UNIAXIAL_1D

PARAMS = dict(
    E=206_000, nu=0.3, Y=360.0, C_1=2000.0, C_2=200.0,
    B=435.0, Rsat=255.0, k=26.0, b=66.0, h=0.4, Ea=159_000, xi=61.0,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# U-1: class definition regression
# ---------------------------------------------------------------------------

def test_param_names():
    model = YUKinematic3D(**PARAMS)
    assert "B" in model.param_names
    assert len(model.param_names) == 12
    assert set(model.param_names) == {
        "E", "nu", "Y", "B", "C_1", "C_2", "Rsat", "k", "b", "h", "Ea", "xi"
    }


def test_state_names():
    model = YUKinematic3D(**PARAMS)
    assert set(model.state_names) == {
        "stress", "theta", "beta", "R", "q", "r", "eps_eq", "theta_max"
    }
    assert set(model.implicit_state_names) == {"stress", "theta", "beta"}


@pytest.mark.parametrize("cls,dim", [
    (YUKinematic3D,  SOLID_3D),
    (YUKinematicPS,  PLANE_STRESS),
    (YUKinematic1D,  UNIAXIAL_1D),
])
def test_subclass_dimension(cls, dim):
    model = cls(**PARAMS)
    assert model.dimension is dim


# ---------------------------------------------------------------------------
# U-2: _calc_E_factor
# ---------------------------------------------------------------------------

def test_calc_E_factor_at_zero():
    model = YUKinematic3D(**PARAMS)
    assert model._calc_E_factor(0.0) == pytest.approx(1.0)


def test_calc_E_factor_asymptote():
    model = YUKinematic3D(**PARAMS)
    assert model._calc_E_factor(1e6) == pytest.approx(PARAMS["Ea"] / PARAMS["E"], abs=1e-6)


def test_calc_E_factor_monotone():
    model = YUKinematic3D(**PARAMS)
    eps_vals = np.linspace(0.0, 0.1, 20)
    factors = np.array([float(model._calc_E_factor(e)) for e in eps_vals])
    assert np.all(np.diff(factors) <= 0)


# ---------------------------------------------------------------------------
# U-7: yield_function scaling
# ---------------------------------------------------------------------------

def _make_state(model, stress, theta=None, beta=None):
    state = model.initial_state()
    state["stress"] = stress
    state["theta"] = theta if theta is not None else np.zeros_like(stress)
    state["beta"]  = beta  if beta  is not None else np.zeros_like(stress)
    return state


def test_yield_function_elastic_stress_is_negative():
    """Zero stress with zero backstress should be well inside yield surface."""
    model = YUKinematic3D(**PARAMS)
    state = _make_state(model, np.zeros(6))
    assert model.yield_function(state) == pytest.approx(-PARAMS["Y"])


def test_yield_function_on_surface_is_zero():
    """Uniaxial stress exactly equal to Y (von Mises) should give f=0."""
    model = YUKinematic3D(**PARAMS)
    Y = PARAMS["Y"]
    sigma = np.array([Y, 0.0, 0.0, 0.0, 0.0, 0.0])
    state = _make_state(model, sigma)
    assert model.yield_function(state) == pytest.approx(0.0, abs=1e-8)


def test_yield_function_outside_is_positive():
    model = YUKinematic3D(**PARAMS)
    Y = PARAMS["Y"]
    sigma = np.array([2 * Y, 0.0, 0.0, 0.0, 0.0, 0.0])
    state = _make_state(model, sigma)
    assert model.yield_function(state) > 0


@pytest.mark.parametrize("cls,ntens", [
    (YUKinematicPS, 3),
    (YUKinematic1D, 1),
])
def test_yield_function_subclasses(cls, ntens):
    model = cls(**PARAMS)
    Y = PARAMS["Y"]
    sigma = np.zeros(ntens)
    state = _make_state(model, sigma)
    assert model.yield_function(state) == pytest.approx(-Y)
