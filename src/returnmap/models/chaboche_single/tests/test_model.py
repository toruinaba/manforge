"""ChabocheKinematic: what the class declares, and its closed-form state update."""

import numpy as np
import pytest

from returnmap.core import PLANE_STRAIN, PLANE_STRESS, SOLID_3D, UNIAXIAL_1D
from returnmap.models import (
    ChabocheKinematic, ChabocheKinematic1D, ChabocheKinematic3D, ChabocheKinematicPS,
)
from returnmap.verification import make_increment

PARAMS = dict(E=210000.0, nu=0.3, sigma_y0=250.0, C=10000.0, gamma=100.0, Qinf=80.0, b=12.0)
CONCRETE = [ChabocheKinematic3D, ChabocheKinematicPS, ChabocheKinematic1D]


def test_param_names():
    for cls in [ChabocheKinematic, *CONCRETE]:
        assert cls.param_names == ["E", "nu", "sigma_y0", "C", "gamma", "Qinf", "b"]


@pytest.mark.parametrize("cls", CONCRETE)
def test_every_state_is_explicit(cls):
    model = cls(**PARAMS)
    assert model.state_names == ["stress", "alpha", "R", "ep"]
    assert model.implicit_state_names == []


@pytest.mark.parametrize("cls, dimension", [
    (ChabocheKinematic3D, SOLID_3D), (ChabocheKinematicPS, PLANE_STRESS),
    (ChabocheKinematic1D, UNIAXIAL_1D),
])
def test_the_concrete_class_is_the_parent_at_that_dimension(cls, dimension):
    rng = np.random.default_rng(3)
    parent, child = ChabocheKinematic(dimension=dimension, **PARAMS), cls(**PARAMS)
    n = child.ntens
    state = {"stress": 300.0 * rng.standard_normal(n), "alpha": 10.0 * rng.standard_normal(n),
             "R": 20.0, "ep": 0.01}
    assert parent.yield_function(state) == child.yield_function(state)


def test_the_parent_takes_any_dimension():
    model = ChabocheKinematic(dimension=PLANE_STRAIN, **PARAMS)
    assert model.ntens == 4


def test_isotropic_hardening_enlarges_the_yield_surface():
    model = ChabocheKinematic3D(**PARAMS)
    state = {"stress": np.array([300.0, 0, 0, 0, 0, 0]), "alpha": np.zeros(6), "R": 0.0, "ep": 0.0}
    grown = dict(state, R=25.0)
    assert float(model.yield_function(state)) == pytest.approx(50.0)
    assert float(model.yield_function(grown)) == pytest.approx(25.0)


def test_the_update_is_the_backward_euler_closed_form():
    """alpha = (alpha_n + 2/3 C dl n) / (1 + gamma dl),  R = (R_n + b Qinf dl) / (1 + b dl)."""
    model = ChabocheKinematic3D(**PARAMS)
    state_n = {"stress": np.array([280.0, -40.0, 10.0, 60.0, 0.0, -20.0]),
               "alpha": np.array([12.0, -9.0, -3.0, 4.0, 0.0, -1.0]), "R": 15.0, "ep": 0.02}
    dlambda = 3.0e-3
    inc = make_increment(model, state_n, stress_trial=state_n["stress"])
    new = {u.name: u.value for u in model.update_state(state_n, dlambda, inc)}

    n_hat = model.flow(state_n).stress_like
    expected_alpha = (state_n["alpha"] + 2.0 / 3.0 * 10000.0 * dlambda * n_hat) / (1.0 + 100.0 * dlambda)
    np.testing.assert_allclose(new["alpha"], expected_alpha, rtol=1e-14)
    assert float(new["R"]) == pytest.approx((15.0 + 12.0 * 80.0 * dlambda) / (1.0 + 12.0 * dlambda), rel=1e-14)
    assert float(new["ep"]) == pytest.approx(0.02 + dlambda, rel=1e-14)


def test_the_backstress_stays_deviatoric():
    model = ChabocheKinematic3D(**PARAMS)
    state_n = {"stress": np.array([280.0, -40.0, 10.0, 60.0, 0.0, -20.0]),
               "alpha": np.zeros(6), "R": 0.0, "ep": 0.0}
    inc = make_increment(model, state_n, stress_trial=state_n["stress"])
    new = {u.name: u.value for u in model.update_state(state_n, 2.0e-3, inc)}
    assert abs(float(np.sum(new["alpha"][:3]))) < 1e-10
