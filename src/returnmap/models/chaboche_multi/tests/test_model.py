"""ChabocheKinematicMulti: what the class declares, and its closed-form state update."""

import numpy as np
import pytest

from returnmap.models import (
    ChabocheKinematicMulti, ChabocheKinematicMulti1D, ChabocheKinematicMulti3D,
    ChabocheKinematicMultiPS,
)
from returnmap.verification import make_increment

PARAMS = dict(
    E=210000.0, nu=0.3, sigma_y0=250.0,
    C1=80000.0, gamma1=800.0, C2=10000.0, gamma2=100.0, C3=1000.0, gamma3=10.0,
    Qinf1=40.0, b1=40.0, Qinf2=30.0, b2=5.0, Qinf3=-10.0, b3=1.0,
)
CONCRETE = [ChabocheKinematicMulti3D, ChabocheKinematicMultiPS, ChabocheKinematicMulti1D]


def test_param_names():
    for cls in [ChabocheKinematicMulti, *CONCRETE]:
        assert cls.param_names == [
            "E", "nu", "sigma_y0", "C1", "gamma1", "C2", "gamma2", "C3", "gamma3",
            "Qinf1", "b1", "Qinf2", "b2", "Qinf3", "b3",
        ]


@pytest.mark.parametrize("cls", CONCRETE)
def test_every_state_is_explicit(cls):
    model = cls(**PARAMS)
    assert model.state_names == [
        "stress", "alpha1", "alpha2", "alpha3", "alpha", "R1", "R2", "R3", "R", "ep",
    ]
    assert model.implicit_state_names == []


def _updated(model, state_n, dlambda):
    inc = make_increment(model, state_n, stress_trial=state_n["stress"])
    return {u.name: u.value for u in model.update_state(state_n, dlambda, inc)}


@pytest.fixture
def state_n():
    return {
        "stress": np.array([280.0, -40.0, 10.0, 60.0, 0.0, -20.0]),
        "alpha1": np.array([6.0, -4.0, -2.0, 3.0, 0.0, -1.0]),
        "alpha2": np.array([-3.0, 2.0, 1.0, 0.0, 2.0, 0.0]),
        "alpha3": np.array([1.0, 1.0, -2.0, 0.0, 0.0, 4.0]),
        "alpha": np.array([4.0, -1.0, -3.0, 3.0, 2.0, 3.0]),
        "R1": 5.0, "R2": 2.0, "R3": -1.0, "R": 6.0, "ep": 0.02,
    }


def test_each_term_is_the_backward_euler_closed_form(state_n):
    model = ChabocheKinematicMulti3D(**PARAMS)
    dlambda = 3.0e-3
    new = _updated(model, state_n, dlambda)
    n_hat = model.flow(state_n).stress_like
    for i in (1, 2, 3):
        C, gamma = PARAMS[f"C{i}"], PARAMS[f"gamma{i}"]
        Qinf, b = PARAMS[f"Qinf{i}"], PARAMS[f"b{i}"]
        np.testing.assert_allclose(
            new[f"alpha{i}"],
            (state_n[f"alpha{i}"] + 2.0 / 3.0 * C * dlambda * n_hat) / (1.0 + gamma * dlambda),
            rtol=1e-14)
        assert float(new[f"R{i}"]) == pytest.approx(
            (state_n[f"R{i}"] + b * Qinf * dlambda) / (1.0 + b * dlambda), rel=1e-14)
    assert float(new["ep"]) == pytest.approx(0.02 + dlambda, rel=1e-14)


def test_the_totals_are_the_sums_of_the_parts(state_n):
    model = ChabocheKinematicMulti3D(**PARAMS)
    new = _updated(model, state_n, 3.0e-3)
    np.testing.assert_allclose(new["alpha"], new["alpha1"] + new["alpha2"] + new["alpha3"], rtol=1e-15)
    assert float(new["R"]) == pytest.approx(float(new["R1"] + new["R2"] + new["R3"]), rel=1e-15)


def test_the_yield_function_reads_the_parts(state_n):
    """The totals are output; the yield surface is placed by the parts."""
    model = ChabocheKinematicMulti3D(**PARAMS)
    shifted = dict(state_n, alpha=np.zeros(6), R=0.0)
    assert model.yield_function(shifted) == model.yield_function(state_n)
