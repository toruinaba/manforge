"""Verify AFKinematic parent and its dimension-specialized subclasses are numerically identical."""

import numpy as np
import pytest
from manforge.models import AFKinematic, AFKinematic3D, AFKinematicPS, AFKinematic1D
from manforge.core.dimension import SOLID_3D, PLANE_STRESS, UNIAXIAL_1D, PLANE_STRAIN

PARAMS = dict(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Parent-vs-subclass numerical equivalence
# ---------------------------------------------------------------------------

def test_af_3d_parent_vs_subclass_yield_function(rng):
    parent = AFKinematic(dimension=SOLID_3D, **PARAMS)
    child = AFKinematic3D(**PARAMS)
    stress = rng.standard_normal(6)
    alpha = rng.standard_normal(6) * 10
    state = {"stress": stress, "alpha": alpha, "ep": 0.01}
    assert parent.yield_function(state) == pytest.approx(child.yield_function(state))


def test_af_3d_parent_vs_subclass_update_state(rng):
    parent = AFKinematic(dimension=SOLID_3D, **PARAMS)
    child = AFKinematic3D(**PARAMS)
    stress = rng.standard_normal(6) * 300
    alpha = rng.standard_normal(6) * 10
    state_new = {"stress": stress, "alpha": alpha, "ep": 0.01}
    state_n = {"stress": stress, "alpha": alpha, "ep": 0.01}
    dlambda = 0.005
    res_p = parent.update_state(dlambda, state_new, state_n)
    res_c = child.update_state(dlambda, state_new, state_n)
    # alpha and ep updates
    for rp, rc in zip(res_p, res_c):
        np.testing.assert_allclose(rp.value, rc.value)


def test_af_ps_parent_vs_subclass_yield_function(rng):
    parent = AFKinematic(dimension=PLANE_STRESS, **PARAMS)
    child = AFKinematicPS(**PARAMS)
    stress = rng.standard_normal(3)
    alpha = rng.standard_normal(3) * 10
    state = {"stress": stress, "alpha": alpha, "ep": 0.01}
    assert parent.yield_function(state) == pytest.approx(child.yield_function(state))


def test_af_1d_parent_vs_subclass_yield_function(rng):
    parent = AFKinematic(dimension=UNIAXIAL_1D, **PARAMS)
    child = AFKinematic1D(**PARAMS)
    stress = rng.standard_normal(1) * 300
    alpha = rng.standard_normal(1) * 10
    state = {"stress": stress, "alpha": alpha, "ep": 0.01}
    assert parent.yield_function(state) == pytest.approx(child.yield_function(state))


# ---------------------------------------------------------------------------
# State fields / MRO correctness
# ---------------------------------------------------------------------------

def test_af_state_names():
    m = AFKinematic3D(**PARAMS)
    assert m.state_names == ["stress", "alpha", "ep"]
    assert m.implicit_state_names == []


def test_af_parent_state_names():
    m = AFKinematic(dimension=SOLID_3D, **PARAMS)
    assert m.state_names == ["stress", "alpha", "ep"]
    assert m.implicit_state_names == []


def test_af_subclasses_explicit_path():
    for cls in [AFKinematic3D, AFKinematicPS, AFKinematic1D]:
        m = cls(**PARAMS)
        assert m.implicit_state_names == [], f"{cls.__name__} should use explicit path"


# ---------------------------------------------------------------------------
# Direct-parent construction with non-default dimension
# ---------------------------------------------------------------------------

def test_af_parent_plane_strain_construction():
    m = AFKinematic(dimension=PLANE_STRAIN, **PARAMS)
    assert m.ntens == 4


def test_af_parent_plane_strain_yield_function(rng):
    m = AFKinematic(dimension=PLANE_STRAIN, **PARAMS)
    stress = rng.standard_normal(4)
    alpha = rng.standard_normal(4) * 10
    state = {"stress": stress, "alpha": alpha, "ep": 0.0}
    f = m.yield_function(state)
    assert np.isfinite(float(f))


# ---------------------------------------------------------------------------
# param_names on parent and all subclasses
# ---------------------------------------------------------------------------

def test_af_param_names():
    for cls in [AFKinematic, AFKinematic3D, AFKinematicPS, AFKinematic1D]:
        assert cls.param_names == ["E", "nu", "sigma_y0", "C_k", "gamma"]
