"""Verify J2Isotropic parent and its dimension-specialized subclasses are numerically identical."""

import numpy as np
import pytest
from manforge.models import J2Isotropic, J2Isotropic3D, J2IsotropicPS, J2Isotropic1D
from manforge.core.dimension import SOLID_3D, PLANE_STRESS, UNIAXIAL_1D, PLANE_STRAIN

PARAMS = dict(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Parent-vs-subclass numerical equivalence
# ---------------------------------------------------------------------------

def test_j2_3d_parent_vs_subclass_yield_function(rng):
    parent = J2Isotropic(dimension=SOLID_3D, **PARAMS)
    child = J2Isotropic3D(**PARAMS)
    stress = rng.standard_normal(6)
    state = {"stress": stress, "ep": 0.01}
    assert parent.yield_function(state) == pytest.approx(child.yield_function(state))


def test_j2_3d_parent_vs_subclass_update_state(rng):
    parent = J2Isotropic(dimension=SOLID_3D, **PARAMS)
    child = J2Isotropic3D(**PARAMS)
    stress = rng.standard_normal(6)
    state = {"stress": stress, "ep": 0.01}
    dlambda = 0.005
    res_p = parent.update_state(dlambda, state, state)
    res_c = child.update_state(dlambda, state, state)
    np.testing.assert_allclose(res_p[0].value, res_c[0].value)


def test_j2_ps_parent_vs_subclass_yield_function(rng):
    parent = J2Isotropic(dimension=PLANE_STRESS, **PARAMS)
    child = J2IsotropicPS(**PARAMS)
    stress = rng.standard_normal(3)
    state = {"stress": stress, "ep": 0.01}
    assert parent.yield_function(state) == pytest.approx(child.yield_function(state))


def test_j2_ps_parent_vs_subclass_update_state(rng):
    parent = J2Isotropic(dimension=PLANE_STRESS, **PARAMS)
    child = J2IsotropicPS(**PARAMS)
    stress = rng.standard_normal(3)
    state = {"stress": stress, "ep": 0.01}
    dlambda = 0.005
    res_p = parent.update_state(dlambda, state, state)
    res_c = child.update_state(dlambda, state, state)
    np.testing.assert_allclose(res_p[0].value, res_c[0].value)


def test_j2_1d_parent_vs_subclass_yield_function(rng):
    parent = J2Isotropic(dimension=UNIAXIAL_1D, **PARAMS)
    child = J2Isotropic1D(**PARAMS)
    stress = rng.standard_normal(1)
    state = {"stress": stress, "ep": 0.01}
    assert parent.yield_function(state) == pytest.approx(child.yield_function(state))


def test_j2_1d_parent_vs_subclass_update_state(rng):
    parent = J2Isotropic(dimension=UNIAXIAL_1D, **PARAMS)
    child = J2Isotropic1D(**PARAMS)
    stress = rng.standard_normal(1)
    state = {"stress": stress, "ep": 0.01}
    dlambda = 0.005
    res_p = parent.update_state(dlambda, state, state)
    res_c = child.update_state(dlambda, state, state)
    np.testing.assert_allclose(res_p[0].value, res_c[0].value)


# ---------------------------------------------------------------------------
# State fields / MRO correctness
# ---------------------------------------------------------------------------

def test_j2_state_names():
    m = J2Isotropic3D(**PARAMS)
    assert m.state_names == ["stress", "ep"]
    assert m.implicit_state_names == []


def test_j2_parent_state_names():
    m = J2Isotropic(dimension=SOLID_3D, **PARAMS)
    assert m.state_names == ["stress", "ep"]


def test_j2_subclasses_inherit_state_fields():
    for cls, dim in [(J2Isotropic3D, SOLID_3D), (J2IsotropicPS, PLANE_STRESS),
                     (J2Isotropic1D, UNIAXIAL_1D)]:
        m = cls(**PARAMS)
        assert "ep" in m.state_names
        assert m.implicit_state_names == []


# ---------------------------------------------------------------------------
# Direct-parent construction with non-default dimension
# ---------------------------------------------------------------------------

def test_j2_parent_plane_strain_construction():
    m = J2Isotropic(dimension=PLANE_STRAIN, **PARAMS)
    assert m.ntens == 4


def test_j2_parent_yields_correctly_plane_strain(rng):
    m = J2Isotropic(dimension=PLANE_STRAIN, **PARAMS)
    stress = rng.standard_normal(4)
    state = {"stress": stress, "ep": 0.0}
    f = m.yield_function(state)
    assert np.isfinite(float(f))


# ---------------------------------------------------------------------------
# param_names on parent and all subclasses
# ---------------------------------------------------------------------------

def test_j2_param_names():
    for cls in [J2Isotropic, J2Isotropic3D, J2IsotropicPS, J2Isotropic1D]:
        assert cls.param_names == ["E", "nu", "sigma_y0", "H"]
