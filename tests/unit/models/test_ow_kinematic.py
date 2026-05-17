"""Verify OWKinematic parent and its dimension-specialized subclasses are numerically identical."""

import numpy as np
import pytest
from manforge.models import OWKinematic, OWKinematic3D, OWKinematicPS, OWKinematic1D
from manforge.core.dimension import SOLID_3D, PLANE_STRESS, UNIAXIAL_1D, PLANE_STRAIN

PARAMS = dict(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=1.0)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _elastic_C(model):
    return model.isotropic_C(
        model.E * model.nu / ((1 + model.nu) * (1 - 2 * model.nu)),
        model.E / (2 * (1 + model.nu)),
    )


# ---------------------------------------------------------------------------
# Parent-vs-subclass numerical equivalence
# ---------------------------------------------------------------------------

def test_ow_3d_parent_vs_subclass_yield_function(rng):
    parent = OWKinematic(dimension=SOLID_3D, **PARAMS)
    child = OWKinematic3D(**PARAMS)
    stress = rng.standard_normal(6) * 300
    alpha = rng.standard_normal(6) * 10
    state = {"stress": stress, "alpha": alpha, "ep": 0.01}
    assert parent.yield_function(state) == pytest.approx(child.yield_function(state))


def test_ow_3d_parent_vs_subclass_state_residual(rng):
    parent = OWKinematic(dimension=SOLID_3D, **PARAMS)
    child = OWKinematic3D(**PARAMS)
    stress = rng.standard_normal(6) * 300
    alpha = rng.standard_normal(6) * 10
    state_new = {"stress": stress, "alpha": alpha, "ep": 0.01}
    state_n = {"stress": stress, "alpha": alpha, "ep": 0.01}
    stress_trial = stress + 10.0
    dlambda = 0.005
    res_p = parent.state_residual(state_new, dlambda, state_n, stress_trial=stress_trial)
    res_c = child.state_residual(state_new, dlambda, state_n, stress_trial=stress_trial)
    for rp, rc in zip(res_p, res_c):
        np.testing.assert_allclose(rp.value, rc.value)


def test_ow_ps_parent_vs_subclass_yield_function(rng):
    parent = OWKinematic(dimension=PLANE_STRESS, **PARAMS)
    child = OWKinematicPS(**PARAMS)
    stress = rng.standard_normal(3) * 300
    alpha = rng.standard_normal(3) * 10
    state = {"stress": stress, "alpha": alpha, "ep": 0.01}
    assert parent.yield_function(state) == pytest.approx(child.yield_function(state))


def test_ow_1d_parent_vs_subclass_yield_function(rng):
    parent = OWKinematic(dimension=UNIAXIAL_1D, **PARAMS)
    child = OWKinematic1D(**PARAMS)
    stress = rng.standard_normal(1) * 300
    alpha = rng.standard_normal(1) * 10
    state = {"stress": stress, "alpha": alpha, "ep": 0.01}
    assert parent.yield_function(state) == pytest.approx(child.yield_function(state))


# ---------------------------------------------------------------------------
# State fields / MRO correctness — all implicit (vector NR)
# ---------------------------------------------------------------------------

def test_ow_implicit_state_names():
    m = OWKinematic3D(**PARAMS)
    assert set(m.implicit_state_names) == {"stress", "alpha", "ep"}


def test_ow_parent_implicit_state_names():
    m = OWKinematic(dimension=SOLID_3D, **PARAMS)
    assert set(m.implicit_state_names) == {"stress", "alpha", "ep"}


def test_ow_subclasses_vector_nr_path():
    for cls in [OWKinematic3D, OWKinematicPS, OWKinematic1D]:
        m = cls(**PARAMS)
        assert "stress" in m.implicit_state_names, f"{cls.__name__} should use vector NR"


# ---------------------------------------------------------------------------
# Direct-parent construction with non-default dimension
# ---------------------------------------------------------------------------

def test_ow_parent_plane_strain_construction():
    m = OWKinematic(dimension=PLANE_STRAIN, **PARAMS)
    assert m.ntens == 4


def test_ow_parent_plane_strain_yield_function(rng):
    m = OWKinematic(dimension=PLANE_STRAIN, **PARAMS)
    stress = rng.standard_normal(4) * 300
    alpha = rng.standard_normal(4) * 10
    state = {"stress": stress, "alpha": alpha, "ep": 0.0}
    f = m.yield_function(state)
    assert np.isfinite(float(f))


# ---------------------------------------------------------------------------
# param_names on parent and all subclasses
# ---------------------------------------------------------------------------

def test_ow_param_names():
    for cls in [OWKinematic, OWKinematic3D, OWKinematicPS, OWKinematic1D]:
        assert cls.param_names == ["E", "nu", "sigma_y0", "C_k", "gamma"]
