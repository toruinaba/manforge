"""Tests common to all kinematic hardening models (AF and OW).

Covers:
- Initial state shapes and zero values
- Elastic domain: stress = C @ deps, tangent = C, state unchanged
- Yield surface consistency after plastic steps
- FD tangent vs AD tangent agreement
- Backstress deviatoricity and ep monotonicity
- gamma=0 limit: linear kinematic hardening (Prager's rule)
- Bauschinger effect
- StrainDriver cyclic integration
- PLANE_STRAIN, PLANE_STRESS, 1D stress states
"""

import pytest
import numpy as np
import autograd.numpy as anp

from manforge.models.af_kinematic import AFKinematic3D, AFKinematicPS, AFKinematic1D
from manforge.models.ow_kinematic import OWKinematic3D, OWKinematicPS, OWKinematic1D
from manforge.core.stress_update import stress_update
from manforge.core.stress_state import PLANE_STRAIN
from manforge.simulation.driver import StrainDriver
from manforge.simulation.types import FieldHistory, FieldType
from manforge.verification.fd_check import check_tangent


_FACTORIES_3D = {
    "af": lambda: AFKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0),
    "ow": lambda: OWKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=1.0),
}
_FACTORIES_PE = {
    "af": lambda: AFKinematic3D(stress_state=PLANE_STRAIN, E=210000.0, nu=0.3, sigma_y0=250.0,
                                C_k=10000.0, gamma=100.0),
    "ow": lambda: OWKinematic3D(stress_state=PLANE_STRAIN, E=210000.0, nu=0.3, sigma_y0=250.0,
                                C_k=10000.0, gamma=1.0),
}
_FACTORIES_PS = {
    "af": lambda: AFKinematicPS(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0),
    "ow": lambda: OWKinematicPS(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=1.0),
}
_FACTORIES_1D = {
    "af": lambda: AFKinematic1D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0),
    "ow": lambda: OWKinematic1D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=1.0),
}
_FACTORIES_GAMMA0 = {
    "af": lambda: AFKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=1000.0, gamma=0.0),
    "ow": lambda: OWKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=1000.0, gamma=0.0),
}

_DEPS_VEC_LIST = [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
    [2e-3, 0.0, 0.0, 2e-3, 0.0, 0.0],
]


@pytest.fixture(params=["af", "ow"])
def km_model(request):
    return _FACTORIES_3D[request.param]()


@pytest.fixture
def km_state0(km_model):
    return km_model.initial_state()


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_initial_state_shapes(km_model):
    state = km_model.initial_state()
    assert state["alpha"].shape == (6,)
    assert state["ep"].shape == ()


def test_initial_state_zeros(km_model):
    state = km_model.initial_state()
    assert anp.allclose(state["alpha"], 0.0)
    assert float(state["ep"]) == 0.0


# ---------------------------------------------------------------------------
# Elastic domain
# ---------------------------------------------------------------------------

def test_elastic_stress_is_trial(km_model, km_state0):
    """Elastic step: stress = C @ deps, tangent = C, state unchanged."""
    C = km_model.elastic_stiffness()
    deps = anp.array([0.5e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    _r = stress_update(km_model, deps, anp.zeros(6), km_state0)

    np.testing.assert_allclose(np.array(_r.stress), np.array(C @ deps), rtol=1e-10)
    np.testing.assert_allclose(np.array(_r.ddsdde), np.array(C), rtol=1e-10)
    np.testing.assert_allclose(np.array(_r.state["alpha"]), np.zeros(6), atol=1e-30)
    assert float(_r.state["ep"]) == 0.0


# ---------------------------------------------------------------------------
# Yield surface consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deps_vec", _DEPS_VEC_LIST)
def test_yield_consistency_plastic(km_model, km_state0, deps_vec):
    """After a plastic step, stress must lie on the yield surface."""
    deps = anp.array(deps_vec)
    _r = stress_update(km_model, deps, anp.zeros(6), km_state0)
    f = km_model.yield_function(_r.stress, _r.state)
    assert abs(float(f)) < 1e-8, f"Yield not satisfied: f = {float(f):.3e}"


# ---------------------------------------------------------------------------
# FD tangent verification
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("deps_vec", _DEPS_VEC_LIST)
def test_tangent_fd_plastic_virgin(km_model, km_state0, deps_vec):
    """AD consistent tangent must match FD for kinematic hardening models."""
    result = check_tangent(km_model, anp.zeros(6), km_state0, anp.array(deps_vec))
    assert result.passed, (
        f"FD tangent check failed: max_rel_err = {result.max_rel_err:.3e}\n"
        f"AD tangent:\n{np.array(result.ddsdde_ad)}\n"
        f"FD tangent:\n{np.array(result.ddsdde_fd)}"
    )


def test_tangent_fd_elastic(km_model, km_state0):
    """Elastic step: FD and AD tangents agree (both equal C)."""
    deps = anp.array([0.5e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = check_tangent(km_model, anp.zeros(6), km_state0, deps)
    assert result.passed, f"Elastic FD tangent failed: {result.max_rel_err:.3e}"


@pytest.mark.slow
def test_tangent_fd_prestressed(km_model, km_state0):
    """FD tangent from a non-virgin (plastically pre-strained) state."""
    deps1 = (lambda _a: (_a.__setitem__(0, 3e-3), _a)[1])(np.zeros(6))
    _r1 = stress_update(km_model, deps1, anp.zeros(6), km_state0)
    deps2 = (lambda _a: (_a.__setitem__(0, 1e-3), _a)[1])(np.zeros(6))
    result = check_tangent(km_model, _r1.stress, _r1.state, deps2)
    assert result.passed, f"Pre-stressed FD tangent failed: {result.max_rel_err:.3e}"


# ---------------------------------------------------------------------------
# Backstress properties
# ---------------------------------------------------------------------------

def test_backstress_is_nonzero_after_plastic_step(km_model, km_state0):
    deps = (lambda _a: (_a.__setitem__(0, 3e-3), _a)[1])(np.zeros(6))
    state_new = stress_update(km_model, deps, anp.zeros(6), km_state0).state
    assert anp.linalg.norm(state_new["alpha"]) > 1e-3


def test_backstress_is_deviatoric(km_model, km_state0):
    """Backstress must remain deviatoric (trace of direct components ≈ 0)."""
    deps = (lambda _a: (_a.__setitem__(0, 3e-3), _a)[1])(np.zeros(6))
    state_new = stress_update(km_model, deps, anp.zeros(6), km_state0).state
    trace = float(state_new["alpha"][0] + state_new["alpha"][1] + state_new["alpha"][2])
    assert abs(trace) < 1e-8, f"Backstress not deviatoric: trace = {trace:.3e}"


def test_ep_increases_with_plastic_loading(km_model, km_state0):
    """Equivalent plastic strain must increase monotonically under plastic loading."""
    eps_values = []
    stress_n = anp.zeros(6)
    state_n = km_state0
    for _ in range(5):
        deps = (lambda _a: (_a.__setitem__(0, 1e-3), _a)[1])(np.zeros(6))
        _r = stress_update(km_model, deps, stress_n, state_n)
        stress_n, state_n = _r.stress, _r.state
        eps_values.append(float(state_n["ep"]))
    assert all(b > a for a, b in zip(eps_values, eps_values[1:]))


# ---------------------------------------------------------------------------
# gamma=0 limit: linear kinematic hardening (Prager's rule)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_type", ["af", "ow"])
def test_gamma0_gives_linear_kinematic(model_type):
    """With gamma=0, both AF and OW reduce to Prager linear kinematic hardening."""
    model = _FACTORIES_GAMMA0[model_type]()
    state0 = model.initial_state()
    deps = (lambda _a: (_a.__setitem__(0, 3e-3), _a)[1])(np.zeros(6))
    _r = stress_update(model, deps, anp.zeros(6), state0)
    f = model.yield_function(_r.stress, _r.state)
    assert abs(float(f)) < 1e-8
    assert float(_r.state["alpha"][0]) > 0.0


@pytest.mark.slow
@pytest.mark.parametrize("model_type", ["af", "ow"])
def test_gamma0_fd_tangent(model_type):
    """FD tangent must pass for the linear kinematic limit (gamma=0)."""
    model = _FACTORIES_GAMMA0[model_type]()
    deps = (lambda _a: (_a.__setitem__(0, 3e-3), _a)[1])(np.zeros(6))
    result = check_tangent(model, anp.zeros(6), model.initial_state(), deps)
    assert result.passed, f"{model_type} gamma=0 FD tangent failed: {result.max_rel_err:.3e}"


# ---------------------------------------------------------------------------
# Bauschinger effect
# ---------------------------------------------------------------------------

def test_bauschinger_effect(km_model, km_state0):
    """After forward plastic loading, reverse yielding occurs at reduced stress."""
    deps_fwd = (lambda _a: (_a.__setitem__(0, 5e-3), _a)[1])(np.zeros(6))
    _r1 = stress_update(km_model, deps_fwd, anp.zeros(6), km_state0)
    stress1, state1 = _r1.stress, _r1.state
    assert anp.linalg.norm(state1["alpha"]) > 1e-3

    # Elastic unloading to near-zero stress
    C = km_model.elastic_stiffness()
    deps_unload = anp.linalg.solve(C, -stress1)
    _r2 = stress_update(km_model, deps_unload, stress1, state1)
    stress2, state2 = _r2.stress, _r2.state

    # Compressive reload: reverse yielding must occur
    deps_rev = (lambda _a: (_a.__setitem__(0, -3e-3), _a)[1])(np.zeros(6))
    _r3 = stress_update(km_model, deps_rev, stress2, state2)
    assert float(_r3.state["ep"]) > float(state2["ep"]), "No reverse yielding detected"


# ---------------------------------------------------------------------------
# Cyclic driver test
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_cyclic_loading_driver(km_model):
    """StrainDriver must run cyclic tension-compression without error."""
    history = np.zeros((20, 6))
    history[:10, 0] = np.linspace(0, 4e-3, 10)
    history[10:, 0] = np.linspace(4e-3, -4e-3, 10)

    load = FieldHistory(type=FieldType.STRAIN, name="Strain", data=history)
    result = StrainDriver().run(
        km_model, load,
        collect_state={"alpha": FieldType.STRESS, "ep": FieldType.STRAIN},
    )

    assert result.stress.shape == (20, 6)
    assert result.fields["ep"].data.shape == (20,)
    assert result.fields["alpha"].data.shape == (20, 6)
    ep = result.fields["ep"].data
    assert np.all(np.diff(ep) >= -1e-12), "ep decreased during loading"


# ---------------------------------------------------------------------------
# PLANE_STRAIN
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_type", ["af", "ow"])
def test_plane_strain_yield_consistency(model_type):
    """Yield surface consistency after plastic step in PLANE_STRAIN."""
    model = _FACTORIES_PE[model_type]()
    state0 = model.initial_state()
    assert state0["alpha"].shape == (4,)
    deps = (lambda _a: (_a.__setitem__(0, 3e-3), _a)[1])(np.zeros(4))
    _r = stress_update(model, deps, anp.zeros(4), state0)
    f = model.yield_function(_r.stress, _r.state)
    assert abs(float(f)) < 1e-8


@pytest.mark.slow
@pytest.mark.parametrize("model_type", ["af", "ow"])
def test_plane_strain_fd_tangent(model_type):
    """FD tangent must pass for PLANE_STRAIN."""
    model = _FACTORIES_PE[model_type]()
    deps = (lambda _a: (_a.__setitem__(0, 3e-3), _a)[1])(np.zeros(4))
    result = check_tangent(model, anp.zeros(4), model.initial_state(), deps)
    assert result.passed, f"{model_type} PLANE_STRAIN FD tangent failed: {result.max_rel_err:.3e}"


# ---------------------------------------------------------------------------
# PLANE_STRESS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_type", ["af", "ow"])
def test_plane_stress_initial_state(model_type):
    model = _FACTORIES_PS[model_type]()
    state0 = model.initial_state()
    assert state0["alpha"].shape == (3,)
    assert state0["ep"].shape == ()


_DEPS_PS = {
    # AF PS: yield strain ≈ sigma_y0 / (E/(1-nu²)) ≈ 1.08e-3; use 1.3e-3
    "af": anp.array([1.3e-3, 0.0, 0.0]),
    "ow": anp.array([2e-3, 0.0, 0.0]),
}


@pytest.mark.parametrize("model_type", ["af", "ow"])
def test_plane_stress_yield_consistency(model_type):
    """Yield surface consistency after plastic step in PLANE_STRESS."""
    model = _FACTORIES_PS[model_type]()
    state0 = model.initial_state()
    deps = _DEPS_PS[model_type]
    _r = stress_update(model, deps, anp.zeros(3), state0)
    f = model.yield_function(_r.stress, _r.state)
    assert abs(float(f)) < 1e-8


@pytest.mark.slow
@pytest.mark.parametrize("model_type", ["af", "ow"])
def test_plane_stress_fd_tangent(model_type):
    """FD tangent must pass for PLANE_STRESS."""
    model = _FACTORIES_PS[model_type]()
    deps = _DEPS_PS[model_type]
    result = check_tangent(model, anp.zeros(3), model.initial_state(), deps)
    assert result.passed, f"{model_type} PLANE_STRESS FD tangent failed: {result.max_rel_err:.3e}"


# ---------------------------------------------------------------------------
# 1D model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_type", ["af", "ow"])
def test_1d_initial_state(model_type):
    model = _FACTORIES_1D[model_type]()
    state0 = model.initial_state()
    assert state0["alpha"].shape == (1,)
    assert state0["ep"].shape == ()


@pytest.mark.parametrize("model_type", ["af", "ow"])
def test_1d_yield_consistency(model_type):
    """1D model: yield surface consistency after plastic step."""
    model = _FACTORIES_1D[model_type]()
    state0 = model.initial_state()
    deps = anp.array([3e-3])
    _r = stress_update(model, deps, anp.zeros(1), state0)
    f = model.yield_function(_r.stress, _r.state)
    assert abs(float(f)) < 1e-8


@pytest.mark.parametrize("model_type", ["af", "ow"])
def test_1d_fd_tangent(model_type):
    """FD tangent must pass for 1D model."""
    model = _FACTORIES_1D[model_type]()
    deps = anp.array([5e-3])
    result = check_tangent(model, anp.zeros(1), model.initial_state(), deps)
    assert result.passed, f"{model_type} 1D FD tangent failed: {result.max_rel_err:.3e}"


@pytest.mark.parametrize("model_type", ["af", "ow"])
def test_1d_bauschinger(model_type):
    """1D model: Bauschinger effect — reverse yielding after forward plastic loading."""
    model = _FACTORIES_1D[model_type]()
    state0 = model.initial_state()

    deps_fwd = anp.array([5e-3])
    _r1 = stress_update(model, deps_fwd, anp.zeros(1), state0)
    stress1, state1 = _r1.stress, _r1.state
    assert float(state1["alpha"][0]) > 0.0

    # Elastic unloading to near zero stress
    C = model.elastic_stiffness()
    deps_unload = anp.linalg.solve(C, -stress1)
    _r2 = stress_update(model, deps_unload, stress1, state1)
    stress2, state2 = _r2.stress, _r2.state

    # Compressive step: should yield plastically (Bauschinger)
    deps_rev = anp.array([-3e-3])
    state3 = stress_update(model, deps_rev, stress2, state2).state
    assert float(state3["ep"]) > float(state2["ep"]), "No 1D reverse yielding (Bauschinger)"
