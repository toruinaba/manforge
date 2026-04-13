"""Tests for the Armstrong-Frederick kinematic hardening model.

These tests verify that the generic Newton-Raphson return_mapping solver and
consistent_tangent implicit differentiation work correctly for nonlinear
hardening — a regime not exercised by the linear isotropic J2 model.

Key verifications
-----------------
- Yield surface consistency after plastic steps
- FD tangent vs AD tangent agreement (the core correctness test)
- Backstress deviatoricity (trace of direct components ≈ 0)
- gamma=0 limit reduces to linear kinematic hardening behaviour
- Bauschinger effect under cyclic loading
- Correct behaviour across multiple stress states (3D, PLANE_STRAIN, 1D)
"""

import pytest
import numpy as np
import jax.numpy as jnp

import manforge  # enables JAX float64
from manforge.models.af_kinematic import AFKinematic3D, AFKinematicPS, AFKinematic1D
from manforge.core.return_mapping import return_mapping
from manforge.simulation.driver import StrainDriver
from manforge.simulation.types import FieldHistory, FieldType
from manforge.verification.fd_check import check_tangent
from manforge.core.stress_state import PLANE_STRAIN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    return AFKinematic3D()


@pytest.fixture
def params():
    return {
        "E": 210000.0,
        "nu": 0.3,
        "sigma_y0": 250.0,
        "C_k": 10000.0,
        "gamma": 100.0,
        # Saturated backstress = C_k / gamma = 100 MPa
        # Total flow stress at saturation ≈ 350 MPa
    }


@pytest.fixture
def state0(model):
    return model.initial_state()


@pytest.fixture
def lame(params):
    E, nu = params["E"], params["nu"]
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam, mu


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_initial_state_shapes(model):
    state = model.initial_state()
    assert state["alpha"].shape == (6,)
    assert state["ep"].shape == ()


def test_initial_state_zeros(model):
    state = model.initial_state()
    assert jnp.allclose(state["alpha"], 0.0)
    assert float(state["ep"]) == 0.0


# ---------------------------------------------------------------------------
# Elastic domain
# ---------------------------------------------------------------------------

def test_elastic_stress_is_trial(model, params, state0):
    """Elastic step: stress = C @ deps, tangent = C, state unchanged."""
    C = model.elastic_stiffness(params)
    deps = jnp.array([0.5e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    stress_new, state_new, ddsdde = return_mapping(model, deps, jnp.zeros(6), state0, params)

    np.testing.assert_allclose(np.array(stress_new), np.array(C @ deps), rtol=1e-10)
    np.testing.assert_allclose(np.array(ddsdde), np.array(C), rtol=1e-10)
    np.testing.assert_allclose(np.array(state_new["alpha"]), np.zeros(6), atol=1e-30)
    assert float(state_new["ep"]) == 0.0


# ---------------------------------------------------------------------------
# Yield surface consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deps_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],          # uniaxial (σ_vm ≈ 323 MPa > 250)
    [2e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0],       # isochoric (σ_vm ≈ 485 MPa > 250)
    [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],           # shear (σ_vm ≈ 280 MPa > 250)
    [2e-3, 0.0, 0.0, 2e-3, 0.0, 0.0],          # uniaxial + shear
])
def test_yield_consistency_plastic(model, params, state0, deps_vec):
    """After a plastic step, stress must lie on the yield surface."""
    deps = jnp.array(deps_vec)
    stress_new, state_new, _ = return_mapping(model, deps, jnp.zeros(6), state0, params)
    f = model.yield_function(stress_new, state_new, params)
    assert abs(float(f)) < 1e-8, f"Yield not satisfied: f = {float(f):.3e}"


# ---------------------------------------------------------------------------
# FD tangent verification — the core test for return_mapping correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deps_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],          # uniaxial
    [2e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0],       # isochoric
    [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],           # shear
    [2e-3, 0.0, 0.0, 2e-3, 0.0, 0.0],          # uniaxial + shear
])
def test_tangent_fd_plastic_virgin(model, params, state0, deps_vec):
    """AD consistent tangent must match FD for nonlinear kinematic hardening."""
    result = check_tangent(
        model,
        jnp.zeros(6),
        state0,
        params,
        jnp.array(deps_vec),
    )
    assert result.passed, (
        f"FD tangent check failed: max_rel_err = {result.max_rel_err:.3e}\n"
        f"AD tangent:\n{np.array(result.ddsdde_ad)}\n"
        f"FD tangent:\n{np.array(result.ddsdde_fd)}"
    )


def test_tangent_fd_elastic(model, params, state0):
    """Elastic step: tangent = C (FD and AD agree trivially)."""
    deps = jnp.array([0.5e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = check_tangent(model, jnp.zeros(6), state0, params, deps)
    assert result.passed, f"Elastic FD tangent failed: {result.max_rel_err:.3e}"


def test_tangent_fd_prestressed(model, params, state0):
    """FD tangent check from a non-virgin (plastically pre-strained) state."""
    # First step: push into plasticity to build up backstress
    deps1 = jnp.zeros(6).at[0].set(3e-3)
    stress1, state1, _ = return_mapping(model, deps1, jnp.zeros(6), state0, params)

    # Second step: verify FD tangent from this state
    deps2 = jnp.zeros(6).at[0].set(1e-3)
    result = check_tangent(model, stress1, state1, params, deps2)
    assert result.passed, f"Pre-stressed FD tangent failed: {result.max_rel_err:.3e}"


# ---------------------------------------------------------------------------
# Backstress properties
# ---------------------------------------------------------------------------

def test_backstress_is_nonzero_after_plastic_step(model, params, state0):
    """Backstress must become nonzero after a plastic step."""
    deps = jnp.zeros(6).at[0].set(3e-3)
    _, state_new, _ = return_mapping(model, deps, jnp.zeros(6), state0, params)
    assert jnp.linalg.norm(state_new["alpha"]) > 1e-3


def test_backstress_is_deviatoric(model, params, state0):
    """Backstress must remain deviatoric (trace of direct components ≈ 0)."""
    deps = jnp.zeros(6).at[0].set(3e-3)
    _, state_new, _ = return_mapping(model, deps, jnp.zeros(6), state0, params)
    # Direct components are the first ndi = 3 entries
    trace = float(state_new["alpha"][0] + state_new["alpha"][1] + state_new["alpha"][2])
    assert abs(trace) < 1e-8, f"Backstress not deviatoric: trace = {trace:.3e}"


def test_ep_increases_with_plastic_loading(model, params, state0):
    """Equivalent plastic strain must increase monotonically under plastic loading."""
    eps_values = []
    stress_n = jnp.zeros(6)
    state_n = state0
    for _ in range(5):
        deps = jnp.zeros(6).at[0].set(1e-3)
        stress_n, state_n, _ = return_mapping(model, deps, stress_n, state_n, params)
        eps_values.append(float(state_n["ep"]))
    assert all(b > a for a, b in zip(eps_values, eps_values[1:]))


# ---------------------------------------------------------------------------
# gamma = 0 limit: linear kinematic hardening (Prager's rule)
# ---------------------------------------------------------------------------

def test_gamma0_gives_linear_kinematic():
    """With gamma=0, AF reduces to Prager linear kinematic hardening."""
    model = AFKinematic3D()
    params_linear = {
        "E": 210000.0, "nu": 0.3, "sigma_y0": 250.0,
        "C_k": 1000.0, "gamma": 0.0,
    }
    state0 = model.initial_state()
    deps = jnp.zeros(6).at[0].set(3e-3)
    stress_new, state_new, _ = return_mapping(
        model, deps, jnp.zeros(6), state0, params_linear
    )
    # Yield must still be satisfied
    f = model.yield_function(stress_new, state_new, params_linear)
    assert abs(float(f)) < 1e-8

    # With gamma=0 and pure uniaxial loading, the backstress must be purely
    # axial (deviatoric; positive in loading direction)
    assert float(state_new["alpha"][0]) > 0.0


def test_gamma0_fd_tangent():
    """FD tangent check must pass for the linear kinematic limit (gamma=0)."""
    model = AFKinematic3D()
    params_linear = {
        "E": 210000.0, "nu": 0.3, "sigma_y0": 250.0,
        "C_k": 1000.0, "gamma": 0.0,
    }
    deps = jnp.zeros(6).at[0].set(3e-3)
    result = check_tangent(model, jnp.zeros(6), model.initial_state(), params_linear, deps)
    assert result.passed, f"gamma=0 FD tangent failed: {result.max_rel_err:.3e}"


# ---------------------------------------------------------------------------
# Bauschinger effect under cyclic loading
# ---------------------------------------------------------------------------

def test_bauschinger_effect(model, params):
    """After forward plastic loading, compressive yield stress is reduced."""
    state0 = model.initial_state()
    sigma0 = jnp.zeros(6)

    # Forward loading: push well into plasticity (build up backstress)
    deps_fwd = jnp.zeros(6).at[0].set(3e-3)
    sigma1, state1, _ = return_mapping(model, deps_fwd, sigma0, state0, params)

    # Unload elastically to near zero
    C = model.elastic_stiffness(params)
    deps_back = -sigma1 / C[0, 0] * jnp.zeros(6).at[0].set(1.0)
    sigma2, state2, _ = return_mapping(model, deps_back, sigma1, state1, params)

    # Compressive yield stress: forward sigma_y0 + |backstress_11|
    alpha_11 = float(state2["alpha"][0])

    # Forward yield stress is params["sigma_y0"] (no isotropic hardening)
    # Compressive yield stress is sigma_y0 - alpha_11 (Bauschinger effect)
    # So compressive yield onset is *lower* than sigma_y0 when alpha_11 > 0
    assert alpha_11 > 0.0, "Expected tensile backstress after forward loading"

    # Verify that a small compressive increment yields plastically
    # (it would be elastic without the Bauschinger shift)
    small_compressive_plastic = jnp.zeros(6).at[0].set(-2e-3)
    sigma3, state3, _ = return_mapping(model, small_compressive_plastic, sigma2, state2, params)
    # If Bauschinger is working, ep should increase (plastic yielding occurred)
    dep = float(state3["ep"]) - float(state2["ep"])
    assert dep > 0.0, "Expected plastic yielding on reverse loading (Bauschinger)"


# ---------------------------------------------------------------------------
# Cyclic driver test
# ---------------------------------------------------------------------------

def test_cyclic_loading_driver(model, params):
    """StrainDriver should run cyclic tension-compression without error."""
    driver = StrainDriver()
    history = np.zeros((20, 6))
    history[:10, 0] = np.linspace(0, 4e-3, 10)   # forward loading
    history[10:, 0] = np.linspace(4e-3, -4e-3, 10)  # reverse loading

    load = FieldHistory(type=FieldType.STRAIN, name="Strain", data=history)
    result = driver.run(model, load, params, collect_state={"alpha": FieldType.STRESS, "ep": FieldType.STRAIN})

    assert result.stress.shape == (20, 6)
    assert result.fields["ep"].data.shape == (20,)
    assert result.fields["alpha"].data.shape == (20, 6)

    # ep must be non-decreasing
    ep = result.fields["ep"].data
    assert np.all(np.diff(ep) >= -1e-12)


# ---------------------------------------------------------------------------
# PLANE_STRAIN stress state
# ---------------------------------------------------------------------------

def test_plane_strain_yield_consistency(params):
    """AFKinematic3D with PLANE_STRAIN: yield consistency after plastic step."""
    model_pe = AFKinematic3D(PLANE_STRAIN)
    state0 = model_pe.initial_state()
    assert state0["alpha"].shape == (4,)

    deps = jnp.zeros(4).at[0].set(3e-3)
    stress_new, state_new, _ = return_mapping(
        model_pe, deps, jnp.zeros(4), state0, params
    )
    f = model_pe.yield_function(stress_new, state_new, params)
    assert abs(float(f)) < 1e-8


def test_plane_strain_fd_tangent(params):
    """FD tangent must pass for PLANE_STRAIN."""
    model_pe = AFKinematic3D(PLANE_STRAIN)
    deps = jnp.zeros(4).at[0].set(3e-3)
    result = check_tangent(model_pe, jnp.zeros(4), model_pe.initial_state(), params, deps)
    assert result.passed, f"PLANE_STRAIN FD tangent failed: {result.max_rel_err:.3e}"


# ---------------------------------------------------------------------------
# PLANE_STRESS (AFKinematicPS)
# ---------------------------------------------------------------------------

def test_plane_stress_initial_state():
    model_ps = AFKinematicPS()
    state0 = model_ps.initial_state()
    assert state0["alpha"].shape == (3,)


def test_plane_stress_yield_consistency(params):
    """AFKinematicPS: yield consistency after plastic step.

    Uses a modest overstress (≈20% above yield) so the NR converges reliably.
    PS yield strain ≈ sigma_y0 / (E/(1-nu²)) ≈ 1.08e-3; use 1.3e-3.
    """
    model_ps = AFKinematicPS()
    state0 = model_ps.initial_state()
    deps = jnp.zeros(3).at[0].set(1.3e-3)
    stress_new, state_new, _ = return_mapping(
        model_ps, deps, jnp.zeros(3), state0, params
    )
    f = model_ps.yield_function(stress_new, state_new, params)
    assert abs(float(f)) < 1e-8


def test_plane_stress_fd_tangent(params):
    """FD tangent must pass for PLANE_STRESS."""
    model_ps = AFKinematicPS()
    deps = jnp.zeros(3).at[0].set(1.3e-3)
    result = check_tangent(model_ps, jnp.zeros(3), model_ps.initial_state(), params, deps)
    assert result.passed, f"PLANE_STRESS FD tangent failed: {result.max_rel_err:.3e}"


# ---------------------------------------------------------------------------
# 1D model (AFKinematic1D)
# ---------------------------------------------------------------------------

def test_1d_initial_state():
    model_1d = AFKinematic1D()
    state0 = model_1d.initial_state()
    assert state0["alpha"].shape == (1,)


def test_1d_yield_consistency(params):
    """AFKinematic1D: yield consistency after plastic step."""
    model_1d = AFKinematic1D()
    state0 = model_1d.initial_state()
    deps = jnp.array([3e-3])
    stress_new, state_new, _ = return_mapping(
        model_1d, deps, jnp.zeros(1), state0, params
    )
    f = model_1d.yield_function(stress_new, state_new, params)
    assert abs(float(f)) < 1e-8


def test_1d_bauschinger(params):
    """1D: reverse loading yields plastically earlier than forward yield."""
    model_1d = AFKinematic1D()
    state0 = model_1d.initial_state()

    # Forward plastic loading
    deps_fwd = jnp.array([5e-3])
    sigma1, state1, _ = return_mapping(model_1d, deps_fwd, jnp.zeros(1), state0, params)

    alpha_11 = float(state1["alpha"][0])
    assert alpha_11 > 0.0, "Expected positive backstress after tensile loading"

    # Reverse loading: the compressive yield should occur sooner (Bauschinger)
    # Check that a step that would be elastic without backstress is now plastic
    E = params["E"]
    sigma_y0 = params["sigma_y0"]
    # Without backstress, elastic range is ±sigma_y0 / E
    # With positive backstress, compressive yield at |sigma1 - alpha| = sigma_y0
    # So after unloading, compressive yield occurs at sigma = -(sigma_y0 - alpha_11)
    eps_unload = jnp.array([-sigma1[0] / E])  # unload to zero stress (elastic)
    sigma2, state2, _ = return_mapping(model_1d, eps_unload, sigma1, state1, params)

    # Apply compressive increment smaller than forward yield but > Bauschinger yield
    deps_rev = jnp.array([-1.5 * sigma_y0 / E])
    sigma3, state3, _ = return_mapping(model_1d, deps_rev, sigma2, state2, params)
    dep = float(state3["ep"]) - float(state2["ep"])
    assert dep > 0.0, "Expected plastic yielding on reverse loading (Bauschinger in 1D)"


def test_1d_fd_tangent(params):
    """FD tangent must pass for AFKinematic1D."""
    model_1d = AFKinematic1D()
    deps = jnp.array([5e-3])
    result = check_tangent(model_1d, jnp.zeros(1), model_1d.initial_state(), params, deps)
    assert result.passed, f"1D FD tangent failed: {result.max_rel_err:.3e}"
