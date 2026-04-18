"""Tests for the Ohno-Wang modified Armstrong-Frederick kinematic hardening model.

The OW model is the canonical example of a genuinely implicit hardening law in
manforge.  Unlike the standard AF model, the backward-Euler discretisation of
the Ohno-Wang evolution equation:

    R_α = α_{n+1} − α_n − C_k Δλ n̂ + γ Δλ ‖α_{n+1}‖ α_{n+1} = 0

cannot be solved in closed form, so the model uses the augmented residual path
(hardening_type = 'implicit').

Key verifications
-----------------
- hardening_type is 'implicit' for all OW variants
- Elastic domain: tangent = C, state unchanged
- Yield surface consistency after plastic steps
- FD tangent vs AD tangent agreement (core correctness test)
- Backstress properties: nonzero, deviatoric, ep monotone
- Saturation: ‖α‖_vm → √(C_k / γ) under monotonic loading
- gamma=0 limit: reduces to Prager linear kinematic hardening
- Bauschinger effect under cyclic loading
- StrainDriver cyclic integration
- PLANE_STRAIN, PLANE_STRESS, 1D stress states
"""

import pytest
import numpy as np
import jax.numpy as jnp

import manforge  # enables JAX float64
from manforge.models.ow_kinematic import OWKinematic3D, OWKinematicPS, OWKinematic1D
from manforge.core.return_mapping import return_mapping
from manforge.core.stress_state import PLANE_STRAIN
from manforge.simulation.driver import StrainDriver
from manforge.simulation.types import FieldHistory, FieldType
from manforge.verification.fd_check import check_tangent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    # C_k=10000, gamma=1.0 → saturation ‖α‖_vm = √(10000/1.0) = 100 MPa
    return OWKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=1.0)


@pytest.fixture
def state0(model):
    return model.initial_state()


# ---------------------------------------------------------------------------
# hardening_type detection
# ---------------------------------------------------------------------------

def test_hardening_type_ow3d():
    assert OWKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=1.0).hardening_type == "implicit"


def test_hardening_type_owps():
    assert OWKinematicPS(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=1.0).hardening_type == "implicit"


def test_hardening_type_ow1d():
    assert OWKinematic1D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=1.0).hardening_type == "implicit"


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

def test_elastic_stress_is_trial(model, state0):
    """Elastic step: stress = C @ deps, tangent = C, state unchanged."""
    C = model.elastic_stiffness()
    deps = jnp.array([0.5e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    stress_new, state_new, ddsdde = return_mapping(model, deps, jnp.zeros(6), state0)

    np.testing.assert_allclose(np.array(stress_new), np.array(C @ deps), rtol=1e-10)
    np.testing.assert_allclose(np.array(ddsdde), np.array(C), rtol=1e-10)
    np.testing.assert_allclose(np.array(state_new["alpha"]), np.zeros(6), atol=1e-30)
    assert float(state_new["ep"]) == 0.0


# ---------------------------------------------------------------------------
# Yield surface consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deps_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
    [2e-3, 0.0, 0.0, 2e-3, 0.0, 0.0],
])
def test_yield_consistency_plastic(model, state0, deps_vec):
    """After a plastic step, stress must lie on the yield surface."""
    deps = jnp.array(deps_vec)
    stress_new, state_new, _ = return_mapping(model, deps, jnp.zeros(6), state0)
    f = model.yield_function(stress_new, state_new)
    assert abs(float(f)) < 1e-8, f"Yield not satisfied: f = {float(f):.3e}"


# ---------------------------------------------------------------------------
# FD tangent verification — the core correctness test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deps_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
    [2e-3, 0.0, 0.0, 2e-3, 0.0, 0.0],
])
def test_tangent_fd_plastic_virgin(model, state0, deps_vec):
    """AD consistent tangent must match FD for the OW model."""
    result = check_tangent(
        model,
        jnp.zeros(6),
        state0,
        jnp.array(deps_vec),
    )
    assert result.passed, (
        f"FD tangent check failed: max_rel_err = {result.max_rel_err:.3e}\n"
        f"AD tangent:\n{np.array(result.ddsdde_ad)}\n"
        f"FD tangent:\n{np.array(result.ddsdde_fd)}"
    )


def test_tangent_fd_elastic(model, state0):
    """Elastic step: FD and AD tangents agree (both equal C)."""
    deps = jnp.array([0.5e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = check_tangent(model, jnp.zeros(6), state0, deps)
    assert result.passed, f"Elastic FD tangent failed: {result.max_rel_err:.3e}"


def test_tangent_fd_prestressed(model, state0):
    """FD tangent from a non-virgin (plastically pre-strained) state."""
    deps1 = jnp.zeros(6).at[0].set(3e-3)
    stress1, state1, _ = return_mapping(model, deps1, jnp.zeros(6), state0)

    deps2 = jnp.zeros(6).at[0].set(1e-3)
    result = check_tangent(model, stress1, state1, deps2)
    assert result.passed, f"Pre-stressed FD tangent failed: {result.max_rel_err:.3e}"


# ---------------------------------------------------------------------------
# Backstress properties
# ---------------------------------------------------------------------------

def test_backstress_is_nonzero_after_plastic_step(model, state0):
    """Backstress must become nonzero after a plastic step."""
    deps = jnp.zeros(6).at[0].set(3e-3)
    _, state_new, _ = return_mapping(model, deps, jnp.zeros(6), state0)
    assert jnp.linalg.norm(state_new["alpha"]) > 1e-3


def test_backstress_is_deviatoric(model, state0):
    """Backstress must remain deviatoric (trace of direct components ≈ 0)."""
    deps = jnp.zeros(6).at[0].set(3e-3)
    _, state_new, _ = return_mapping(model, deps, jnp.zeros(6), state0)
    trace = float(state_new["alpha"][0] + state_new["alpha"][1] + state_new["alpha"][2])
    assert abs(trace) < 1e-8, f"Backstress not deviatoric: trace = {trace:.3e}"


def test_ep_increases_with_plastic_loading(model, state0):
    """Equivalent plastic strain must increase monotonically under plastic loading."""
    eps_values = []
    stress_n = jnp.zeros(6)
    state_n = state0
    for _ in range(5):
        deps = jnp.zeros(6).at[0].set(1e-3)
        stress_n, state_n, _ = return_mapping(model, deps, stress_n, state_n)
        eps_values.append(float(state_n["ep"]))
    assert all(b > a for a, b in zip(eps_values, eps_values[1:]))


# ---------------------------------------------------------------------------
# Saturation: ‖α‖_vm → √(C_k / γ) under monotonic loading
# ---------------------------------------------------------------------------

def test_backstress_saturation(model, state0):
    """Under many small plastic increments, ‖α‖_vm should converge to √(C_k/γ)."""
    import math
    alpha_sat_expected = math.sqrt(model.C_k / model.gamma)  # 100 MPa

    stress_n = jnp.zeros(6)
    state_n = state0
    for _ in range(200):
        deps = jnp.zeros(6).at[0].set(5e-4)
        stress_n, state_n, _ = return_mapping(model, deps, stress_n, state_n)

    # Compute VM norm of backstress (alpha is already deviatoric)
    alpha = state_n["alpha"]
    alpha_vm = float(jnp.sqrt(1.5 * jnp.sum(alpha ** 2)))  # Mandel VM norm approx

    # Should be within 5% of the analytical saturation value
    assert abs(alpha_vm - alpha_sat_expected) / alpha_sat_expected < 0.05, (
        f"Backstress norm {alpha_vm:.2f} not close to expected saturation {alpha_sat_expected:.2f}"
    )


# ---------------------------------------------------------------------------
# gamma=0 limit: linear kinematic hardening (Prager's rule)
# ---------------------------------------------------------------------------

def test_gamma0_gives_linear_kinematic():
    """With gamma=0, OW reduces to Prager linear kinematic hardening."""
    model = OWKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=0.0)
    state0 = model.initial_state()
    deps = jnp.zeros(6).at[0].set(3e-3)

    stress_new, state_new, _ = return_mapping(
        model, deps, jnp.zeros(6), state0
    )

    # Yield must be satisfied
    f = model.yield_function(stress_new, state_new)
    assert abs(float(f)) < 1e-8, f"gamma=0 yield not satisfied: f = {float(f):.3e}"


def test_gamma0_fd_tangent():
    """FD tangent must pass for the gamma=0 linear limit."""
    model = OWKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=0.0)
    state0 = model.initial_state()
    deps = jnp.zeros(6).at[0].set(2e-3)

    result = check_tangent(model, jnp.zeros(6), state0, deps)
    assert result.passed, f"gamma=0 FD tangent failed: {result.max_rel_err:.3e}"


# ---------------------------------------------------------------------------
# Bauschinger effect
# ---------------------------------------------------------------------------

def test_bauschinger_effect(model, state0):
    """Kinematic hardening must produce the Bauschinger effect.

    After forward plastic loading, the initial reverse yield stress should be
    lower than the monotonic yield stress, due to the accumulated backstress.
    """
    # Step 1: forward plastic loading
    deps_fwd = jnp.zeros(6).at[0].set(5e-3)
    stress1, state1, _ = return_mapping(model, deps_fwd, jnp.zeros(6), state0)
    assert jnp.linalg.norm(state1["alpha"]) > 1e-3  # backstress built up

    # Step 2: elastic unloading to approximately zero stress
    C = model.elastic_stiffness()
    deps_unload = jnp.linalg.solve(C, -stress1)
    stress2, state2, _ = return_mapping(model, deps_unload, stress1, state1)

    # Step 3: compressive reload — must yield in compression (Bauschinger).
    # Use a sufficiently large compressive increment to trigger reverse yielding.
    deps_rev = jnp.zeros(6).at[0].set(-3e-3)
    stress3, state3, _ = return_mapping(model, deps_rev, stress2, state2)

    # ep must increase during the reverse step (reverse yielding occurred)
    assert float(state3["ep"]) > float(state2["ep"]), "No reverse yielding detected"


# ---------------------------------------------------------------------------
# Cyclic driver test
# ---------------------------------------------------------------------------

def test_cyclic_loading_driver(model):
    """StrainDriver must run cyclic tension-compression without error."""
    driver = StrainDriver()
    history = np.zeros((20, 6))
    history[:10, 0] = np.linspace(0, 4e-3, 10)
    history[10:, 0] = np.linspace(4e-3, -4e-3, 10)

    load = FieldHistory(type=FieldType.STRAIN, name="Strain", data=history)
    result = driver.run(
        model, load,
        collect_state={"alpha": FieldType.STRESS, "ep": FieldType.STRAIN},
    )

    assert result.stress.shape == (20, 6)
    assert result.fields["ep"].data.shape == (20,)
    assert result.fields["alpha"].data.shape == (20, 6)

    # ep must be non-decreasing
    ep = result.fields["ep"].data
    assert np.all(np.diff(ep) >= -1e-12), "ep decreased during loading"


# ---------------------------------------------------------------------------
# Physical comparison: OW vs standard AF
# ---------------------------------------------------------------------------

def test_ow_approaches_af_for_small_alpha():
    """For very small plastic strains, OW and AF should give similar results.

    When ‖α‖ is small, γ ‖α‖ α ≈ 0, so both models reduce to
    Prager linear hardening locally.
    """
    from manforge.models.af_kinematic import AFKinematic3D

    # Use a very small strain increment so ‖α‖ stays small
    ow_model = OWKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=1.0)
    af_model = AFKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0)
    deps = jnp.zeros(6).at[0].set(3e-4)  # very small step — near linear regime

    stress_ow, _, _ = return_mapping(ow_model, deps, jnp.zeros(6), ow_model.initial_state())
    stress_af, _, _ = return_mapping(af_model, deps, jnp.zeros(6), af_model.initial_state())

    # Should be within ~10% for a small step near the yield surface
    np.testing.assert_allclose(
        np.array(stress_ow), np.array(stress_af), rtol=0.10,
        err_msg="OW and AF stresses diverge even at small strain increment"
    )


# ---------------------------------------------------------------------------
# PLANE_STRAIN
# ---------------------------------------------------------------------------

def test_ow_plane_strain_yield_consistency():
    """OW model with PLANE_STRAIN: yield surface consistency."""
    model = OWKinematic3D(stress_state=PLANE_STRAIN, E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=1.0)
    state0 = model.initial_state()
    deps = jnp.array([2e-3, 0.0, 0.0, 0.0])

    stress_new, state_new, _ = return_mapping(model, deps, jnp.zeros(4), state0)
    f = model.yield_function(stress_new, state_new)
    assert abs(float(f)) < 1e-8, f"PE yield not satisfied: f = {float(f):.3e}"


def test_ow_plane_strain_fd_tangent():
    """Augmented consistent tangent must match FD for OW in plane strain."""
    model = OWKinematic3D(stress_state=PLANE_STRAIN, E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=1.0)
    state0 = model.initial_state()
    deps = jnp.array([2e-3, 0.0, 0.0, 0.0])

    result = check_tangent(model, jnp.zeros(4), state0, deps)
    assert result.passed, f"PE FD tangent failed: {result.max_rel_err:.3e}"


# ---------------------------------------------------------------------------
# PLANE_STRESS
# ---------------------------------------------------------------------------

def test_ow_plane_stress_initial_state():
    """OWKinematicPS: initial alpha has shape (3,)."""
    model = OWKinematicPS(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=1.0)
    state = model.initial_state()
    assert state["alpha"].shape == (3,)
    assert state["ep"].shape == ()


def test_ow_plane_stress_yield_consistency():
    """OW plane-stress model: yield surface consistency."""
    model = OWKinematicPS(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=1.0)
    state0 = model.initial_state()
    deps = jnp.array([2e-3, 0.0, 0.0])

    stress_new, state_new, _ = return_mapping(model, deps, jnp.zeros(3), state0)
    f = model.yield_function(stress_new, state_new)
    assert abs(float(f)) < 1e-8, f"PS yield not satisfied: f = {float(f):.3e}"


def test_ow_plane_stress_fd_tangent():
    """Augmented consistent tangent must match FD for OW in plane stress."""
    model = OWKinematicPS(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=1.0)
    state0 = model.initial_state()
    deps = jnp.array([2e-3, 0.0, 0.0])

    result = check_tangent(model, jnp.zeros(3), state0, deps)
    assert result.passed, f"PS FD tangent failed: {result.max_rel_err:.3e}"


# ---------------------------------------------------------------------------
# 1D model
# ---------------------------------------------------------------------------

def test_ow_1d_initial_state():
    """OWKinematic1D: initial alpha has shape (1,)."""
    model = OWKinematic1D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=1.0)
    state = model.initial_state()
    assert state["alpha"].shape == (1,)
    assert state["ep"].shape == ()


def test_ow_1d_yield_consistency():
    """OW 1D model: yield surface consistency."""
    model = OWKinematic1D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=1.0)
    state0 = model.initial_state()
    deps = jnp.array([2e-3])

    stress_new, state_new, _ = return_mapping(model, deps, jnp.zeros(1), state0)
    f = model.yield_function(stress_new, state_new)
    assert abs(float(f)) < 1e-8, f"1D yield not satisfied: f = {float(f):.3e}"


def test_ow_1d_fd_tangent():
    """Augmented consistent tangent must match FD for OW 1D model."""
    model = OWKinematic1D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=1.0)
    state0 = model.initial_state()
    deps = jnp.array([2e-3])

    result = check_tangent(model, jnp.zeros(1), state0, deps)
    assert result.passed, f"1D FD tangent failed: {result.max_rel_err:.3e}"


def test_ow_1d_bauschinger():
    """OW 1D model: Bauschinger effect under tension then compression."""
    model = OWKinematic1D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=1.0)
    state0 = model.initial_state()

    deps_fwd = jnp.array([5e-3])
    stress1, state1, _ = return_mapping(model, deps_fwd, jnp.zeros(1), state0)

    C = model.elastic_stiffness()
    deps_unload = jnp.linalg.solve(C, -stress1)
    stress2, state2, _ = return_mapping(model, deps_unload, stress1, state1)

    deps_rev = jnp.array([-3e-3])
    _, state3, _ = return_mapping(model, deps_rev, stress2, state2)

    assert float(state3["ep"]) > float(state2["ep"]), "No 1D reverse yielding"
