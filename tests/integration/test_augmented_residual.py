"""Tests for the augmented (ntens+1+n_state) residual system.

The augmented system treats state variables as independent unknowns, enabling
models with implicit hardening laws where state_new cannot be expressed in
closed form as a function of (dlambda, stress, state_n).

Key verifications
-----------------
- hardening_type detection via class variable
- Augmented NR produces the same converged solution as the reduced NR path
  (validated by recasting the AF model as an augmented model whose residual
  equations are algebraically identical to the reduced update)
- FD tangent vs AD tangent agreement for the augmented path
- Yield surface consistency after plastic steps
- Correct dimensionality handling (3D, PLANE_STRAIN, PLANE_STRESS)
"""

import pytest
import numpy as np
import jax.numpy as jnp

import manforge  # enables JAX float64
from manforge.models.af_kinematic import AFKinematic3D, AFKinematicPS
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.core.stress_update import stress_update
from manforge.core.stress_state import PLANE_STRAIN, PLANE_STRESS
from manforge.verification.fd_check import check_tangent
from tests.fixtures.implicit_models import (
    _AFKinematicImplicit3D, _AFKinematicImplicitPS, _AFKinematicImplicitPE,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def af_model():
    return AFKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0)


@pytest.fixture
def implicit_model():
    return _AFKinematicImplicit3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0)


@pytest.fixture
def implicit_ps_model():
    return _AFKinematicImplicitPS(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0)


# ---------------------------------------------------------------------------
# hardening_type detection
# ---------------------------------------------------------------------------

def test_hardening_type_j2_is_reduced(model):
    assert model.hardening_type == "reduced"


def test_hardening_type_af_is_reduced():
    assert AFKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0).hardening_type == "reduced"


def test_hardening_type_augmented_af_is_augmented(implicit_model):
    assert implicit_model.hardening_type == "augmented"


def test_hardening_type_augmented_ps_is_augmented(implicit_ps_model):
    assert implicit_ps_model.hardening_type == "augmented"


# ---------------------------------------------------------------------------
# Augmented NR matches explicit NR — 3D
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deps_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
    [2e-3, 0.0, 0.0, 2e-3, 0.0, 0.0],
])
def test_augmented_matches_reduced_stress_3d(af_model, implicit_model, deps_vec):
    """Augmented AF path must produce the same stress as the reduced AF path."""
    deps = jnp.array(deps_vec)
    stress0 = jnp.zeros(6)
    state0 = af_model.initial_state()

    stress_exp = stress_update(af_model, deps, stress0, state0).stress
    stress_imp = stress_update(implicit_model, deps, stress0, state0).stress

    np.testing.assert_allclose(
        np.array(stress_imp), np.array(stress_exp), atol=1e-7,
        err_msg=f"Implicit stress differs from explicit for deps={deps_vec}"
    )


@pytest.mark.parametrize("deps_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
])
def test_augmented_matches_reduced_state_3d(af_model, implicit_model, deps_vec):
    """Augmented AF path must produce the same state as the reduced AF path."""
    deps = jnp.array(deps_vec)
    stress0 = jnp.zeros(6)
    state0 = af_model.initial_state()

    state_exp = stress_update(af_model, deps, stress0, state0).state
    state_imp = stress_update(implicit_model, deps, stress0, state0).state

    np.testing.assert_allclose(
        np.array(state_imp["alpha"]), np.array(state_exp["alpha"]), atol=1e-7,
        err_msg="Implicit alpha differs from explicit"
    )
    np.testing.assert_allclose(
        float(state_imp["ep"]), float(state_exp["ep"]), atol=1e-10,
        err_msg="Implicit ep differs from explicit"
    )


# ---------------------------------------------------------------------------
# Yield surface consistency — implicit path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deps_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
    [2e-3, 0.0, 0.0, 2e-3, 0.0, 0.0],
])
def test_augmented_yield_consistency_3d(implicit_model, deps_vec):
    """After a plastic step via the augmented path, stress must lie on the yield surface."""
    state0 = implicit_model.initial_state()
    deps = jnp.array(deps_vec)

    _r = stress_update(implicit_model, deps, jnp.zeros(6), state0)
    stress_new, state_new = _r.stress, _r.state
    f = implicit_model.yield_function(stress_new, state_new)
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
@pytest.mark.slow
def test_augmented_fd_tangent_virgin_3d(implicit_model, deps_vec):
    """Augmented consistent tangent must match FD for the augmented AF model."""
    state0 = implicit_model.initial_state()
    result = check_tangent(
        implicit_model,
        jnp.zeros(6),
        state0,
        jnp.array(deps_vec),
    )
    assert result.passed, (
        f"FD tangent check failed: max_rel_err = {result.max_rel_err:.3e}\n"
        f"AD tangent:\n{np.array(result.ddsdde_ad)}\n"
        f"FD tangent:\n{np.array(result.ddsdde_fd)}"
    )


@pytest.mark.slow
def test_augmented_fd_tangent_prestressed_3d(implicit_model):
    """FD tangent from a pre-strained state via the augmented AF path."""
    state0 = implicit_model.initial_state()

    # First step: push into plasticity
    deps1 = jnp.zeros(6).at[0].set(3e-3)
    _r1 = stress_update(implicit_model, deps1, jnp.zeros(6), state0)
    stress1, state1 = _r1.stress, _r1.state

    # Second step: verify FD tangent
    deps2 = jnp.zeros(6).at[0].set(1e-3)
    result = check_tangent(implicit_model, stress1, state1, deps2)
    assert result.passed, f"Pre-stressed FD tangent failed: {result.max_rel_err:.3e}"


# ---------------------------------------------------------------------------
# Plane-stress stress state
# ---------------------------------------------------------------------------

def test_augmented_yield_consistency_plane_stress(implicit_ps_model):
    """Implicit AF plane-stress model: yield surface consistency."""
    state0 = implicit_ps_model.initial_state()
    deps = jnp.array([2e-3, 0.0, 0.0])

    _r = stress_update(implicit_ps_model, deps, jnp.zeros(3), state0)
    stress_new, state_new = _r.stress, _r.state
    f = implicit_ps_model.yield_function(stress_new, state_new)
    assert abs(float(f)) < 1e-8, f"PS yield not satisfied: f = {float(f):.3e}"


@pytest.mark.slow
def test_augmented_fd_tangent_plane_stress(implicit_ps_model):
    """Augmented consistent tangent must match FD for plane-stress implicit AF."""
    state0 = implicit_ps_model.initial_state()
    deps = jnp.array([2e-3, 0.0, 0.0])

    result = check_tangent(implicit_ps_model, jnp.zeros(3), state0, deps)
    assert result.passed, (
        f"PS FD tangent check failed: max_rel_err = {result.max_rel_err:.3e}"
    )


@pytest.mark.slow
def test_augmented_matches_reduced_stress_plane_stress(implicit_ps_model):
    """Augmented AF PS path must produce the same stress as the reduced AF PS path."""
    deps = jnp.array([2e-3, -1e-3, 0.0])

    explicit_model = AFKinematicPS(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0)
    state0 = explicit_model.initial_state()

    stress_exp = stress_update(explicit_model, deps, jnp.zeros(3), state0).stress
    stress_imp = stress_update(implicit_ps_model, deps, jnp.zeros(3), state0).stress

    np.testing.assert_allclose(
        np.array(stress_imp), np.array(stress_exp), atol=1e-7,
    )


# ---------------------------------------------------------------------------
# Elastic step: implicit path falls through to elastic tangent = C
# ---------------------------------------------------------------------------

def test_augmented_elastic_step_3d(implicit_model):
    """Elastic step via augmented-state model returns C as tangent, state unchanged."""
    state0 = implicit_model.initial_state()
    C = implicit_model.elastic_stiffness()
    deps = jnp.array([0.5e-3, 0.0, 0.0, 0.0, 0.0, 0.0])

    _r = stress_update(implicit_model, deps, jnp.zeros(6), state0)
    stress_new, state_new, ddsdde = _r.stress, _r.state, _r.ddsdde

    np.testing.assert_allclose(np.array(stress_new), np.array(C @ deps), rtol=1e-10)
    np.testing.assert_allclose(np.array(ddsdde), np.array(C), rtol=1e-10)
    np.testing.assert_allclose(np.array(state_new["alpha"]), np.zeros(6), atol=1e-30)
    assert float(state_new["ep"]) == 0.0


# ---------------------------------------------------------------------------
# Tangent: explicit vs implicit direct comparison
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deps_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
])
def test_augmented_tangent_matches_reduced_3d(af_model, implicit_model, deps_vec):
    """Consistent tangent from the augmented system must match the reduced path tangent."""
    deps = jnp.array(deps_vec)
    stress0 = jnp.zeros(6)
    state0 = af_model.initial_state()

    ddsdde_exp = stress_update(af_model, deps, stress0, state0).ddsdde
    ddsdde_imp = stress_update(implicit_model, deps, stress0, state0).ddsdde

    np.testing.assert_allclose(
        np.array(ddsdde_imp), np.array(ddsdde_exp), atol=1e-6,
        err_msg=f"Implicit tangent differs from explicit for deps={deps_vec}"
    )


# ---------------------------------------------------------------------------
# PLANE_STRAIN stress state
# ---------------------------------------------------------------------------

def test_hardening_type_augmented_pe_is_augmented():
    assert _AFKinematicImplicitPE().hardening_type == "augmented"


def test_augmented_yield_consistency_plane_strain():
    """Implicit AF plane-strain model: yield surface consistency."""
    model = _AFKinematicImplicitPE()
    state0 = model.initial_state()
    deps = jnp.array([2e-3, 0.0, 0.0, 0.0])

    _r = stress_update(model, deps, jnp.zeros(4), state0)
    stress_new, state_new = _r.stress, _r.state
    f = model.yield_function(stress_new, state_new)
    assert abs(float(f)) < 1e-8, f"PE yield not satisfied: f = {float(f):.3e}"


@pytest.mark.slow
def test_augmented_fd_tangent_plane_strain():
    """Augmented consistent tangent must match FD for plane-strain implicit AF."""
    model = _AFKinematicImplicitPE()
    state0 = model.initial_state()
    deps = jnp.array([2e-3, 0.0, 0.0, 0.0])

    result = check_tangent(model, jnp.zeros(4), state0, deps)
    assert result.passed, (
        f"PE FD tangent check failed: max_rel_err = {result.max_rel_err:.3e}"
    )


def test_augmented_matches_reduced_plane_strain():
    """Implicit AF plane-strain path must produce the same stress as the explicit path."""
    deps = jnp.array([2e-3, -1e-3, 0.0, 1e-3])

    explicit_model = AFKinematic3D(stress_state=PLANE_STRAIN,
                                   E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0)
    implicit_model = _AFKinematicImplicitPE()
    state0 = explicit_model.initial_state()

    _r_exp = stress_update(explicit_model, deps, jnp.zeros(4), state0)
    stress_exp, ddsdde_exp = _r_exp.stress, _r_exp.ddsdde
    _r_imp = stress_update(implicit_model, deps, jnp.zeros(4), state0)
    stress_imp, ddsdde_imp = _r_imp.stress, _r_imp.ddsdde

    np.testing.assert_allclose(
        np.array(stress_imp), np.array(stress_exp), atol=1e-7,
    )
    np.testing.assert_allclose(
        np.array(ddsdde_imp), np.array(ddsdde_exp), atol=1e-6,
    )
