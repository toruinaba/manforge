"""Tests for the augmented (ntens+1+n_state) residual system.

The augmented system treats state variables as independent unknowns, enabling
models with implicit hardening laws where state_new cannot be expressed in
closed form as a function of (dlambda, stress, state_n, params).

Key verifications
-----------------
- uses_implicit_state detection via MRO
- Augmented NR produces the same converged solution as the explicit NR path
  (validated by recasting the AF model as an implicit model whose residual
  equations are algebraically identical to the explicit update)
- FD tangent vs AD tangent agreement for the implicit path
- Yield surface consistency after plastic steps
- Correct dimensionality handling (3D, PLANE_STRAIN, PLANE_STRESS)
"""

import pytest
import numpy as np
import jax.numpy as jnp

import manforge  # enables JAX float64
from manforge.models.af_kinematic import AFKinematic3D, AFKinematicPS
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.core.return_mapping import return_mapping
from manforge.core.stress_state import PLANE_STRAIN, PLANE_STRESS
from manforge.utils.smooth import smooth_abs
from manforge.verification.fd_check import check_tangent


# ---------------------------------------------------------------------------
# Test model: AF kinematic hardening recast in purely implicit residual form
#
# The explicit update for alpha is:
#   alpha_new = (alpha_n + C_k * dlambda * n_hat) / (1 + gamma * dlambda)
#
# Rearranged as a residual:
#   R_alpha = alpha_new * (1 + gamma * dlambda) - alpha_n - C_k * dlambda * n_hat = 0
#
# Note: n_hat here is evaluated at (stress - alpha_n), i.e. the OLD backstress,
# exactly as in the explicit path.  This makes the two paths algebraically
# identical at convergence, so we can validate exact agreement.
# ---------------------------------------------------------------------------

class _AFKinematicImplicit3D(AFKinematic3D):
    """AF kinematic 3D model with hardening expressed as an implicit residual.

    Mathematically identical to AFKinematic3D at convergence — used to
    validate the augmented residual machinery without introducing model error.
    """

    def hardening_residual(self, state_new, dlambda, stress, state_n, params):
        alpha_n = state_n["alpha"]
        xi = stress - alpha_n
        s_xi = self._dev(xi)
        vm_safe = smooth_abs(self._vonmises(xi))
        n_hat = s_xi / vm_safe

        scale = 1.0 + params["gamma"] * dlambda
        R_alpha = state_new["alpha"] * scale - alpha_n - params["C_k"] * dlambda * n_hat
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        return {"alpha": R_alpha, "ep": R_ep}


class _AFKinematicImplicitPS(AFKinematicPS):
    """Plane-stress variant of the implicit AF model."""

    def hardening_residual(self, state_new, dlambda, stress, state_n, params):
        alpha_n = state_n["alpha"]
        xi = stress - alpha_n
        s_xi = self._dev(xi)
        vm_safe = smooth_abs(self._vonmises(xi))
        n_hat = s_xi / vm_safe

        scale = 1.0 + params["gamma"] * dlambda
        R_alpha = state_new["alpha"] * scale - alpha_n - params["C_k"] * dlambda * n_hat
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        return {"alpha": R_alpha, "ep": R_ep}


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def params():
    return {
        "E": 210000.0,
        "nu": 0.3,
        "sigma_y0": 250.0,
        "C_k": 10000.0,
        "gamma": 100.0,
    }


# ---------------------------------------------------------------------------
# uses_implicit_state detection
# ---------------------------------------------------------------------------

def test_uses_implicit_state_j2_is_false():
    assert J2Isotropic3D().uses_implicit_state is False


def test_uses_implicit_state_af_is_false():
    assert AFKinematic3D().uses_implicit_state is False


def test_uses_implicit_state_implicit_af_is_true():
    assert _AFKinematicImplicit3D().uses_implicit_state is True


def test_uses_implicit_state_implicit_ps_is_true():
    assert _AFKinematicImplicitPS().uses_implicit_state is True


# ---------------------------------------------------------------------------
# Augmented NR matches explicit NR — 3D
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deps_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
    [2e-3, 0.0, 0.0, 2e-3, 0.0, 0.0],
])
def test_implicit_matches_explicit_stress_3d(params, deps_vec):
    """Implicit AF path must produce the same stress as the explicit AF path."""
    deps = jnp.array(deps_vec)
    stress0 = jnp.zeros(6)

    explicit_model = AFKinematic3D()
    implicit_model = _AFKinematicImplicit3D()
    state0 = explicit_model.initial_state()

    stress_exp, _, _ = return_mapping(explicit_model, deps, stress0, state0, params)
    stress_imp, _, _ = return_mapping(implicit_model, deps, stress0, state0, params)

    np.testing.assert_allclose(
        np.array(stress_imp), np.array(stress_exp), atol=1e-7,
        err_msg=f"Implicit stress differs from explicit for deps={deps_vec}"
    )


@pytest.mark.parametrize("deps_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
])
def test_implicit_matches_explicit_state_3d(params, deps_vec):
    """Implicit AF path must produce the same state as the explicit AF path."""
    deps = jnp.array(deps_vec)
    stress0 = jnp.zeros(6)

    explicit_model = AFKinematic3D()
    implicit_model = _AFKinematicImplicit3D()
    state0 = explicit_model.initial_state()

    _, state_exp, _ = return_mapping(explicit_model, deps, stress0, state0, params)
    _, state_imp, _ = return_mapping(implicit_model, deps, stress0, state0, params)

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
def test_implicit_yield_consistency_3d(params, deps_vec):
    """After a plastic step via the implicit path, stress must lie on the yield surface."""
    model = _AFKinematicImplicit3D()
    state0 = model.initial_state()
    deps = jnp.array(deps_vec)

    stress_new, state_new, _ = return_mapping(model, deps, jnp.zeros(6), state0, params)
    f = model.yield_function(stress_new, state_new, params)
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
def test_implicit_fd_tangent_virgin_3d(params, deps_vec):
    """Augmented consistent tangent must match FD for the implicit AF model."""
    model = _AFKinematicImplicit3D()
    state0 = model.initial_state()
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


def test_implicit_fd_tangent_prestressed_3d(params):
    """FD tangent from a pre-strained state via the implicit AF path."""
    model = _AFKinematicImplicit3D()
    state0 = model.initial_state()

    # First step: push into plasticity
    deps1 = jnp.zeros(6).at[0].set(3e-3)
    stress1, state1, _ = return_mapping(model, deps1, jnp.zeros(6), state0, params)

    # Second step: verify FD tangent
    deps2 = jnp.zeros(6).at[0].set(1e-3)
    result = check_tangent(model, stress1, state1, params, deps2)
    assert result.passed, f"Pre-stressed FD tangent failed: {result.max_rel_err:.3e}"


# ---------------------------------------------------------------------------
# Plane-stress stress state
# ---------------------------------------------------------------------------

@pytest.fixture
def params_ps():
    return {
        "E": 210000.0,
        "nu": 0.3,
        "sigma_y0": 250.0,
        "C_k": 10000.0,
        "gamma": 100.0,
    }


def test_implicit_yield_consistency_plane_stress(params_ps):
    """Implicit AF plane-stress model: yield surface consistency."""
    model = _AFKinematicImplicitPS()
    state0 = model.initial_state()
    deps = jnp.array([2e-3, 0.0, 0.0])

    stress_new, state_new, _ = return_mapping(model, deps, jnp.zeros(3), state0, params_ps)
    f = model.yield_function(stress_new, state_new, params_ps)
    assert abs(float(f)) < 1e-8, f"PS yield not satisfied: f = {float(f):.3e}"


def test_implicit_fd_tangent_plane_stress(params_ps):
    """Augmented consistent tangent must match FD for plane-stress implicit AF."""
    model = _AFKinematicImplicitPS()
    state0 = model.initial_state()
    deps = jnp.array([2e-3, 0.0, 0.0])

    result = check_tangent(model, jnp.zeros(3), state0, params_ps, deps)
    assert result.passed, (
        f"PS FD tangent check failed: max_rel_err = {result.max_rel_err:.3e}"
    )


def test_implicit_matches_explicit_stress_plane_stress(params_ps):
    """Implicit AF PS path must produce the same stress as the explicit AF PS path."""
    deps = jnp.array([2e-3, -1e-3, 0.0])

    explicit_model = AFKinematicPS()
    implicit_model = _AFKinematicImplicitPS()
    state0 = explicit_model.initial_state()

    stress_exp, _, _ = return_mapping(explicit_model, deps, jnp.zeros(3), state0, params_ps)
    stress_imp, _, _ = return_mapping(implicit_model, deps, jnp.zeros(3), state0, params_ps)

    np.testing.assert_allclose(
        np.array(stress_imp), np.array(stress_exp), atol=1e-7,
    )


# ---------------------------------------------------------------------------
# Elastic step: implicit path falls through to elastic tangent = C
# ---------------------------------------------------------------------------

def test_implicit_elastic_step_3d(params):
    """Elastic step via implicit-state model returns C as tangent, state unchanged."""
    model = _AFKinematicImplicit3D()
    state0 = model.initial_state()
    C = model.elastic_stiffness(params)
    deps = jnp.array([0.5e-3, 0.0, 0.0, 0.0, 0.0, 0.0])

    stress_new, state_new, ddsdde = return_mapping(model, deps, jnp.zeros(6), state0, params)

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
def test_implicit_tangent_matches_explicit_3d(params, deps_vec):
    """Consistent tangent from the augmented system must match the explicit path tangent."""
    deps = jnp.array(deps_vec)
    stress0 = jnp.zeros(6)

    explicit_model = AFKinematic3D()
    implicit_model = _AFKinematicImplicit3D()
    state0 = explicit_model.initial_state()

    _, _, ddsdde_exp = return_mapping(explicit_model, deps, stress0, state0, params)
    _, _, ddsdde_imp = return_mapping(implicit_model, deps, stress0, state0, params)

    np.testing.assert_allclose(
        np.array(ddsdde_imp), np.array(ddsdde_exp), atol=1e-6,
        err_msg=f"Implicit tangent differs from explicit for deps={deps_vec}"
    )


# ---------------------------------------------------------------------------
# PLANE_STRAIN stress state
# ---------------------------------------------------------------------------

@pytest.fixture
def params_pe():
    return {
        "E": 210000.0,
        "nu": 0.3,
        "sigma_y0": 250.0,
        "C_k": 10000.0,
        "gamma": 100.0,
    }


class _AFKinematicImplicitPE(AFKinematic3D):
    """Plane-strain variant of the implicit AF model (uses MaterialModel3D with PLANE_STRAIN)."""

    def __init__(self):
        super().__init__(stress_state=PLANE_STRAIN)

    def hardening_residual(self, state_new, dlambda, stress, state_n, params):
        alpha_n = state_n["alpha"]
        xi = stress - alpha_n
        s_xi = self._dev(xi)
        vm_safe = smooth_abs(self._vonmises(xi))
        n_hat = s_xi / vm_safe

        scale = 1.0 + params["gamma"] * dlambda
        R_alpha = state_new["alpha"] * scale - alpha_n - params["C_k"] * dlambda * n_hat
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        return {"alpha": R_alpha, "ep": R_ep}


def test_uses_implicit_state_implicit_pe_is_true():
    assert _AFKinematicImplicitPE().uses_implicit_state is True


def test_implicit_yield_consistency_plane_strain(params_pe):
    """Implicit AF plane-strain model: yield surface consistency."""
    model = _AFKinematicImplicitPE()
    state0 = model.initial_state()
    deps = jnp.array([2e-3, 0.0, 0.0, 0.0])

    stress_new, state_new, _ = return_mapping(model, deps, jnp.zeros(4), state0, params_pe)
    f = model.yield_function(stress_new, state_new, params_pe)
    assert abs(float(f)) < 1e-8, f"PE yield not satisfied: f = {float(f):.3e}"


def test_implicit_fd_tangent_plane_strain(params_pe):
    """Augmented consistent tangent must match FD for plane-strain implicit AF."""
    model = _AFKinematicImplicitPE()
    state0 = model.initial_state()
    deps = jnp.array([2e-3, 0.0, 0.0, 0.0])

    result = check_tangent(model, jnp.zeros(4), state0, params_pe, deps)
    assert result.passed, (
        f"PE FD tangent check failed: max_rel_err = {result.max_rel_err:.3e}"
    )


def test_implicit_matches_explicit_plane_strain(params_pe):
    """Implicit AF plane-strain path must produce the same stress as the explicit path."""
    deps = jnp.array([2e-3, -1e-3, 0.0, 1e-3])

    explicit_model = AFKinematic3D(stress_state=PLANE_STRAIN)
    implicit_model = _AFKinematicImplicitPE()
    state0 = explicit_model.initial_state()

    stress_exp, _, ddsdde_exp = return_mapping(
        explicit_model, deps, jnp.zeros(4), state0, params_pe
    )
    stress_imp, _, ddsdde_imp = return_mapping(
        implicit_model, deps, jnp.zeros(4), state0, params_pe
    )

    np.testing.assert_allclose(
        np.array(stress_imp), np.array(stress_exp), atol=1e-7,
    )
    np.testing.assert_allclose(
        np.array(ddsdde_imp), np.array(ddsdde_exp), atol=1e-6,
    )
