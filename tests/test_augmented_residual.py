"""Tests for the augmented (ntens+1+n_state) residual system.

The augmented system treats state variables as independent unknowns, enabling
models with implicit hardening laws where state_new cannot be expressed in
closed form as a function of (dlambda, stress, state_n).

Key verifications
-----------------
- hardening_type detection via class variable
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
from manforge.core.stress_update import stress_update
from manforge.core.stress_state import PLANE_STRAIN, PLANE_STRESS
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

    hardening_type = "implicit"

    def hardening_residual(self, state_new, dlambda, stress, state_n):
        alpha_n = state_n["alpha"]
        xi = stress - alpha_n
        s_xi = self._dev(xi)
        vm_safe = self._vonmises(xi)
        n_hat = s_xi / vm_safe

        scale = 1.0 + self.gamma * dlambda
        R_alpha = state_new["alpha"] * scale - alpha_n - self.C_k * dlambda * n_hat
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        return {"alpha": R_alpha, "ep": R_ep}


class _AFKinematicImplicitPS(AFKinematicPS):
    """Plane-stress variant of the implicit AF model."""

    hardening_type = "implicit"

    def hardening_residual(self, state_new, dlambda, stress, state_n):
        alpha_n = state_n["alpha"]
        xi = stress - alpha_n
        s_xi = self._dev(xi)
        vm_safe = self._vonmises(xi)
        n_hat = s_xi / vm_safe

        scale = 1.0 + self.gamma * dlambda
        R_alpha = state_new["alpha"] * scale - alpha_n - self.C_k * dlambda * n_hat
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        return {"alpha": R_alpha, "ep": R_ep}


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

def test_hardening_type_j2_is_explicit():
    assert J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0).hardening_type == "explicit"


def test_hardening_type_af_is_explicit():
    assert AFKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0).hardening_type == "explicit"


def test_hardening_type_implicit_af_is_implicit(implicit_model):
    assert implicit_model.hardening_type == "implicit"


def test_hardening_type_implicit_ps_is_implicit(implicit_ps_model):
    assert implicit_ps_model.hardening_type == "implicit"


# ---------------------------------------------------------------------------
# Augmented NR matches explicit NR — 3D
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deps_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
    [2e-3, 0.0, 0.0, 2e-3, 0.0, 0.0],
])
def test_implicit_matches_explicit_stress_3d(af_model, implicit_model, deps_vec):
    """Implicit AF path must produce the same stress as the explicit AF path."""
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
def test_implicit_matches_explicit_state_3d(af_model, implicit_model, deps_vec):
    """Implicit AF path must produce the same state as the explicit AF path."""
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
def test_implicit_yield_consistency_3d(implicit_model, deps_vec):
    """After a plastic step via the implicit path, stress must lie on the yield surface."""
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
def test_implicit_fd_tangent_virgin_3d(implicit_model, deps_vec):
    """Augmented consistent tangent must match FD for the implicit AF model."""
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


def test_implicit_fd_tangent_prestressed_3d(implicit_model):
    """FD tangent from a pre-strained state via the implicit AF path."""
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

def test_implicit_yield_consistency_plane_stress(implicit_ps_model):
    """Implicit AF plane-stress model: yield surface consistency."""
    state0 = implicit_ps_model.initial_state()
    deps = jnp.array([2e-3, 0.0, 0.0])

    _r = stress_update(implicit_ps_model, deps, jnp.zeros(3), state0)
    stress_new, state_new = _r.stress, _r.state
    f = implicit_ps_model.yield_function(stress_new, state_new)
    assert abs(float(f)) < 1e-8, f"PS yield not satisfied: f = {float(f):.3e}"


def test_implicit_fd_tangent_plane_stress(implicit_ps_model):
    """Augmented consistent tangent must match FD for plane-stress implicit AF."""
    state0 = implicit_ps_model.initial_state()
    deps = jnp.array([2e-3, 0.0, 0.0])

    result = check_tangent(implicit_ps_model, jnp.zeros(3), state0, deps)
    assert result.passed, (
        f"PS FD tangent check failed: max_rel_err = {result.max_rel_err:.3e}"
    )


def test_implicit_matches_explicit_stress_plane_stress(implicit_ps_model):
    """Implicit AF PS path must produce the same stress as the explicit AF PS path."""
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

def test_implicit_elastic_step_3d(implicit_model):
    """Elastic step via implicit-state model returns C as tangent, state unchanged."""
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
def test_implicit_tangent_matches_explicit_3d(af_model, implicit_model, deps_vec):
    """Consistent tangent from the augmented system must match the explicit path tangent."""
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

class _AFKinematicImplicitPE(AFKinematic3D):
    """Plane-strain variant of the implicit AF model (uses MaterialModel3D with PLANE_STRAIN)."""

    hardening_type = "implicit"

    def __init__(self):
        super().__init__(stress_state=PLANE_STRAIN,
                         E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0)

    def hardening_residual(self, state_new, dlambda, stress, state_n):
        alpha_n = state_n["alpha"]
        xi = stress - alpha_n
        s_xi = self._dev(xi)
        vm_safe = self._vonmises(xi)
        n_hat = s_xi / vm_safe

        scale = 1.0 + self.gamma * dlambda
        R_alpha = state_new["alpha"] * scale - alpha_n - self.C_k * dlambda * n_hat
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        return {"alpha": R_alpha, "ep": R_ep}


def test_hardening_type_implicit_pe_is_implicit():
    assert _AFKinematicImplicitPE().hardening_type == "implicit"


def test_implicit_yield_consistency_plane_strain():
    """Implicit AF plane-strain model: yield surface consistency."""
    model = _AFKinematicImplicitPE()
    state0 = model.initial_state()
    deps = jnp.array([2e-3, 0.0, 0.0, 0.0])

    _r = stress_update(model, deps, jnp.zeros(4), state0)
    stress_new, state_new = _r.stress, _r.state
    f = model.yield_function(stress_new, state_new)
    assert abs(float(f)) < 1e-8, f"PE yield not satisfied: f = {float(f):.3e}"


def test_implicit_fd_tangent_plane_strain():
    """Augmented consistent tangent must match FD for plane-strain implicit AF."""
    model = _AFKinematicImplicitPE()
    state0 = model.initial_state()
    deps = jnp.array([2e-3, 0.0, 0.0, 0.0])

    result = check_tangent(model, jnp.zeros(4), state0, deps)
    assert result.passed, (
        f"PE FD tangent check failed: max_rel_err = {result.max_rel_err:.3e}"
    )


def test_implicit_matches_explicit_plane_strain():
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
