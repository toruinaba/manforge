"""Tests for partial-implicit state declarations.

Validates that mixing explicit and implicit state variables works correctly:
- A model with only some states in implicit_state_names converges to the
  same solution as the fully-explicit counterpart.
- stress = Implicit(shape=NTENS) (σ as NR unknown) converges to the same
  solution as the default (σ derived via fixed-point).

Uses AF kinematic hardening as the reference because its state_residual has
a known closed-form solution, so all paths must agree at convergence.
"""

import pytest
import numpy as np
import autograd.numpy as anp

from manforge.core.state import Implicit, Explicit, NTENS
from manforge.models.af_kinematic import AFKinematic3D
from manforge.simulation.integrator import PythonIntegrator
from manforge.verification.fd_check import check_tangent


# ---------------------------------------------------------------------------
# Partial-implicit model: alpha is implicit, ep is explicit
# ---------------------------------------------------------------------------

class _AFAlphaImplicit(AFKinematic3D):
    """AF 3D with only alpha declared as an implicit state (ep remains explicit).

    MRO override: re-declare ``alpha`` as Implicit; ``ep`` and ``stress`` stay
    Explicit from parent.  update_state returns the explicit keys (stress, ep).
    state_residual returns only the implicit key (alpha).
    """

    alpha = Implicit(shape=NTENS, doc="backstress (implicit override)")

    def update_state(self, dlambda, state_n, state_trial):
        return [self.ep(state_n["ep"] + dlambda)]

    def state_residual(self, state_new, dlambda, state_n, state_trial, *, stress_trial):
        alpha_n = state_n["alpha"]
        stress = state_trial["stress"]
        xi = stress - alpha_n
        s_xi = self._dev(xi)
        vm_safe = self._vonmises(xi)
        n_hat = s_xi / vm_safe
        scale = 1.0 + self.gamma * dlambda
        R_alpha = state_new["alpha"] * scale - alpha_n - self.C_k * dlambda * n_hat
        return [self.alpha(R_alpha)]


# ---------------------------------------------------------------------------
# Partial-implicit with stress = Implicit: α implicit, ep explicit, σ in NR
# ---------------------------------------------------------------------------

class _AFAlphaImplicitStress(AFKinematic3D):
    """AF 3D with alpha implicit and σ as an independent NR unknown."""

    stress = Implicit(shape=NTENS, doc="Cauchy stress (implicit override)")
    alpha = Implicit(shape=NTENS, doc="backstress (implicit override)")

    def update_state(self, dlambda, state_n, state_trial):
        # stress is Implicit: only return the explicit ep key
        return [self.ep(state_n["ep"] + dlambda)]

    def state_residual(self, state_new, dlambda, state_n, state_trial, *, stress_trial):
        alpha_n = state_n["alpha"]
        stress = state_new["stress"]
        xi = stress - alpha_n
        s_xi = self._dev(xi)
        vm_safe = self._vonmises(xi)
        n_hat = s_xi / vm_safe
        scale = 1.0 + self.gamma * dlambda
        R_stress = self.default_stress_residual(state_new, dlambda, stress_trial)
        R_alpha = state_new["alpha"] * scale - alpha_n - self.C_k * dlambda * n_hat
        return [self.stress(R_stress), self.alpha(R_alpha)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PARAMS = dict(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0)


@pytest.fixture
def explicit_model():
    return AFKinematic3D(**_PARAMS)


@pytest.fixture
def partial_model():
    return _AFAlphaImplicit(**_PARAMS)


@pytest.fixture
def partial_stress_model():
    return _AFAlphaImplicitStress(**_PARAMS)


# ---------------------------------------------------------------------------
# API checks
# ---------------------------------------------------------------------------

def test_partial_implicit_api():
    m = _AFAlphaImplicit(**_PARAMS)
    assert m.implicit_state_names == ["alpha"]


def test_partial_implicit_stress_api():
    m = _AFAlphaImplicitStress(**_PARAMS)
    assert "alpha" in m.implicit_state_names
    assert "stress" in m.implicit_state_names


# ---------------------------------------------------------------------------
# Stress and state match between all three paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deps_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
])
def test_partial_implicit_stress_matches_explicit(explicit_model, partial_model, deps_vec):
    """Partial-implicit path must produce the same stress as the fully-explicit path."""
    deps = anp.array(deps_vec)
    stress0 = anp.zeros(6)
    state0 = explicit_model.initial_state()

    stress_exp = PythonIntegrator(explicit_model).stress_update(deps, stress0, state0).stress
    stress_imp = PythonIntegrator(partial_model).stress_update(deps, stress0, state0).stress

    np.testing.assert_allclose(
        np.array(stress_imp), np.array(stress_exp), atol=1e-7,
        err_msg=f"Partial-implicit stress differs from explicit for deps={deps_vec}"
    )


@pytest.mark.parametrize("deps_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
])
def test_partial_implicit_state_matches_explicit(explicit_model, partial_model, deps_vec):
    """Partial-implicit state (alpha, ep) must match the fully-explicit path."""
    deps = anp.array(deps_vec)
    stress0 = anp.zeros(6)
    state0 = explicit_model.initial_state()

    state_exp = PythonIntegrator(explicit_model).stress_update(deps, stress0, state0).state
    state_imp = PythonIntegrator(partial_model).stress_update(deps, stress0, state0).state

    np.testing.assert_allclose(
        np.array(state_imp["alpha"]), np.array(state_exp["alpha"]), atol=1e-7,
    )
    np.testing.assert_allclose(float(state_imp["ep"]), float(state_exp["ep"]), atol=1e-10)


@pytest.mark.parametrize("deps_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0],
])
def test_implicit_stress_flag_matches_no_flag(partial_model, partial_stress_model, deps_vec):
    """stress = Implicit must converge to the same result as default (Explicit) stress."""
    deps = anp.array(deps_vec)
    stress0 = anp.zeros(6)
    state0 = partial_model.initial_state()

    r_no_flag = PythonIntegrator(partial_model).stress_update(deps, stress0, state0)
    r_flag = PythonIntegrator(partial_stress_model).stress_update(deps, stress0, state0)

    np.testing.assert_allclose(
        np.array(r_flag.stress), np.array(r_no_flag.stress), atol=1e-7,
        err_msg=f"stress=Implicit stress differs for deps={deps_vec}"
    )
    np.testing.assert_allclose(
        np.array(r_flag.state["alpha"]), np.array(r_no_flag.state["alpha"]), atol=1e-7,
    )


# ---------------------------------------------------------------------------
# Yield surface consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deps_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
])
def test_partial_implicit_yield_consistency(partial_model, deps_vec):
    """After a plastic step, stress must lie on the yield surface."""
    state0 = partial_model.initial_state()
    deps = anp.array(deps_vec)

    _r = PythonIntegrator(partial_model).stress_update(deps, anp.zeros(6), state0)
    from manforge.core.state import State
    state_with_stress = dict(_r.state)
    state_with_stress["stress"] = _r.stress
    f = partial_model.yield_function(state_with_stress)
    assert abs(float(f)) < 1e-8, f"Yield not satisfied: f = {float(f):.3e}"


# ---------------------------------------------------------------------------
# FD tangent check
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("deps_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
])
def test_partial_implicit_fd_tangent(partial_model, deps_vec):
    """Consistent tangent must match FD for the partial-implicit model."""
    state0 = partial_model.initial_state()
    result = check_tangent(
        PythonIntegrator(partial_model),
        anp.zeros(6),
        state0,
        anp.array(deps_vec),
    )
    assert result.passed, (
        f"FD tangent check failed: max_rel_err = {result.max_rel_err:.3e}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("deps_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
])
def test_partial_implicit_stress_fd_tangent(partial_stress_model, deps_vec):
    """Consistent tangent must match FD for partial-implicit with stress=Implicit."""
    state0 = partial_stress_model.initial_state()
    result = check_tangent(
        PythonIntegrator(partial_stress_model),
        anp.zeros(6),
        state0,
        anp.array(deps_vec),
    )
    assert result.passed, (
        f"FD tangent check failed (implicit stress): max_rel_err = {result.max_rel_err:.3e}"
    )


# ---------------------------------------------------------------------------
# Tangent agreement between paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deps_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0],
])
def test_partial_implicit_tangent_matches_explicit(explicit_model, partial_model, deps_vec):
    """Consistent tangent from partial-implicit path must match explicit path."""
    deps = anp.array(deps_vec)
    stress0 = anp.zeros(6)
    state0 = explicit_model.initial_state()

    ddsdde_exp = PythonIntegrator(explicit_model).stress_update(deps, stress0, state0).ddsdde
    ddsdde_imp = PythonIntegrator(partial_model).stress_update(deps, stress0, state0).ddsdde

    np.testing.assert_allclose(
        np.array(ddsdde_imp), np.array(ddsdde_exp), atol=1e-6,
        err_msg=f"Partial-implicit tangent differs from explicit for deps={deps_vec}"
    )
