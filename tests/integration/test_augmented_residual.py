"""Tests for the implicit-state (ntens+1+n_state) residual system.

The implicit-state system treats state variables as independent NR unknowns,
enabling models with implicit hardening laws where state_new cannot be
expressed in closed form as a function of (dlambda, stress, state_n).

Key verifications
-----------------
- implicit_state_names / implicit_stress API detection via class variable
- Implicit NR produces the same converged solution as the explicit NR path
  (validated by recasting the AF model as an implicit model whose residual
  equations are algebraically identical to the explicit update)
- FD tangent vs AD tangent agreement for the implicit path
- Yield surface consistency after plastic steps
- Correct dimensionality handling (3D, PLANE_STRAIN, PLANE_STRESS)
"""

import pytest
import numpy as np
import autograd.numpy as anp

import manforge  # enables JAX float64
from manforge.models.af_kinematic import AFKinematic3D, AFKinematicPS
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.core.stress_state import PLANE_STRAIN, PLANE_STRESS
from manforge.simulation.integrator import PythonIntegrator
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
# API detection: explicit vs implicit state names
# ---------------------------------------------------------------------------

def test_j2_has_no_implicit_states(model):
    assert model.implicit_state_names == []
    assert model.implicit_stress is False


def test_af_has_no_implicit_states():
    m = AFKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0)
    assert m.implicit_state_names == []
    assert m.implicit_stress is False


def test_implicit_af_has_implicit_states(implicit_model):
    assert implicit_model.implicit_state_names == ["alpha", "ep"]
    assert implicit_model.implicit_stress is True


def test_implicit_ps_has_implicit_states(implicit_ps_model):
    assert implicit_ps_model.implicit_state_names == ["alpha", "ep"]
    assert implicit_ps_model.implicit_stress is True


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
    deps = anp.array(deps_vec)
    stress0 = anp.zeros(6)
    state0 = af_model.initial_state()

    stress_exp = PythonIntegrator(af_model).stress_update(deps, stress0, state0).stress
    stress_imp = PythonIntegrator(implicit_model).stress_update(deps, stress0, state0).stress

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
    deps = anp.array(deps_vec)
    stress0 = anp.zeros(6)
    state0 = af_model.initial_state()

    state_exp = PythonIntegrator(af_model).stress_update(deps, stress0, state0).state
    state_imp = PythonIntegrator(implicit_model).stress_update(deps, stress0, state0).state

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
    deps = anp.array(deps_vec)

    _r = PythonIntegrator(implicit_model).stress_update(deps, anp.zeros(6), state0)
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
        PythonIntegrator(implicit_model),
        anp.zeros(6),
        state0,
        anp.array(deps_vec),
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
    deps1 = (lambda _a: (_a.__setitem__(0, 3e-3), _a)[1])(np.zeros(6))
    _r1 = PythonIntegrator(implicit_model).stress_update(deps1, anp.zeros(6), state0)
    stress1, state1 = _r1.stress, _r1.state

    # Second step: verify FD tangent
    deps2 = (lambda _a: (_a.__setitem__(0, 1e-3), _a)[1])(np.zeros(6))
    result = check_tangent(PythonIntegrator(implicit_model), stress1, state1, deps2)
    assert result.passed, f"Pre-stressed FD tangent failed: {result.max_rel_err:.3e}"


# ---------------------------------------------------------------------------
# Plane-stress stress state
# ---------------------------------------------------------------------------

def test_augmented_yield_consistency_plane_stress(implicit_ps_model):
    """Implicit AF plane-stress model: yield surface consistency."""
    state0 = implicit_ps_model.initial_state()
    deps = anp.array([2e-3, 0.0, 0.0])

    _r = PythonIntegrator(implicit_ps_model).stress_update(deps, anp.zeros(3), state0)
    stress_new, state_new = _r.stress, _r.state
    f = implicit_ps_model.yield_function(stress_new, state_new)
    assert abs(float(f)) < 1e-8, f"PS yield not satisfied: f = {float(f):.3e}"


@pytest.mark.slow
def test_augmented_fd_tangent_plane_stress(implicit_ps_model):
    """Augmented consistent tangent must match FD for plane-stress implicit AF."""
    state0 = implicit_ps_model.initial_state()
    deps = anp.array([2e-3, 0.0, 0.0])

    result = check_tangent(PythonIntegrator(implicit_ps_model), anp.zeros(3), state0, deps)
    assert result.passed, (
        f"PS FD tangent check failed: max_rel_err = {result.max_rel_err:.3e}"
    )


@pytest.mark.slow
def test_augmented_matches_reduced_stress_plane_stress(implicit_ps_model):
    """Augmented AF PS path must produce the same stress as the reduced AF PS path."""
    deps = anp.array([2e-3, -1e-3, 0.0])

    explicit_model = AFKinematicPS(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0)
    state0 = explicit_model.initial_state()

    stress_exp = PythonIntegrator(explicit_model).stress_update(deps, anp.zeros(3), state0).stress
    stress_imp = PythonIntegrator(implicit_ps_model).stress_update(deps, anp.zeros(3), state0).stress

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
    deps = anp.array([0.5e-3, 0.0, 0.0, 0.0, 0.0, 0.0])

    _r = PythonIntegrator(implicit_model).stress_update(deps, anp.zeros(6), state0)
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
    deps = anp.array(deps_vec)
    stress0 = anp.zeros(6)
    state0 = af_model.initial_state()

    ddsdde_exp = PythonIntegrator(af_model).stress_update(deps, stress0, state0).ddsdde
    ddsdde_imp = PythonIntegrator(implicit_model).stress_update(deps, stress0, state0).ddsdde

    np.testing.assert_allclose(
        np.array(ddsdde_imp), np.array(ddsdde_exp), atol=1e-6,
        err_msg=f"Implicit tangent differs from explicit for deps={deps_vec}"
    )


# ---------------------------------------------------------------------------
# PLANE_STRAIN stress state
# ---------------------------------------------------------------------------

def test_implicit_pe_has_implicit_states():
    m = _AFKinematicImplicitPE()
    assert m.implicit_state_names == ["alpha", "ep"]
    assert m.implicit_stress is True


def test_augmented_yield_consistency_plane_strain():
    """Implicit AF plane-strain model: yield surface consistency."""
    model = _AFKinematicImplicitPE()
    state0 = model.initial_state()
    deps = anp.array([2e-3, 0.0, 0.0, 0.0])

    _r = PythonIntegrator(model).stress_update(deps, anp.zeros(4), state0)
    stress_new, state_new = _r.stress, _r.state
    f = model.yield_function(stress_new, state_new)
    assert abs(float(f)) < 1e-8, f"PE yield not satisfied: f = {float(f):.3e}"


@pytest.mark.slow
def test_augmented_fd_tangent_plane_strain():
    """Augmented consistent tangent must match FD for plane-strain implicit AF."""
    model = _AFKinematicImplicitPE()
    state0 = model.initial_state()
    deps = anp.array([2e-3, 0.0, 0.0, 0.0])

    result = check_tangent(PythonIntegrator(model), anp.zeros(4), state0, deps)
    assert result.passed, (
        f"PE FD tangent check failed: max_rel_err = {result.max_rel_err:.3e}"
    )


def test_augmented_matches_reduced_plane_strain():
    """Implicit AF plane-strain path must produce the same stress as the explicit path."""
    deps = anp.array([2e-3, -1e-3, 0.0, 1e-3])

    explicit_model = AFKinematic3D(stress_state=PLANE_STRAIN,
                                   E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0)
    implicit_model = _AFKinematicImplicitPE()
    state0 = explicit_model.initial_state()

    _r_exp = PythonIntegrator(explicit_model).stress_update(deps, anp.zeros(4), state0)
    stress_exp, ddsdde_exp = _r_exp.stress, _r_exp.ddsdde
    _r_imp = PythonIntegrator(implicit_model).stress_update(deps, anp.zeros(4), state0)
    stress_imp, ddsdde_imp = _r_imp.stress, _r_imp.ddsdde

    np.testing.assert_allclose(
        np.array(stress_imp), np.array(stress_exp), atol=1e-7,
    )
    np.testing.assert_allclose(
        np.array(ddsdde_imp), np.array(ddsdde_exp), atol=1e-6,
    )
