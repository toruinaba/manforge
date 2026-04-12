"""Tests for the J2Isotropic3D analytical return-mapping path.

Covers:
- plastic_corrector and analytical_tangent as standalone methods
- method="analytical" vs method="autodiff" agreement
- method="auto" selects the analytical path
- check_tangent with method="analytical" (FD verification of closed-form tangent)
- method="analytical" raises NotImplementedError on a model without hooks
"""

import jax.numpy as jnp
import pytest

import manforge  # noqa: F401
from manforge.core.return_mapping import return_mapping
from manforge.core.material import MaterialModel3D
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.verification.fd_check import check_tangent


@pytest.fixture
def model():
    return J2Isotropic3D()


# ---------------------------------------------------------------------------
# plastic_corrector — standalone
# ---------------------------------------------------------------------------

def test_plastic_corrector_elastic_path_not_called(model, steel_params):
    """plastic_corrector is only invoked in the plastic regime.

    When return_mapping detects an elastic step (f_trial ≤ 0), it returns
    before calling plastic_corrector.  This test verifies the elastic step
    still works under method='analytical'.
    """
    strain_inc = jnp.array([1e-4, 0.0, 0.0, 0.0, 0.0, 0.0])  # tiny, stays elastic
    stress_n = jnp.zeros(6)
    state_n = model.initial_state()

    stress_new, state_new, ddsdde = return_mapping(
        model, strain_inc, stress_n, state_n, steel_params, method="analytical"
    )
    C = model.elastic_stiffness(steel_params)
    assert jnp.allclose(stress_new, C @ strain_inc, rtol=1e-10)
    assert jnp.allclose(ddsdde, C, rtol=1e-10)


def test_plastic_corrector_standalone_plastic(model, steel_params):
    """plastic_corrector returns correct (stress, state, dlambda) for a plastic step."""
    E, nu = steel_params["E"], steel_params["nu"]
    mu = E / (2.0 * (1.0 + nu))
    H = steel_params["H"]
    sigma_y0 = steel_params["sigma_y0"]

    C = model.elastic_stiffness(steel_params)
    deps11 = 2e-3
    strain_inc = jnp.array([deps11, 0.0, 0.0, 0.0, 0.0, 0.0])
    stress_trial = C @ strain_inc
    state_n = model.initial_state()

    result = model.plastic_corrector(stress_trial, C, state_n, steel_params)
    assert result is not None, "plastic_corrector should return a result for plastic step"

    stress_new, state_new, dlambda = result

    # Yield consistency: f(σ_new, state_new) ≈ 0
    f_final = model.yield_function(stress_new, state_new, steel_params)
    assert abs(float(f_final)) < 1e-8, f"|f| = {float(abs(f_final)):.3e}"

    # State update: ep_new = ep_n + dlambda
    ep_n = float(state_n["ep"])
    assert abs(float(state_new["ep"]) - (ep_n + float(dlambda))) < 1e-12


def test_plastic_corrector_dlambda_positive(model, steel_params):
    """Δλ must be positive for a genuinely plastic increment."""
    C = model.elastic_stiffness(steel_params)
    strain_inc = jnp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    stress_trial = C @ strain_inc
    state_n = model.initial_state()

    _, _, dlambda = model.plastic_corrector(stress_trial, C, state_n, steel_params)
    assert float(dlambda) > 0.0


# ---------------------------------------------------------------------------
# method="analytical" vs method="autodiff" — stress and state
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strain_inc_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],          # uniaxial
    [1e-3, -5e-4, -5e-4, 0.0, 0.0, 0.0],       # triaxial isochoric
    [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],           # shear
    [1e-3, 5e-4, 2e-4, 1e-3, 5e-4, 2e-4],      # mixed
])
def test_analytical_stress_matches_autodiff(model, steel_params, strain_inc_vec):
    """method='analytical' stress must match method='autodiff' to tight tolerance."""
    strain_inc = jnp.array(strain_inc_vec)
    stress_n = jnp.zeros(6)
    state_n = model.initial_state()

    s_ad, st_ad, _ = return_mapping(model, strain_inc, stress_n, state_n, steel_params, method="autodiff")
    s_an, st_an, _ = return_mapping(model, strain_inc, stress_n, state_n, steel_params, method="analytical")

    assert jnp.allclose(s_an, s_ad, atol=1e-6), \
        f"max stress diff = {float(jnp.max(jnp.abs(s_an - s_ad))):.3e}"
    assert abs(float(st_an["ep"]) - float(st_ad["ep"])) < 1e-10


def test_analytical_stress_matches_autodiff_nonzero_initial_stress(model, steel_params):
    """Works correctly from a pre-stressed state."""
    # First step to build up pre-stress
    strain_inc1 = jnp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    s1, st1, _ = return_mapping(model, strain_inc1, jnp.zeros(6), model.initial_state(), steel_params)

    # Second plastic step
    strain_inc2 = jnp.array([1e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    s_ad, _, _ = return_mapping(model, strain_inc2, s1, st1, steel_params, method="autodiff")
    s_an, _, _ = return_mapping(model, strain_inc2, s1, st1, steel_params, method="analytical")

    assert jnp.allclose(s_an, s_ad, atol=1e-6)


# ---------------------------------------------------------------------------
# method="analytical" vs method="autodiff" — tangent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strain_inc_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1e-3, -5e-4, -5e-4, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
])
def test_analytical_tangent_matches_autodiff(model, steel_params, strain_inc_vec):
    """Analytical DDSDDE must match autodiff DDSDDE to 1e-5 relative tolerance."""
    strain_inc = jnp.array(strain_inc_vec)
    stress_n = jnp.zeros(6)
    state_n = model.initial_state()

    _, _, D_ad = return_mapping(model, strain_inc, stress_n, state_n, steel_params, method="autodiff")
    _, _, D_an = return_mapping(model, strain_inc, stress_n, state_n, steel_params, method="analytical")

    rel_err = jnp.abs(D_an - D_ad) / (jnp.abs(D_ad) + 1.0)
    assert float(jnp.max(rel_err)) < 1e-5, \
        f"max tangent rel err = {float(jnp.max(rel_err)):.3e}"


# ---------------------------------------------------------------------------
# method="auto" uses the analytical path when available
# ---------------------------------------------------------------------------

def test_method_auto_uses_analytical(model, steel_params):
    """method='auto' should use plastic_corrector when it returns non-None."""
    strain_inc = jnp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    stress_n = jnp.zeros(6)
    state_n = model.initial_state()

    s_auto, _, D_auto = return_mapping(model, strain_inc, stress_n, state_n, steel_params, method="auto")
    s_an,   _, D_an   = return_mapping(model, strain_inc, stress_n, state_n, steel_params, method="analytical")

    assert jnp.allclose(s_auto, s_an, atol=1e-12), "auto should match analytical path"
    assert jnp.allclose(D_auto, D_an, atol=1e-12), "auto should match analytical tangent"


# ---------------------------------------------------------------------------
# Finite-difference verification of analytical tangent
# ---------------------------------------------------------------------------

def test_analytical_tangent_fd_check_elastic(model, steel_params):
    """Elastic step: analytical tangent = C = FD tangent."""
    result = check_tangent(
        model,
        jnp.zeros(6),
        model.initial_state(),
        steel_params,
        jnp.array([1e-4, 0.0, 0.0, 0.0, 0.0, 0.0]),
        method="analytical",
    )
    assert result.passed, f"FD check failed: max_rel_err = {result.max_rel_err:.3e}"


@pytest.mark.parametrize("strain_inc_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1e-3, -5e-4, -5e-4, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
])
def test_analytical_tangent_fd_check_plastic(model, steel_params, strain_inc_vec):
    """Plastic step: analytical tangent passes finite-difference check."""
    result = check_tangent(
        model,
        jnp.zeros(6),
        model.initial_state(),
        steel_params,
        jnp.array(strain_inc_vec),
        method="analytical",
    )
    assert result.passed, f"FD check failed: max_rel_err = {result.max_rel_err:.3e}"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_method_analytical_raises_if_no_hooks(steel_params):
    """method='analytical' raises NotImplementedError for a model without hooks."""

    class MinimalModel(MaterialModel3D):
        param_names = ["E", "nu", "sigma_y0"]
        state_names = []

        def elastic_stiffness(self, params):
            E, nu = params["E"], params["nu"]
            mu = E / (2.0 * (1.0 + nu))
            lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            return self.isotropic_C(lam, mu)

        def yield_function(self, stress, state, params):
            return self._vonmises(stress) - params["sigma_y0"]

        def hardening_increment(self, dlambda, stress, state, params):
            return {}

    minimal_model = MinimalModel()
    strain_inc = jnp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])

    with pytest.raises(NotImplementedError):
        return_mapping(minimal_model, strain_inc, jnp.zeros(6), {}, steel_params, method="analytical")


def test_method_invalid_raises_value_error(model, steel_params):
    """Unrecognised method string raises ValueError."""
    with pytest.raises(ValueError, match="method must be"):
        return_mapping(model, jnp.zeros(6), jnp.zeros(6), model.initial_state(),
                       steel_params, method="wrong")
