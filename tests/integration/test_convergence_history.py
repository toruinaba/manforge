"""Tests for NR convergence history in ReturnMappingResult."""

import jax.numpy as jnp
import numpy as np
import pytest

import manforge  # noqa: F401 — enables JAX float64
from manforge.core.stress_update import stress_update
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.models.ow_kinematic import OWKinematic3D
from manforge.simulation.driver import StrainDriver
from manforge.simulation.types import FieldHistory, FieldType


@pytest.fixture
def j2_model():
    return J2Isotropic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)


@pytest.fixture
def ow_model():
    return OWKinematic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, C_k=50_000.0, gamma=500.0)


# ---------------------------------------------------------------------------
# Elastic step: no NR iteration
# ---------------------------------------------------------------------------

def test_elastic_step_no_history(j2_model):
    deps = jnp.array([1e-4, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = stress_update(j2_model, deps, jnp.zeros(6), j2_model.initial_state())
    assert result.is_plastic is False
    assert result.n_iterations == 0
    assert result.residual_history == []


# ---------------------------------------------------------------------------
# Analytical path: closed-form, no NR
# ---------------------------------------------------------------------------

def test_analytical_path_no_history(j2_model):
    deps = jnp.array([3e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = stress_update(j2_model, deps, jnp.zeros(6), j2_model.initial_state(),
                            method="user_defined")
    assert result.is_plastic is True
    assert result.n_iterations == 0
    assert result.residual_history == []


# ---------------------------------------------------------------------------
# Scalar NR (reduced hardening, J2)
# ---------------------------------------------------------------------------

def test_scalar_nr_records_history(j2_model):
    deps = jnp.array([3e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = stress_update(j2_model, deps, jnp.zeros(6), j2_model.initial_state(),
                            method="numerical_newton")
    assert result.is_plastic is True
    assert result.n_iterations >= 1
    assert len(result.residual_history) == result.n_iterations + 1
    assert result.residual_history[-1] < 1e-10


def test_j2_linear_converges_in_one_step(j2_model):
    """J2 with linear isotropic hardening is exact in one Newton step."""
    deps = jnp.array([3e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = stress_update(j2_model, deps, jnp.zeros(6), j2_model.initial_state(),
                            method="numerical_newton")
    assert result.n_iterations == 1


def test_scalar_nr_residual_decreasing(j2_model):
    deps = jnp.array([3e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = stress_update(j2_model, deps, jnp.zeros(6), j2_model.initial_state(),
                            method="numerical_newton")
    history = result.residual_history
    for i in range(1, len(history)):
        assert history[i] < history[i - 1]


# ---------------------------------------------------------------------------
# Augmented NR (augmented hardening, OWKinematic)
# ---------------------------------------------------------------------------

def test_augmented_nr_records_history(ow_model):
    deps = jnp.zeros(6).at[0].set(3e-3)
    result = stress_update(ow_model, deps, jnp.zeros(6), ow_model.initial_state())
    assert result.is_plastic is True
    assert result.n_iterations >= 1
    assert len(result.residual_history) == result.n_iterations + 1
    assert result.residual_history[-1] < 1e-10


def test_augmented_nr_residual_eventually_decreasing(ow_model):
    """Augmented NR converges; the final residual must be smaller than the first."""
    deps = jnp.zeros(6).at[0].set(3e-3)
    result = stress_update(ow_model, deps, jnp.zeros(6), ow_model.initial_state())
    history = result.residual_history
    assert history[-1] < history[0]


def test_augmented_nr_quadratic_convergence(ow_model):
    """OWKinematic produces enough NR iterations to observe quadratic convergence."""
    deps = jnp.zeros(6).at[0].set(3e-3)
    result = stress_update(ow_model, deps, jnp.zeros(6), ow_model.initial_state())
    history = result.residual_history

    # Need at least 3 points for a local convergence order estimate
    if len(history) >= 3:
        # Local convergence order: p = log(e_{k+1}/e_k) / log(e_k/e_{k-1})
        orders = []
        for i in range(1, len(history) - 1):
            e_prev, e_curr, e_next = history[i - 1], history[i], history[i + 1]
            if e_prev > 0 and e_curr > 0 and e_next > 0:
                import math
                den = math.log(e_curr / e_prev)
                if abs(den) > 1e-30:
                    orders.append(math.log(e_next / e_curr) / den)
        if orders:
            assert orders[-1] > 1.5, (
                f"Expected near-quadratic convergence, got order {orders[-1]:.2f}. "
                f"residual_history = {history}"
            )


# ---------------------------------------------------------------------------
# StrainDriver integration: access history via step_results
# ---------------------------------------------------------------------------

def test_driver_j2_analytical_no_history(j2_model):
    """J2 uses analytical plastic_corrector by default: NR history is empty."""
    driver = StrainDriver()
    load = FieldHistory(FieldType.STRAIN, "Strain", np.linspace(0.0, 5e-3, 20))
    dr = driver.run(j2_model, load)

    for rm in dr.step_results:
        assert rm.n_iterations == 0
        assert rm.residual_history == []


def test_driver_j2_autodiff_has_history(j2_model):
    """With method='numerical_newton', J2 plastic steps record NR history."""
    driver = StrainDriver()
    load = FieldHistory(FieldType.STRAIN, "Strain", np.linspace(0.0, 5e-3, 20))
    dr = driver.run(j2_model, load, method="numerical_newton")

    for rm in dr.step_results:
        if rm.is_plastic:
            assert rm.n_iterations >= 1
            assert len(rm.residual_history) == rm.n_iterations + 1
        else:
            assert rm.n_iterations == 0
            assert rm.residual_history == []


def test_driver_ow_step_results_have_history(ow_model):
    driver = StrainDriver()
    load = FieldHistory(FieldType.STRAIN, "Strain", np.linspace(0.0, 5e-3, 20))
    dr = driver.run(ow_model, load)

    for rm in dr.step_results:
        if rm.is_plastic:
            assert rm.n_iterations >= 1
            assert len(rm.residual_history) == rm.n_iterations + 1
        else:
            assert rm.n_iterations == 0
            assert rm.residual_history == []
