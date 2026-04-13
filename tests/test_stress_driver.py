"""Tests for StressDriver (stress-controlled simulation driver)."""

import jax.numpy as jnp
import numpy as np
import pytest

import manforge  # noqa: F401 — enables float64
from manforge.core.stress_state import SOLID_3D, UNIAXIAL_1D
from manforge.models.j2_isotropic import J2Isotropic3D, J2Isotropic1D
from manforge.simulation.driver import StressDriver, UniaxialDriver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model_3d():
    return J2Isotropic3D(SOLID_3D)


@pytest.fixture
def model_1d():
    return J2Isotropic1D(UNIAXIAL_1D)


# ---------------------------------------------------------------------------
# Elastic-only stress control
# ---------------------------------------------------------------------------

def test_elastic_strain_matches_compliance(model_3d, steel_params):
    """In the elastic regime, StressDriver must recover S @ σ exactly."""
    # Prescribe stress well below yield (sigma_y0 = 250 MPa)
    sigma_y0 = steel_params["sigma_y0"]
    N = 10
    # Ramp σ11 from 0 to 0.5 * sigma_y0, all other components zero
    stress_history = np.zeros((N, 6))
    stress_history[:, 0] = np.linspace(0.0, 0.5 * sigma_y0, N)

    result = StressDriver().run(model_3d, stress_history, steel_params)
    stress_out = result["stress"]
    strain_out = result["strain"]

    # Compliance matrix
    C = np.array(model_3d.elastic_stiffness(steel_params))
    S = np.linalg.inv(C)

    for i in range(N):
        expected_strain = S @ stress_history[i]
        np.testing.assert_allclose(
            strain_out[i], expected_strain, atol=1e-10,
            err_msg=f"strain mismatch at step {i}"
        )
    # Stress output must match target
    np.testing.assert_allclose(stress_out, stress_history, atol=1e-8)


# ---------------------------------------------------------------------------
# Uniaxial stress in 3D model
# ---------------------------------------------------------------------------

def test_uniaxial_stress_3d_lateral_strains_negative(model_3d, steel_params):
    """Under σ11 loading with other components zero, lateral strains must be negative."""
    sigma_y0 = steel_params["sigma_y0"]
    N = 20
    # Ramp σ11 past yield
    stress_history = np.zeros((N, 6))
    stress_history[:, 0] = np.linspace(0.0, 1.5 * sigma_y0, N)

    result = StressDriver().run(model_3d, stress_history, steel_params)
    strain_out = result["strain"]

    # After a few elastic steps, lateral strains (22, 33) must be negative
    assert np.all(strain_out[1:, 1] <= 0.0), "ε22 should be non-positive"
    assert np.all(strain_out[1:, 2] <= 0.0), "ε33 should be non-positive"


def test_uniaxial_stress_3d_off_diagonal_stress_near_zero(model_3d, steel_params):
    """Off-axis stress components must remain near zero under uniaxial stress target."""
    sigma_y0 = steel_params["sigma_y0"]
    N = 20
    stress_history = np.zeros((N, 6))
    stress_history[:, 0] = np.linspace(0.0, 1.5 * sigma_y0, N)

    result = StressDriver().run(model_3d, stress_history, steel_params)
    stress_out = result["stress"]

    # Components 1-5 (σ22, σ33, σ12, σ13, σ23) must be ~0
    np.testing.assert_allclose(
        stress_out[:, 1:], 0.0, atol=1e-6,
        err_msg="Off-axis stresses should be ~0 under uniaxial stress control"
    )


def test_uniaxial_stress_3d_sigma11_matches_target(model_3d, steel_params):
    """σ11 output must match the prescribed target at every step."""
    sigma_y0 = steel_params["sigma_y0"]
    N = 20
    stress_history = np.zeros((N, 6))
    stress_history[:, 0] = np.linspace(0.0, 1.5 * sigma_y0, N)

    result = StressDriver().run(model_3d, stress_history, steel_params)
    stress_out = result["stress"]

    np.testing.assert_allclose(
        stress_out[:, 0], stress_history[:, 0], atol=1e-8,
        err_msg="σ11 output must match prescribed target"
    )


def test_output_shapes(model_3d, steel_params):
    """StressDriver must return stress and strain arrays of shape (N, ntens)."""
    N = 15
    stress_history = np.zeros((N, 6))
    stress_history[:, 0] = np.linspace(0, 200.0, N)

    result = StressDriver().run(model_3d, stress_history, steel_params)

    assert result["stress"].shape == (N, 6)
    assert result["strain"].shape == (N, 6)


# ---------------------------------------------------------------------------
# Consistency with UniaxialDriver (1D model)
# ---------------------------------------------------------------------------

def test_consistency_with_uniaxial_driver_1d(model_1d, steel_params):
    """StressDriver must recover the strain used by UniaxialDriver (1D model).

    Procedure:
    1. Run UniaxialDriver with a strain history → get stress history.
    2. Feed that stress history into StressDriver → get recovered strain.
    3. The recovered strain must match the original strain history.
    """
    sigma_y0 = steel_params["sigma_y0"]
    E = steel_params["E"]

    # Build a strain history that goes into the plastic regime
    eps_yield = sigma_y0 / E
    N = 25
    eps_history_1d = np.linspace(0.0, 3 * eps_yield, N)

    # Step 1: strain → stress
    sigma_history_1d = UniaxialDriver().run(model_1d, eps_history_1d, steel_params)

    # Step 2: stress → strain via StressDriver
    sigma_history_2d = sigma_history_1d.reshape(N, 1)
    result = StressDriver().run(model_1d, sigma_history_2d, steel_params)
    recovered_strain = result["strain"][:, 0]

    np.testing.assert_allclose(
        recovered_strain, eps_history_1d, atol=1e-8,
        err_msg="StressDriver did not recover original strain from UniaxialDriver output"
    )


# ---------------------------------------------------------------------------
# Convergence failure
# ---------------------------------------------------------------------------

def test_convergence_failure_raises(model_3d, steel_params):
    """Insufficient iterations for a plastic step must raise RuntimeError.

    With max_iter=1 and a target stress in the plastic regime, the elastic
    compliance initial guess produces stress_trial == sigma_target, which the
    plastic corrector projects back to the yield surface.  The corrected strain
    is never re-evaluated within the single allowed iteration, so the outer NR
    cannot reach the target and must raise RuntimeError.
    """
    sigma_y0 = steel_params["sigma_y0"]
    # Target stress is above yield — plastic regime, 1 outer NR step is insufficient
    stress_history = np.zeros((1, 6))
    stress_history[0, 0] = 1.5 * sigma_y0

    driver = StressDriver(max_iter=1)
    with pytest.raises(RuntimeError, match="NR did not converge"):
        driver.run(model_3d, stress_history, steel_params)
