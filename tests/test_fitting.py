"""Smoke tests for the parameter fitting pipeline.

Strategy
--------
1. Generate synthetic stress-strain data from J2Isotropic3D with
   *known* parameters (``sigma_y0=250, H=1000``).
2. Start optimisation from a perturbed initial point.
3. Verify the recovered parameters are close to the true values (rtol=0.1).
4. Verify residual is near zero and FitResult structure is complete.
"""

import numpy as np
import pytest

from manforge.simulation.driver import StrainDriver
from manforge.simulation.types import FieldHistory, FieldType
from manforge.fitting.optimizer import fit_params, FitResult

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def driver():
    return StrainDriver()


@pytest.fixture
def synthetic_data(model, driver, steel_params):
    """Uniaxial strain history and synthetic stress response (σ11 only)."""
    strain = np.linspace(0.0, 5e-3, 40)
    load = FieldHistory(FieldType.STRAIN, "Strain", strain)
    result = driver.run(model, load, steel_params)
    return {"strain": strain, "stress": result.stress[:, 0]}


# ---------------------------------------------------------------------------
# Structure test
# ---------------------------------------------------------------------------

def test_fit_result_structure(model, driver, synthetic_data, steel_params):
    """FitResult has all required fields with correct types."""
    fit_config = {
        "sigma_y0": (220.0, (50.0, 600.0)),
        "H":        (800.0, (0.0, 5000.0)),
    }
    fixed = {"E": steel_params["E"], "nu": steel_params["nu"]}

    result = fit_params(model, driver, synthetic_data, fit_config,
                        fixed_params=fixed, method="L-BFGS-B")

    assert isinstance(result, FitResult)
    assert isinstance(result.params, dict)
    assert isinstance(result.residual, float)
    assert isinstance(result.success, bool)
    assert isinstance(result.n_iter, int)
    assert isinstance(result.message, str)
    assert isinstance(result.history, list)
    # All free + fixed params present
    for key in ["E", "nu", "sigma_y0", "H"]:
        assert key in result.params


# ---------------------------------------------------------------------------
# Convergence test — L-BFGS-B
# ---------------------------------------------------------------------------

def test_fit_uniaxial_synthetic(model, driver, synthetic_data, steel_params):
    """L-BFGS-B recovers sigma_y0 and H within 10% of true values."""
    fit_config = {
        "sigma_y0": (220.0, (50.0, 600.0)),
        "H":        (800.0, (0.0, 5000.0)),
    }
    fixed = {"E": steel_params["E"], "nu": steel_params["nu"]}

    result = fit_params(model, driver, synthetic_data, fit_config,
                        fixed_params=fixed, method="L-BFGS-B")

    assert result.residual < 1.0, f"Residual too large: {result.residual:.3e}"
    assert abs(result.params["sigma_y0"] - steel_params["sigma_y0"]) / steel_params["sigma_y0"] < 0.1
    assert abs(result.params["H"] - steel_params["H"]) / steel_params["H"] < 0.1


# ---------------------------------------------------------------------------
# Convergence test — Nelder-Mead
# ---------------------------------------------------------------------------

def test_fit_nelder_mead(model, driver, synthetic_data, steel_params):
    """Nelder-Mead also converges to true parameters."""
    fit_config = {
        "sigma_y0": (220.0, (None, None)),
        "H":        (800.0, (None, None)),
    }
    fixed = {"E": steel_params["E"], "nu": steel_params["nu"]}

    result = fit_params(model, driver, synthetic_data, fit_config,
                        fixed_params=fixed, method="Nelder-Mead")

    assert result.residual < 1.0
    assert abs(result.params["sigma_y0"] - steel_params["sigma_y0"]) / steel_params["sigma_y0"] < 0.1


# ---------------------------------------------------------------------------
# History populated for gradient-based methods
# ---------------------------------------------------------------------------

def test_fit_history_populated(model, driver, synthetic_data, steel_params):
    """History list is non-empty after L-BFGS-B run."""
    fit_config = {
        "sigma_y0": (220.0, (50.0, 600.0)),
        "H":        (800.0, (0.0, 5000.0)),
    }
    fixed = {"E": steel_params["E"], "nu": steel_params["nu"]}

    result = fit_params(model, driver, synthetic_data, fit_config,
                        fixed_params=fixed, method="L-BFGS-B")

    assert len(result.history) > 0
    assert "sigma_y0" in result.history[0]
    assert "H" in result.history[0]


# ---------------------------------------------------------------------------
# GeneralDriver smoke test
# ---------------------------------------------------------------------------

def test_general_driver_runs(model, steel_params):
    """StrainDriver (general) produces (N, 6) stress output without errors."""
    N = 20
    strain6 = np.zeros((N, 6))
    strain6[:, 0] = np.linspace(0.0, 5e-3, N)  # uniaxial ε11
    load = FieldHistory(FieldType.STRAIN, "Strain", strain6)

    result = StrainDriver().run(model, load, steel_params)

    assert result.stress.shape == (N, 6)
    # σ11 should be non-trivial (plastic hardening)
    assert result.stress[-1, 0] > steel_params["sigma_y0"]
