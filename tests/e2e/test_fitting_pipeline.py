"""Smoke tests for the parameter fitting pipeline.

Strategy
--------
1. Generate synthetic stress-strain data from J2Isotropic3D with
   *known* parameters (``sigma_y0=250, H=1000``).
2. Start optimisation from a perturbed initial point.
3. Verify the recovered parameters are close to the true values (rtol=0.1).
4. Verify residual is near zero and FitResult structure is complete.

Convergence tests are ``slow``; the wiring tests cap ``max_iter`` so they
run in the fast suite.
"""

import numpy as np
import pytest

from manforge.simulation.driver import StrainDriver, StressDriver
from manforge.simulation.integrator import PythonIntegrator
from manforge.simulation.types import FieldHistory, FieldType
from manforge.fitting import FitTarget, Observation, ParameterSpace
from manforge.fitting.optimizer import fit_params, FitResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def driver_factory():
    return lambda integrator: StrainDriver(integrator)


@pytest.fixture
def synthetic_data(model, driver_factory):
    """Uniaxial strain history and synthetic stress response (σ11 only)."""
    strain = np.linspace(0.0, 5e-3, 40)
    load = FieldHistory(FieldType.STRAIN, "Strain", strain)
    result = driver_factory(PythonIntegrator(model)).run(load)
    return {"strain": strain, "stress": result.stress[:, 0]}


# ---------------------------------------------------------------------------
# Wiring tests — capped iterations, fast suite
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method", ["L-BFGS-B", "Nelder-Mead", "differential_evolution"]
)
def test_max_iter_bounds_work(model, driver_factory, synthetic_data, method):
    """max_iter caps the optimiser for every method."""
    fit_config = {
        "sigma_y0": (220.0, (50.0, 600.0)),
        "H":        (800.0, (0.0, 5000.0)),
    }
    fixed = {"E": model.E, "nu": model.nu}

    result = fit_params(model, driver_factory, synthetic_data, fit_config,
                        fixed_params=fixed, method=method, max_iter=1)

    assert result.n_iter > 0
    assert set(result.params) == {"E", "nu", "sigma_y0", "H"}


def test_options_override_max_iter(model, driver_factory, synthetic_data):
    """options= wins over the max_iter keyword."""
    fit_config = {"sigma_y0": (220.0, (50.0, 600.0))}
    fixed = {"E": model.E, "nu": model.nu, "H": model.H}

    capped = fit_params(model, driver_factory, synthetic_data, fit_config,
                        fixed_params=fixed, max_iter=50, options={"maxiter": 1})

    assert len(capped.history) <= 1


def test_derived_param_reaches_model(model, driver_factory, synthetic_data):
    """A derived parameter is reconstructed and passed to the model.

    ``H`` is optimised as a multiple of ``sigma_y0`` — the helper variable
    ``H_ratio`` is not a model parameter and must not reach the constructor.
    """
    space = ParameterSpace(
        free={"sigma_y0": (220.0, (50.0, 600.0)), "H_ratio": (2.0, (0.1, 20.0))},
        fixed={"E": model.E, "nu": model.nu},
        derived={"H": lambda p: p["sigma_y0"] * p["H_ratio"]},
    )

    result = fit_params(model, driver_factory, synthetic_data, space,
                        max_iter=1)

    assert result.params["H"] == pytest.approx(
        result.params["sigma_y0"] * result.params["H_ratio"]
    )
    assert "H_ratio" in result.history[0]


def test_derived_ratio_enforces_ordering(model, driver_factory, synthetic_data):
    """A ratio bounded below by 1 keeps H >= sigma_y0 at every iterate."""
    space = ParameterSpace(
        free={"sigma_y0": (220.0, (50.0, 600.0)), "H_ratio": (2.0, (1.0, 20.0))},
        fixed={"E": model.E, "nu": model.nu},
        derived={"H": lambda p: p["sigma_y0"] * p["H_ratio"]},
    )

    result = fit_params(model, driver_factory, synthetic_data, space,
                        max_iter=3)

    for iterate in [*result.history, result.params]:
        values = space.resolve([iterate["sigma_y0"], iterate["H_ratio"]])
        assert values["H"] >= values["sigma_y0"]


# ---------------------------------------------------------------------------
# Multiple targets
# ---------------------------------------------------------------------------

def _strain_target(model, peak, n=20, **kwargs):
    """Strain-driven target whose curve is this model's own response."""
    load = FieldHistory(FieldType.STRAIN, "Strain", np.linspace(0.0, peak, n))
    computed = StrainDriver(PythonIntegrator(model)).run(load).stress[:, 0]
    return FitTarget(
        load=load,
        observations=[Observation(values=computed, component=0)],
        **kwargs,
    )


def test_multiple_strain_targets_recover_params(model, driver_factory):
    """Two loading histories fitted at once still recover the true params."""
    targets = [
        _strain_target(model, 3e-3, name="small"),
        _strain_target(model, 8e-3, name="large"),
    ]
    fit_config = {
        "sigma_y0": (220.0, (50.0, 600.0)),
        "H":        (800.0, (0.0, 5000.0)),
    }
    fixed = {"E": model.E, "nu": model.nu}

    result = fit_params(model, driver_factory, targets, fit_config,
                        fixed_params=fixed, max_iter=60)

    assert result.residual < 1e-3
    assert result.params["sigma_y0"] == pytest.approx(model.sigma_y0, rel=0.05)
    assert result.params["H"] == pytest.approx(model.H, rel=0.05)


def test_stress_driven_target(model):
    """A target may be driven by stress, with its own driver_factory."""
    stress_hist = np.zeros((15, 6))
    stress_hist[:, 0] = np.linspace(0.0, 300.0, 15)
    load = FieldHistory(FieldType.STRESS, "Stress", stress_hist)
    strain_computed = StressDriver(PythonIntegrator(model)).run(load).strain[:, 0]

    target = FitTarget(
        load=load,
        observations=[Observation(values=strain_computed, field="Strain", component=0)],
        driver_factory=lambda i: StressDriver(i),
        name="ratchetting",
    )
    fit_config = {"sigma_y0": (240.0, (50.0, 600.0))}
    fixed = {"E": model.E, "nu": model.nu, "H": model.H}

    result = fit_params(model, None, target, fit_config,
                        fixed_params=fixed, max_iter=40)

    assert result.params["sigma_y0"] == pytest.approx(model.sigma_y0, rel=0.05)


def test_mixed_strain_and_stress_driven_targets(model, driver_factory):
    """Strain- and stress-driven targets coexist in one objective."""
    stress_hist = np.zeros((15, 6))
    stress_hist[:, 0] = np.linspace(0.0, 300.0, 15)
    stress_load = FieldHistory(FieldType.STRESS, "Stress", stress_hist)
    strain_computed = StressDriver(PythonIntegrator(model)).run(stress_load).strain[:, 0]

    targets = [
        _strain_target(model, 5e-3, name="tension"),
        FitTarget(
            load=stress_load,
            observations=[
                Observation(values=strain_computed, field="Strain", component=0)
            ],
            driver_factory=lambda i: StressDriver(i),
            name="stress-controlled",
        ),
    ]
    fit_config = {"sigma_y0": (230.0, (50.0, 600.0))}
    fixed = {"E": model.E, "nu": model.nu, "H": model.H}

    result = fit_params(model, driver_factory, targets, fit_config,
                        fixed_params=fixed, max_iter=40)

    assert result.params["sigma_y0"] == pytest.approx(model.sigma_y0, rel=0.05)


def test_stress_and_state_observations_share_one_run(model, driver_factory):
    """σ11 and ep from the same run are both fitted."""
    load = FieldHistory(FieldType.STRAIN, "Strain", np.linspace(0.0, 8e-3, 25))
    run = StrainDriver(PythonIntegrator(model)).run(load)
    ep_computed = np.array([r.state["ep"] for r in run.step_results])

    target = FitTarget(
        load=load,
        observations=[
            Observation(values=run.stress[:, 0], component=0, name="sigma11"),
            # ep is O(1e-3) against stress O(1e2); weight to make it matter
            Observation(values=ep_computed, field="ep", weight=1e8, name="ep"),
        ],
    )
    fit_config = {
        "sigma_y0": (220.0, (50.0, 600.0)),
        "H":        (800.0, (0.0, 5000.0)),
    }
    fixed = {"E": model.E, "nu": model.nu}

    result = fit_params(model, driver_factory, target, fit_config,
                        fixed_params=fixed, max_iter=60)

    assert result.params["sigma_y0"] == pytest.approx(model.sigma_y0, rel=0.05)
    assert result.params["H"] == pytest.approx(model.H, rel=0.05)


# ---------------------------------------------------------------------------
# Structure test
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_fit_result_structure(model, driver_factory, synthetic_data):
    """FitResult has all required fields with correct types."""
    fit_config = {
        "sigma_y0": (220.0, (50.0, 600.0)),
        "H":        (800.0, (0.0, 5000.0)),
    }
    fixed = {"E": model.E, "nu": model.nu}

    result = fit_params(model, driver_factory, synthetic_data, fit_config,
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

@pytest.mark.slow
def test_fit_uniaxial_synthetic(model, driver_factory, synthetic_data):
    """L-BFGS-B recovers sigma_y0 and H within 10% of true values."""
    fit_config = {
        "sigma_y0": (220.0, (50.0, 600.0)),
        "H":        (800.0, (0.0, 5000.0)),
    }
    fixed = {"E": model.E, "nu": model.nu}

    result = fit_params(model, driver_factory, synthetic_data, fit_config,
                        fixed_params=fixed, method="L-BFGS-B")

    assert result.residual < 1.0, f"Residual too large: {result.residual:.3e}"
    assert abs(result.params["sigma_y0"] - model.sigma_y0) / model.sigma_y0 < 0.1
    assert abs(result.params["H"] - model.H) / model.H < 0.1


# ---------------------------------------------------------------------------
# Convergence test — Nelder-Mead
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_fit_nelder_mead(model, driver_factory, synthetic_data):
    """Nelder-Mead also converges to true parameters."""
    fit_config = {
        "sigma_y0": (220.0, (None, None)),
        "H":        (800.0, (None, None)),
    }
    fixed = {"E": model.E, "nu": model.nu}

    result = fit_params(model, driver_factory, synthetic_data, fit_config,
                        fixed_params=fixed, method="Nelder-Mead")

    assert result.residual < 1.0
    assert abs(result.params["sigma_y0"] - model.sigma_y0) / model.sigma_y0 < 0.1


# ---------------------------------------------------------------------------
# History populated for gradient-based methods
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_fit_history_populated(model, driver_factory, synthetic_data):
    """History list is non-empty after L-BFGS-B run."""
    fit_config = {
        "sigma_y0": (220.0, (50.0, 600.0)),
        "H":        (800.0, (0.0, 5000.0)),
    }
    fixed = {"E": model.E, "nu": model.nu}

    result = fit_params(model, driver_factory, synthetic_data, fit_config,
                        fixed_params=fixed, method="L-BFGS-B")

    assert len(result.history) > 0
    assert "sigma_y0" in result.history[0]
    assert "H" in result.history[0]
