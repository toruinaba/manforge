"""Integration tests: Driver produces identical results with bare model vs PythonIntegrator."""

import numpy as np
import pytest

from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.simulation import PythonIntegrator, StrainDriver, StressDriver
from manforge.simulation.types import FieldHistory, FieldType
from manforge.verification import generate_strain_history


@pytest.fixture
def model():
    return J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)


@pytest.fixture
def strain_load(model):
    history = generate_strain_history(model)
    return FieldHistory(FieldType.STRAIN, "eps", history)


def test_strain_driver_bare_vs_integrator(model, strain_load):
    """StrainDriver.run(model, load) == StrainDriver.run(PythonIntegrator(model), load)."""
    driver = StrainDriver()

    result_bare = driver.run(model, strain_load, method="numerical_newton")
    result_pi   = driver.run(PythonIntegrator(model, method="numerical_newton"), strain_load)

    np.testing.assert_allclose(result_bare.stress, result_pi.stress, rtol=1e-12)
    np.testing.assert_allclose(result_bare.strain, result_pi.strain, rtol=1e-12)
    assert len(result_bare.step_results) == len(result_pi.step_results)


def test_strain_driver_iter_run_bare_vs_integrator(model, strain_load):
    """iter_run produces identical steps for bare model and PythonIntegrator."""
    driver = StrainDriver()
    pi = PythonIntegrator(model, method="numerical_newton")

    for step_bare, step_pi in zip(
        driver.iter_run(model, strain_load, method="numerical_newton"),
        driver.iter_run(pi, strain_load),
    ):
        np.testing.assert_allclose(step_bare.result.stress, step_pi.result.stress, rtol=1e-12)
        assert step_bare.result.is_plastic == step_pi.result.is_plastic


def test_stress_driver_bare_vs_integrator(model):
    """StressDriver.run gives identical results for bare model and PythonIntegrator."""
    sigma_max = 1.5 * model.sigma_y0
    targets = np.array([0.3 * sigma_max, 0.8 * sigma_max, sigma_max, 0.5 * sigma_max])
    stress_data = np.zeros((len(targets), model.ntens))
    stress_data[:, 0] = targets
    load = FieldHistory(FieldType.STRESS, "sigma", stress_data)

    driver = StressDriver()
    result_bare = driver.run(model, load, method="numerical_newton")
    result_pi   = driver.run(PythonIntegrator(model, method="numerical_newton"), load)

    np.testing.assert_allclose(result_bare.stress, result_pi.stress, rtol=1e-12)
    np.testing.assert_allclose(result_bare.strain, result_pi.strain, rtol=1e-12)
