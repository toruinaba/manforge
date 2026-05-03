"""Integration tests: Driver works with all Python Integrator types."""

import numpy as np
import pytest

from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.simulation import (
    PythonIntegrator,
    PythonNumericalIntegrator,
    PythonAnalyticalIntegrator,
    StrainDriver,
    StressDriver,
)
from manforge.simulation.types import FieldHistory, FieldType
from manforge.verification import generate_strain_history


@pytest.fixture
def model():
    return J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)


@pytest.fixture
def strain_load(model):
    history = generate_strain_history(model)
    return FieldHistory(FieldType.STRAIN, "eps", history)


def test_strain_driver_numerical_vs_analytical(model, strain_load):
    """PythonNumericalIntegrator and PythonAnalyticalIntegrator give consistent results."""
    result_num = StrainDriver(PythonNumericalIntegrator(model)).run(strain_load)
    result_ana = StrainDriver(PythonAnalyticalIntegrator(model)).run(strain_load)

    np.testing.assert_allclose(result_num.stress, result_ana.stress, rtol=1e-6)
    np.testing.assert_allclose(result_num.strain, result_ana.strain, rtol=1e-12)
    assert len(result_num.step_results) == len(result_ana.step_results)


def test_strain_driver_iter_run_numerical_vs_analytical(model, strain_load):
    """iter_run produces consistent steps for numerical and analytical integrators."""
    for step_num, step_ana in zip(
        StrainDriver(PythonNumericalIntegrator(model)).iter_run(strain_load),
        StrainDriver(PythonAnalyticalIntegrator(model)).iter_run(strain_load),
    ):
        np.testing.assert_allclose(step_num.result.stress, step_ana.result.stress, rtol=1e-6)
        assert step_num.result.is_plastic == step_ana.result.is_plastic


def test_stress_driver_numerical_vs_analytical(model):
    """StressDriver gives consistent results for numerical and analytical integrators."""
    sigma_max = 1.5 * model.sigma_y0
    targets = np.array([0.3 * sigma_max, 0.8 * sigma_max, sigma_max, 0.5 * sigma_max])
    stress_data = np.zeros((len(targets), model.ntens))
    stress_data[:, 0] = targets
    load = FieldHistory(FieldType.STRESS, "sigma", stress_data)

    result_num = StressDriver(PythonNumericalIntegrator(model)).run(load)
    result_ana = StressDriver(PythonAnalyticalIntegrator(model)).run(load)

    np.testing.assert_allclose(result_num.stress, result_ana.stress, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result_num.strain, result_ana.strain, rtol=1e-6, atol=1e-12)


def test_driver_raises_on_bare_model(model, strain_load):
    """Driver raises TypeError when given a bare MaterialModel."""
    with pytest.raises(TypeError, match="StressIntegrator"):
        StrainDriver(model)


def test_python_integrator_auto_method(model, strain_load):
    """PythonIntegrator (auto) produces same results as numerical newton for J2 model."""
    result_auto = StrainDriver(PythonIntegrator(model)).run(strain_load)
    result_num  = StrainDriver(PythonNumericalIntegrator(model)).run(strain_load)

    np.testing.assert_allclose(result_auto.stress, result_num.stress, rtol=1e-6)
