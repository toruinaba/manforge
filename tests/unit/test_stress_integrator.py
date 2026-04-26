"""Unit tests: PythonIntegrator matches bare stress_update/initial_state/etc."""

import numpy as np
import pytest

from manforge.core.stress_update import stress_update
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.simulation.integrator import PythonIntegrator


@pytest.fixture
def model():
    return J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)


def test_initial_state_delegates(model):
    pi = PythonIntegrator(model)
    assert pi.initial_state() == model.initial_state()


def test_elastic_stiffness_delegates(model):
    pi = PythonIntegrator(model)
    np.testing.assert_array_equal(pi.elastic_stiffness(), model.elastic_stiffness())


def test_ntens_delegates(model):
    pi = PythonIntegrator(model)
    assert pi.ntens == model.ntens


def test_stress_update_elastic_matches_bare(model):
    pi = PythonIntegrator(model, method="numerical_newton")
    strain_inc = np.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])
    stress_n = np.zeros(model.ntens)
    state_n = model.initial_state()

    result_pi = pi.stress_update(strain_inc, stress_n, state_n)
    result_bare = stress_update(model, strain_inc, stress_n, state_n, method="numerical_newton")

    np.testing.assert_allclose(result_pi.stress, result_bare.stress, rtol=1e-12)
    assert result_pi.is_plastic == result_bare.is_plastic


def test_stress_update_plastic_matches_bare(model):
    pi = PythonIntegrator(model, method="numerical_newton")
    strain_inc = np.array([5e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    stress_n = np.zeros(model.ntens)
    state_n = model.initial_state()

    result_pi = pi.stress_update(strain_inc, stress_n, state_n)
    result_bare = stress_update(model, strain_inc, stress_n, state_n, method="numerical_newton")

    np.testing.assert_allclose(result_pi.stress, result_bare.stress, rtol=1e-12)
    np.testing.assert_allclose(result_pi.ddsdde, result_bare.ddsdde, rtol=1e-12)
    assert result_pi.is_plastic == result_bare.is_plastic
