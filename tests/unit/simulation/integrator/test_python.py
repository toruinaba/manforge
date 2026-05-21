"""Smoke tests for simulation/integrator/python.py method selection logic.

TODO: cover _method="auto"/"numerical_newton"/"user_defined" selection,
      PythonIntegrator vs PythonNumericalIntegrator vs PythonAnalyticalIntegrator.
"""

import numpy as np
import pytest
from manforge.simulation.integrator import (
    PythonIntegrator,
    PythonNumericalIntegrator,
    PythonAnalyticalIntegrator,
)
from manforge.models import J2Isotropic3D


def test_integrators_importable():
    assert PythonIntegrator is not None
    assert PythonNumericalIntegrator is not None
    assert PythonAnalyticalIntegrator is not None


def test_numerical_integrator_method():
    assert PythonNumericalIntegrator._method == "numerical_newton"


def test_analytical_integrator_method():
    assert PythonAnalyticalIntegrator._method == "user_defined"


# ---------------------------------------------------------------------------
# user_defined_tangent argument forwarding
# ---------------------------------------------------------------------------

class _CapturingJ2(J2Isotropic3D):
    """J2Isotropic3D subclass that records kwargs passed to user_defined_tangent."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.captured: dict = {}

    def user_defined_tangent(self, stress, state, dlambda, C, state_n, *, stress_trial=None, strain_inc=None):
        self.captured = {"stress_trial": stress_trial, "strain_inc": strain_inc}
        return super().user_defined_tangent(
            stress, state, dlambda, C, state_n,
            stress_trial=stress_trial, strain_inc=strain_inc,
        )


def test_user_defined_tangent_receives_stress_trial_and_strain_inc():
    """stress_trial and strain_inc must reach user_defined_tangent when called via stress_update."""
    model = _CapturingJ2(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
    integrator = PythonAnalyticalIntegrator(model)

    stress_n = np.zeros(6)
    state_n = model.initial_state()
    # Large enough strain increment to trigger plasticity
    strain_inc = np.array([0.01, -0.005, -0.005, 0.0, 0.0, 0.0])

    integrator.stress_update(strain_inc, stress_n, state_n)

    assert model.captured.get("stress_trial") is not None, (
        "stress_trial was not forwarded to user_defined_tangent"
    )
    assert model.captured.get("strain_inc") is not None, (
        "strain_inc was not forwarded to user_defined_tangent"
    )
    np.testing.assert_allclose(
        model.captured["strain_inc"], strain_inc,
        err_msg="strain_inc value forwarded incorrectly",
    )
    C_n = model.elastic_stiffness(state_n)
    expected_stress_trial = stress_n + C_n @ strain_inc
    np.testing.assert_allclose(
        model.captured["stress_trial"], expected_stress_trial,
        err_msg="stress_trial value forwarded incorrectly",
    )
