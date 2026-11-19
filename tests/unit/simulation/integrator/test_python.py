"""Smoke tests for simulation/integrator/python.py method selection logic.

TODO: cover _method="auto"/"numerical_newton"/"user_defined" selection,
      PythonIntegrator vs PythonNumericalIntegrator vs PythonAnalyticalIntegrator.
"""

import numpy as np
import pytest
from returnmap.simulation import (
    PythonIntegrator, PythonNumericalIntegrator, PythonAnalyticalIntegrator,
)
from returnmap.models import J2Isotropic3D


def test_integrators_importable():
    assert PythonIntegrator is not None
    assert PythonNumericalIntegrator is not None
    assert PythonAnalyticalIntegrator is not None


def test_numerical_integrator_method():
    assert PythonNumericalIntegrator._method == "numerical_newton"


def test_analytical_integrator_method():
    assert PythonAnalyticalIntegrator._method == "user_defined"


# ---------------------------------------------------------------------------
# The Increment handed to the hand-written hooks
# ---------------------------------------------------------------------------

class _CapturingJ2(J2Isotropic3D):
    """J2Isotropic3D subclass that records what its hand-written hooks receive."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.captured: dict = {}

    def user_defined_return_mapping(self, inc):
        self.captured["return_mapping"] = inc
        return super().user_defined_return_mapping(inc)

    def user_defined_tangent(self, new, dlambda, inc):
        self.captured["tangent"] = (new, dlambda, inc)
        return super().user_defined_tangent(new, dlambda, inc)


def test_hand_written_hooks_receive_the_increment():
    """Both hooks get the same Increment: σ_trial, Δε and the step-n state."""
    model = _CapturingJ2(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
    integrator = PythonAnalyticalIntegrator(model)

    stress_n = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    state_n = model.initial_state()
    # Large enough strain increment to trigger plasticity
    strain_inc = np.array([0.01, -0.005, -0.005, 0.0, 0.0, 0.0])

    result = integrator.stress_update(strain_inc, stress_n, state_n)

    inc = model.captured["return_mapping"]
    new, dlambda, inc_tangent = model.captured["tangent"]
    assert inc_tangent is inc, "the tangent hook must see the very same increment"
    assert result.increment is inc, "and the result must carry it"

    np.testing.assert_allclose(inc.strain_inc, strain_inc)
    expected_stress_trial = stress_n + model.elastic_stiffness(state_n) @ strain_inc
    np.testing.assert_allclose(inc.trial["stress"], expected_stress_trial)
    # σ_n is the stress passed to stress_update, not whatever state_n carried.
    np.testing.assert_allclose(inc.state_n["stress"], stress_n)
    np.testing.assert_allclose(inc.trial["ep"], state_n["ep"])

    # The tangent hook gets the converged state, stress included.
    np.testing.assert_allclose(new["stress"], np.asarray(result.stress))
    assert float(dlambda) == float(result.dlambda)
