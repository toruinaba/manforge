"""Smoke tests for simulation/integrator/python.py method selection logic.

TODO: cover _method="auto"/"numerical_newton"/"user_defined" selection,
      PythonIntegrator vs PythonNumericalIntegrator vs PythonAnalyticalIntegrator.
"""

import pytest
from manforge.simulation.integrator import (
    PythonIntegrator,
    PythonNumericalIntegrator,
    PythonAnalyticalIntegrator,
)


def test_integrators_importable():
    assert PythonIntegrator is not None
    assert PythonNumericalIntegrator is not None
    assert PythonAnalyticalIntegrator is not None


def test_numerical_integrator_method():
    assert PythonNumericalIntegrator._method == "numerical_newton"


def test_analytical_integrator_method():
    assert PythonAnalyticalIntegrator._method == "user_defined"
