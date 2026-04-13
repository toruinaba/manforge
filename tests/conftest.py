import sys
import os
import pytest

import manforge  # noqa: F401 — enables JAX float64 for the entire test session

from manforge.models.j2_isotropic import J2Isotropic3D

# Add fortran/ to sys.path so compiled modules are importable session-wide.
# Harmless if the directory contains no compiled modules.
_FORTRAN_DIR = os.path.join(os.path.dirname(__file__), "..", "fortran")
sys.path.insert(0, os.path.abspath(_FORTRAN_DIR))


@pytest.fixture
def steel_params():
    """Typical steel material parameters (SI-consistent units: MPa, -)."""
    return {
        "E": 210000.0,
        "nu": 0.3,
        "sigma_y0": 250.0,
        "H": 1000.0,
    }


@pytest.fixture
def model():
    """Default J2Isotropic3D model instance (SOLID_3D)."""
    return J2Isotropic3D()


@pytest.fixture
def initial_state(model):
    """Initial (virgin) state for the default model."""
    return model.initial_state()


@pytest.fixture
def lame_constants(steel_params):
    """Lame constants (lambda, mu) derived from steel_params."""
    E, nu = steel_params["E"], steel_params["nu"]
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam, mu
