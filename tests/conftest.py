import sys
import os
import pytest
import jax.numpy as jnp

import manforge  # noqa: F401 — enables JAX float64 for the entire test session

from manforge.models.j2_isotropic import J2Isotropic3D
from tests.fixtures.strain_vectors import (
    DEPS_UNIAXIAL_3D, DEPS_EQUIBIAXIAL_3D, DEPS_PURE_SHEAR_3D, DEPS_COMBINED_3D,
)

# Add fortran/ to sys.path so compiled modules are importable session-wide.
# Harmless if the directory contains no compiled modules.
_FORTRAN_DIR = os.path.join(os.path.dirname(__file__), "..", "fortran")
sys.path.insert(0, os.path.abspath(_FORTRAN_DIR))


@pytest.fixture
def model():
    """Default J2Isotropic3D model instance with typical steel parameters."""
    return J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)


@pytest.fixture
def initial_state(model):
    """Initial (virgin) state for the default model."""
    return model.initial_state()


@pytest.fixture
def lame_constants(model):
    """Lame constants (lambda, mu) derived from model parameters."""
    E, nu = model.E, model.nu
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam, mu


# ---------------------------------------------------------------------------
# Canonical 3D strain increment fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def deps_uniaxial():
    return jnp.array(DEPS_UNIAXIAL_3D)


@pytest.fixture
def deps_equibiaxial():
    return jnp.array(DEPS_EQUIBIAXIAL_3D)


@pytest.fixture
def deps_pure_shear():
    return jnp.array(DEPS_PURE_SHEAR_3D)


@pytest.fixture
def deps_combined():
    return jnp.array(DEPS_COMBINED_3D)


@pytest.fixture
def deps_catalog():
    return {
        "uniaxial":    jnp.array(DEPS_UNIAXIAL_3D),
        "equibiaxial": jnp.array(DEPS_EQUIBIAXIAL_3D),
        "pure_shear":  jnp.array(DEPS_PURE_SHEAR_3D),
        "combined":    jnp.array(DEPS_COMBINED_3D),
    }
