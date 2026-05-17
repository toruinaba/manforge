import sys
import os
import pytest
import autograd.numpy as anp

import manforge  # noqa: F401

from manforge.models.j2_isotropic import J2Isotropic3D, J2IsotropicPS, J2Isotropic1D
from manforge.models.af_kinematic import AFKinematic3D, AFKinematicPS, AFKinematic1D
from manforge.models.ow_kinematic import OWKinematic3D, OWKinematicPS, OWKinematic1D
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
    return anp.array(DEPS_UNIAXIAL_3D)


@pytest.fixture
def deps_equibiaxial():
    return anp.array(DEPS_EQUIBIAXIAL_3D)


@pytest.fixture
def deps_pure_shear():
    return anp.array(DEPS_PURE_SHEAR_3D)


@pytest.fixture
def deps_combined():
    return anp.array(DEPS_COMBINED_3D)


@pytest.fixture
def deps_catalog():
    return {
        "uniaxial":    anp.array(DEPS_UNIAXIAL_3D),
        "equibiaxial": anp.array(DEPS_EQUIBIAXIAL_3D),
        "pure_shear":  anp.array(DEPS_PURE_SHEAR_3D),
        "combined":    anp.array(DEPS_COMBINED_3D),
    }


# ---------------------------------------------------------------------------
# Model fixtures for all dimensions (J2 / AF / OW × 3D / PS / 1D)
# ---------------------------------------------------------------------------

@pytest.fixture
def j2_model_3d():
    return J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)


@pytest.fixture
def j2_model_ps():
    return J2IsotropicPS(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)


@pytest.fixture
def j2_model_1d():
    return J2Isotropic1D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)


@pytest.fixture
def af_model_3d():
    return AFKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0)


@pytest.fixture
def af_model_ps():
    return AFKinematicPS(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0)


@pytest.fixture
def af_model_1d():
    return AFKinematic1D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0)


@pytest.fixture
def ow_model_3d():
    return OWKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=1.0)


@pytest.fixture
def ow_model_ps():
    return OWKinematicPS(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=1.0)


@pytest.fixture
def ow_model_1d():
    return OWKinematic1D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=1.0)
