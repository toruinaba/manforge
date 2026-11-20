import pytest
import autograd.numpy as anp

import returnmap  # noqa: F401

from returnmap.models import (
    J2Isotropic3D, J2IsotropicPS, J2Isotropic1D, AFKinematic3D, AFKinematicPS,
    AFKinematic1D, OWKinematic3D, OWKinematicPS, OWKinematic1D,
)
from tests.fixtures.strain_vectors import (
    DEPS_UNIAXIAL_3D, DEPS_EQUIBIAXIAL_3D, DEPS_PURE_SHEAR_3D, DEPS_COMBINED_3D,
)

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
