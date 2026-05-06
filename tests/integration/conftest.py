import pytest

from manforge.models.j2_isotropic import J2Isotropic3D, J2IsotropicPS, J2Isotropic1D
from manforge.models.af_kinematic import AFKinematic3D, AFKinematicPS, AFKinematic1D
from manforge.models.ow_kinematic import OWKinematic3D, OWKinematicPS, OWKinematic1D
from manforge.core.dimension import PLANE_STRAIN


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
