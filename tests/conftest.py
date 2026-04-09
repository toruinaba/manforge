import pytest


@pytest.fixture
def steel_params():
    """Typical steel material parameters (SI-consistent units: MPa, -)."""
    return {
        "E": 210000.0,
        "nu": 0.3,
        "sigma_y0": 250.0,
        "H": 1000.0,
    }
