"""Smoke tests for verification/test_cases.py.

TODO: cover generate_strain_history shapes, scenario variety, seed reproducibility.
"""

import pytest
from manforge.verification import generate_strain_history
from manforge.models.j2_isotropic import J2Isotropic3D


def test_generate_strain_history_importable():
    assert callable(generate_strain_history)


def test_generate_strain_history_shape():
    model = J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
    hist = generate_strain_history(model)
    assert hist.ndim == 2
    assert hist.shape[1] == model.ntens
