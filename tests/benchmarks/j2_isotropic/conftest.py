"""Scenario factory for J2 isotropic benchmarks.

All scenarios are cumulative total-strain histories (shape: (N, ntens)).
The `scenario` fixture parametrizes over scenario keys; tests receive a
(model_factory, strain_history) pair.
"""

import numpy as np
import pytest

from manforge.models.j2_isotropic import J2Isotropic3D, J2Isotropic1D
from manforge.simulation.types import FieldHistory
from manforge.verification.test_cases import estimate_yield_strain


# ---------------------------------------------------------------------------
# Standard steel parameters (consistent with root conftest `model` fixture)
# ---------------------------------------------------------------------------

_E = 210000.0
_NU = 0.3
_SIGMA_Y0 = 250.0
_H = 1000.0


def _j2_3d():
    return J2Isotropic3D(E=_E, nu=_NU, sigma_y0=_SIGMA_Y0, H=_H)


def _j2_1d():
    return J2Isotropic1D(E=_E, nu=_NU, sigma_y0=_SIGMA_Y0, H=_H)


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------

def _uniaxial_monotonic(model_factory):
    model = model_factory()
    eps_y = estimate_yield_strain(model)
    ntens = model.ntens
    data = np.zeros((50, ntens))
    data[:, 0] = np.linspace(0.0, 5e-3, 50)
    return data


def _uniaxial_cyclic(model_factory):
    model = model_factory()
    eps_y = estimate_yield_strain(model)
    ntens = model.ntens
    peaks = [0.0, 5e-3, -5e-3, 3e-3, 0.0]
    return FieldHistory.cyclic_strain(peaks, n_per_segment=24, ntens=ntens, component=0).data


def _pure_shear_monotonic_3d():
    ntens = 6
    data = np.zeros((40, ntens))
    data[:, 3] = np.linspace(0.0, 4e-3, 40)  # γ12 (engineering shear)
    return data


def _triaxial_isochoric_3d():
    """Deviatoric strain history: tr(ε) = 0."""
    ntens = 6
    n = 40
    data = np.zeros((n, ntens))
    t = np.linspace(0.0, 1.0, n)
    data[:, 0] = 3e-3 * t
    data[:, 1] = -1.5e-3 * t
    data[:, 2] = -1.5e-3 * t
    return data


def _multistep_random(model_factory):
    rng = np.random.default_rng(seed=0)
    model = model_factory()
    ntens = model.ntens
    eps_y = estimate_yield_strain(model)
    # random increments in first component, cumsum to get total strain
    increments = rng.uniform(-2.0 * eps_y, 2.0 * eps_y, size=60)
    cumstrain = np.zeros((60, ntens))
    cumstrain[:, 0] = np.cumsum(increments)
    return cumstrain


# ---------------------------------------------------------------------------
# Parametric fixtures
# ---------------------------------------------------------------------------

_3D_SCENARIOS = {
    "uniaxial_monotonic":   lambda: _uniaxial_monotonic(_j2_3d),
    "uniaxial_cyclic":      lambda: _uniaxial_cyclic(_j2_3d),
    "pure_shear_monotonic": _pure_shear_monotonic_3d,
    "triaxial_isochoric":   _triaxial_isochoric_3d,
    "multistep_random":     lambda: _multistep_random(_j2_3d),
}

_1D_SCENARIOS = {
    "uniaxial_monotonic": lambda: _uniaxial_monotonic(_j2_1d),
    "uniaxial_cyclic":    lambda: _uniaxial_cyclic(_j2_1d),
    "multistep_random":   lambda: _multistep_random(_j2_1d),
}


@pytest.fixture(params=[
    pytest.param(("3d", k), id=f"3d_{k}") for k in _3D_SCENARIOS
] + [
    pytest.param(("1d", k), id=f"1d_{k}") for k in _1D_SCENARIOS
])
def j2_scenario(request):
    """Yields (model, strain_history) pairs for Path A parametrization."""
    dim, key = request.param
    if dim == "3d":
        model = _j2_3d()
        history = _3D_SCENARIOS[key]()
    else:
        model = _j2_1d()
        history = _1D_SCENARIOS[key]()
    return model, history


@pytest.fixture(params=[
    pytest.param(k, id=k) for k in _3D_SCENARIOS
])
def j2_3d_scenario(request):
    """Yields (model, strain_history) pairs for Path B (3D only)."""
    key = request.param
    model = _j2_3d()
    history = _3D_SCENARIOS[key]()
    return model, history
