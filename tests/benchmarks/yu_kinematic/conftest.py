"""Scenario factory for YU kinematic hardening benchmarks.

All scenarios are cumulative total-strain histories (shape: (N, ntens)).
Two fixtures:
  yu_3d_scenario  — 3D only; used for analytical vs autograd comparison
                    (user_defined_return_mapping is 3D-only)
  yu_smoke_scenario — 3D + 1D; used for smoke convergence checks
"""

import numpy as np
import pytest

from manforge.models import YUKinematic1D, YUKinematic3D
from manforge.simulation.types import FieldHistory

PARAMS = dict(
    E=206_000, nu=0.3, Y=360.0, C_1=2000.0, C_2=200.0,
    B=435.0, Rsat=255.0, k=26.0, b=66.0, h=0.4, Ea=159_000, xi=61.0,
)

# ---------------------------------------------------------------------------
# 3D scenario builders
# B-Y = 435 - 360 = 75 MPa: keep theta_max << 75 for non-slow scenarios
# ---------------------------------------------------------------------------

def _3d_uniaxial_monotonic():
    n = 50
    data = np.zeros((n, 6))
    data[:, 0] = np.linspace(0.0, 5e-3, n)
    return data


def _3d_small_amplitude_cyclic():
    return FieldHistory.cyclic_strain(
        [1e-3, -1e-3, 1e-3], n_per_segment=15, ntens=6,
    ).data


def _3d_uniaxial_cyclic():
    return FieldHistory.cyclic_strain(
        [0.05, -0.05, 0.05, -0.05], n_per_segment=50, ntens=6,
    ).data


# ---------------------------------------------------------------------------
# 1D scenario builders
# ---------------------------------------------------------------------------

def _1d_uniaxial_monotonic():
    n = 50
    data = np.zeros((n, 1))
    data[:, 0] = np.linspace(0.0, 5e-3, n)
    return data


def _1d_uniaxial_cyclic():
    return FieldHistory.cyclic_strain(
        [0.05, -0.05, 0.05, -0.05], n_per_segment=50, ntens=1,
    ).data


# ---------------------------------------------------------------------------
# Scenario registries: (builder, is_slow)
# ---------------------------------------------------------------------------

_3D_SCENARIOS = {
    "uniaxial_monotonic":     (_3d_uniaxial_monotonic, False),
    "small_amplitude_cyclic": (_3d_small_amplitude_cyclic, False),
    "uniaxial_cyclic":        (_3d_uniaxial_cyclic, True),
}

_1D_SCENARIOS = {
    "uniaxial_monotonic": (_1d_uniaxial_monotonic, False),
    "uniaxial_cyclic":    (_1d_uniaxial_cyclic, True),
}


def _params(prefix, scenarios):
    return [
        pytest.param(
            (prefix, k),
            marks=[pytest.mark.slow] if slow else [],
            id=f"{prefix}_{k}",
        )
        for k, (_, slow) in scenarios.items()
    ]


# ---------------------------------------------------------------------------
# Parametric fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=_params("3d", _3D_SCENARIOS))
def yu_3d_scenario(request):
    """Yields (model, history_data) for 3D scenarios only."""
    _, key = request.param
    builder, _ = _3D_SCENARIOS[key]
    return YUKinematic3D(**PARAMS), builder()


@pytest.fixture(params=_params("3d", _3D_SCENARIOS) + _params("1d", _1D_SCENARIOS))
def yu_smoke_scenario(request):
    """Yields (model, history_data) for all 3D + 1D scenarios."""
    dim, key = request.param
    if dim == "3d":
        builder, _ = _3D_SCENARIOS[key]
        return YUKinematic3D(**PARAMS), builder()
    builder, _ = _1D_SCENARIOS[key]
    return YUKinematic1D(**PARAMS), builder()
