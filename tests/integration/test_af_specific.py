import numpy as np
"""AF (Armstrong-Frederick) model-specific tests.

Covers physics unique to the AF model:
- hardening_type == "reduced" for all variants
- gamma=0 gives purely linear kinematic backstress (positive axial component)
"""

import pytest
import autograd.numpy as anp

from manforge.models.af_kinematic import AFKinematic3D, AFKinematicPS, AFKinematic1D
from manforge.simulation.integrator import PythonIntegrator


# ---------------------------------------------------------------------------
# hardening_type detection
# ---------------------------------------------------------------------------

def test_hardening_type_af3d():
    assert AFKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0).hardening_type == "reduced"


def test_hardening_type_afps():
    assert AFKinematicPS(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0).hardening_type == "reduced"


def test_hardening_type_af1d():
    assert AFKinematic1D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0).hardening_type == "reduced"


# ---------------------------------------------------------------------------
# gamma=0 limit: purely linear kinematic backstress
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_gamma0_backstress_purely_axial():
    """With gamma=0 and uniaxial loading, backstress is purely axial (linear Prager)."""
    model = AFKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=1000.0, gamma=0.0)
    state0 = model.initial_state()
    deps = (lambda _a: (_a.__setitem__(0, 3e-3), _a)[1])(np.zeros(6))
    _r = PythonIntegrator(model).stress_update(deps, anp.zeros(6), state0)
    assert float(_r.state["alpha"][0]) > 0.0
