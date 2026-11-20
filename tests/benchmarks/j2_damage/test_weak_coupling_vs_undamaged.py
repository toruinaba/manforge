"""Weak-coupling equivalence: J2IsotropicDamage vs undamaged J2 + post-processed D.

Weak coupling means the effective-space plasticity never sees D.  So the
undamaged J2 history (σ̃, ep) must reproduce ep exactly, and integrating the
Lemaitre law on that history *outside* the model (same backward-Euler
discretisation, evaluated with σ̃_{n+1}) must reproduce D and σ = (1 − D) σ̃.
"""

import numpy as np
import pytest

import returnmap  # noqa: F401 — enables float64
from returnmap.models import (
    J2Isotropic3D, J2IsotropicPS, J2Isotropic1D, J2IsotropicDamage3D,
    J2IsotropicDamagePS, J2IsotropicDamage1D,
)
from returnmap.simulation import (
    StrainDriver, PythonNumericalIntegrator, FieldHistory, FieldType,
)
from returnmap.utils import smooth_macaulay

P = dict(E=200000.0, nu=0.3, sigma_y0=250.0, H=2000.0)
DP = dict(S=0.5, s=1.5, p_D=0.01)

CASES = [
    (J2IsotropicDamage3D, J2Isotropic3D, [1.0, -0.5, -0.5, 0.6, 0.0, 0.3]),
    (J2IsotropicDamagePS, J2IsotropicPS, [1.0, 0.3, 0.4]),
    (J2IsotropicDamage1D, J2Isotropic1D, [1.0]),
]


def _path(direction):
    """Load, partial unload, reload in a fixed direction."""
    amp = np.concatenate([
        np.linspace(0.0, 0.06, 31)[1:],
        np.linspace(0.06, 0.055, 4)[1:],
        np.linspace(0.055, 0.09, 16)[1:],
    ])
    return np.outer(amp, direction)


def _macaulay(x):
    return float(smooth_macaulay(x))


@pytest.mark.parametrize("dmg_cls,ref_cls,direction", CASES)
def test_matches_undamaged_history_times_one_minus_d(dmg_cls, ref_cls, direction):
    dmg, ref = dmg_cls(**P, **DP), ref_cls(**P)
    load = FieldHistory(FieldType.STRAIN, _path(direction))
    r_dmg = StrainDriver(PythonNumericalIntegrator(dmg)).run(load)
    r_ref = StrainDriver(PythonNumericalIntegrator(ref)).run(load)

    D, ep_prev = 0.0, 0.0
    for step_d, step_r in zip(r_dmg.step_results, r_ref.step_results):
        sig_eff = np.asarray(step_r.stress)
        ep = float(step_r.state["ep"])
        Y = float(dmg.energy_release_rate(sig_eff))
        D += (Y / DP["S"]) ** DP["s"] * (
            _macaulay(ep - DP["p_D"]) - _macaulay(ep_prev - DP["p_D"])
        )
        ep_prev = ep

        assert float(step_d.state["ep"]) == pytest.approx(ep, rel=1e-10, abs=1e-14)
        assert float(step_d.state["D"]) == pytest.approx(D, rel=1e-8, abs=1e-12)
        np.testing.assert_allclose(
            np.asarray(step_d.stress), (1.0 - D) * sig_eff, rtol=1e-8, atol=1e-8
        )

    assert D > 0.02, "path too short to exercise damage"
