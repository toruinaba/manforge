"""Weak coupling for the split-stress model: plasticity is exactly undamaged J2.

J2IsotropicDamageUnilateral keeps σ̃ = C0 : ε_e as its own state.  Weak
coupling means σ̃ and ep reproduce undamaged J2 to round-off on any path —
including reversals that switch principal stresses between tension and
compression — while the nominal σ satisfies C0 : ∂ψ*/∂σ(σ, D) = σ̃ and D
never decreases.
"""

import numpy as np
import pytest

import returnmap  # noqa: F401 — enables float64
from returnmap.models import (
    J2Isotropic3D, J2IsotropicPS, J2Isotropic1D, J2IsotropicDamageUnilateral3D,
    J2IsotropicDamageUnilateralPS, J2IsotropicDamageUnilateral1D,
)
from returnmap.simulation import (
    StrainDriver, PythonNumericalIntegrator, FieldHistory, FieldType,
)

P = dict(E=200000.0, nu=0.3, sigma_y0=250.0, H=2000.0)
DP = dict(S=0.5, s=1.0, p_D=0.005, h=0.2)
CASES = [
    (J2IsotropicDamageUnilateral3D, J2Isotropic3D, [1.0, -0.5, -0.4, 0.6, 0.0, 0.3]),
    (J2IsotropicDamageUnilateralPS, J2IsotropicPS, [1.0, 0.3, 0.4]),
    (J2IsotropicDamageUnilateral1D, J2Isotropic1D, [1.0]),
]


def _path(direction):
    amp = np.concatenate([np.linspace(0.0, 0.02, 21)[1:],
                          np.linspace(0.02, -0.02, 41)[1:],
                          np.linspace(-0.02, 0.01, 31)[1:]])
    return np.outer(amp, direction)


@pytest.mark.slow  # ~1–2 s per case (93 steps, nested autodiff)
@pytest.mark.parametrize("uni_cls,ref_cls,direction", CASES)
def test_effective_history_is_undamaged_j2(uni_cls, ref_cls, direction):
    m, ref = uni_cls(**P, **DP), ref_cls(**P)
    load = FieldHistory(FieldType.STRAIN, "Strain", _path(direction))
    r_m = StrainDriver(PythonNumericalIntegrator(m)).run(load)
    r_ref = StrainDriver(PythonNumericalIntegrator(ref)).run(load)
    C0 = np.asarray(m.undamaged_stiffness())

    D_prev = 0.0
    for step, step_ref in zip(r_m.step_results, r_ref.step_results):
        s_eff = np.asarray(step.state["stress_eff"])
        np.testing.assert_allclose(s_eff, np.asarray(step_ref.stress), rtol=1e-9, atol=1e-7)
        assert float(step.state["ep"]) == pytest.approx(float(step_ref.state["ep"]),
                                                        rel=1e-9, abs=1e-14)
        D = float(step.state["D"])
        np.testing.assert_allclose(
            C0 @ np.asarray(m.elastic_strain(np.asarray(step.stress), D)), s_eff,
            rtol=1e-9, atol=1e-7)
        assert D >= D_prev - 1e-15
        D_prev = D
    assert D_prev > 0.01, "path too short to exercise damage"
