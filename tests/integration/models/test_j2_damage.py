"""Integration tests for J2IsotropicDamage: tangent, elastic branch, threshold."""

import numpy as np
import pytest

import returnmap  # noqa: F401 — enables float64
from returnmap.models import (
    J2IsotropicDamage3D, J2IsotropicDamagePS, J2IsotropicDamage1D,
)
from returnmap.simulation import (
    StressDriver, PythonNumericalIntegrator, FieldHistory, FieldType,
)
from returnmap.verification import check_tangent

P = dict(E=200000.0, nu=0.3, sigma_y0=250.0, H=2000.0)
DP = dict(S=0.5, s=1.0, p_D=0.01)
# Isochoric directions: volumetric loading would drive σ_H, and hence Y, up
# elastically without bound.
DIRS = {6: [1.0, -0.5, -0.5, 0.6, 0.0, 0.3], 3: [1.0, 0.3, 0.4], 1: [1.0]}
CLASSES = [J2IsotropicDamage3D, J2IsotropicDamagePS, J2IsotropicDamage1D]


def _advance(model, deps, n):
    integ = PythonNumericalIntegrator(model)
    stress, state = np.zeros(model.ntens), model.initial_state()
    for _ in range(n):
        r = integ.stress_update(deps, stress, state)
        stress, state = np.asarray(r.stress), r.state
    return integ, stress, state


@pytest.mark.parametrize("cls", CLASSES)
def test_plastic_tangent_matches_fd_with_active_damage(cls):
    m = cls(**P, **DP)
    deps = np.array(DIRS[m.ntens]) * 2e-3
    integ, stress, state = _advance(m, deps, 20)
    assert float(state["D"]) > 0.01
    res = check_tangent(integ, deps, stress, state, eps=1e-8)
    assert res.passed, f"max rel err = {res.max_rel_err:.3e}"


@pytest.mark.parametrize("cls", CLASSES)
def test_unloading_freezes_damage_and_uses_damaged_stiffness(cls):
    m = cls(**P, **DP)
    deps = np.array(DIRS[m.ntens]) * 2e-3
    integ, stress, state = _advance(m, deps, 20)
    r = integ.stress_update(-0.1 * deps, stress, state)
    assert not r.is_plastic
    assert float(r.state["D"]) == float(state["D"])
    C0 = np.asarray(m.undamaged_stiffness())
    np.testing.assert_allclose(np.asarray(r.ddsdde), (1 - float(state["D"])) * C0)
    np.testing.assert_allclose(
        np.asarray(r.stress), stress + (1 - float(state["D"])) * C0 @ (-0.1 * deps)
    )


def test_no_damage_below_threshold():
    m = J2IsotropicDamage3D(**P, **{**DP, "p_D": 1.0})
    deps = np.array(DIRS[6]) * 2e-3
    _, _, state = _advance(m, deps, 10)
    assert float(state["ep"]) > 0.0
    assert abs(float(state["D"])) < 1e-12


def test_stress_driver_uses_nominal_stress():
    """StressDriver targets nominal σ and converges with the damaged tangent."""
    m = J2IsotropicDamage1D(**P, **DP)
    sig = np.concatenate([np.linspace(0.0, 300.0, 11)[1:], np.linspace(300.0, 340.0, 9)[1:]])
    load = FieldHistory(FieldType.STRESS, "sigma11", sig)
    res = StressDriver(PythonNumericalIntegrator(m)).run(load)
    np.testing.assert_allclose(np.asarray(res.stress)[:, 0], sig, atol=1e-6)


# ---------------------------------------------------------------------------
# Crack-closure variant
# ---------------------------------------------------------------------------

from returnmap.models import (  # noqa: E402
    J2IsotropicDamage1D as _Base1D,
    J2IsotropicDamageCrackClosure3D, J2IsotropicDamageCrackClosurePS,
    J2IsotropicDamageCrackClosure1D,
)

CC_CASES = [
    # uniaxial-like 3D path: repeated principal values throughout
    (J2IsotropicDamageCrackClosure3D, [1.0, -0.5, -0.5, 0.0, 0.0, 0.0]),
    (J2IsotropicDamageCrackClosure3D, DIRS[6]),
    (J2IsotropicDamageCrackClosurePS, DIRS[3]),
    (J2IsotropicDamageCrackClosure1D, DIRS[1]),
]


@pytest.mark.parametrize("cls,direction", CC_CASES)
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_crack_closure_tangent_matches_fd(cls, direction, sign):
    m = cls(**P, **DP, h=0.2)
    deps = sign * np.array(direction) * 2e-3
    integ, stress, state = _advance(m, deps, 20)
    assert float(state["D"]) > 1e-3
    res = check_tangent(integ, deps, stress, state, eps=1e-8)
    assert res.passed, f"max rel err = {res.max_rel_err:.3e}"


@pytest.mark.parametrize("h", [0.0, 0.2, 1.0])
def test_crack_closure_1d_compression_damages_h_times_tension(h):
    """1D, s = 1: |σ̃| and ep histories are sign-symmetric, so D_comp = h · D_tens."""
    m = J2IsotropicDamageCrackClosure1D(**P, **DP, h=h)
    deps = np.array([2e-3])
    _, s_t, st_t = _advance(m, deps, 20)
    _, s_c, st_c = _advance(m, -deps, 20)
    _, _, st_base = _advance(_Base1D(**P, **DP), deps, 20)
    assert float(st_t["D"]) == pytest.approx(float(st_base["D"]), rel=1e-10)
    assert float(st_c["D"]) == pytest.approx(h * float(st_t["D"]), rel=1e-8, abs=1e-14)
