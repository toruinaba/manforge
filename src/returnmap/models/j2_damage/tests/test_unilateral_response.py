"""Integration tests for J2IsotropicDamageUnilateral (stage 2: split stress)."""

import numpy as np
import pytest

import returnmap  # noqa: F401 — enables float64
from returnmap.models import (
    J2IsotropicDamage3D, J2IsotropicDamagePS, J2IsotropicDamage1D,
    J2IsotropicDamageCrackClosure1D, J2IsotropicDamageUnilateral3D,
    J2IsotropicDamageUnilateralPS, J2IsotropicDamageUnilateral1D,
)
from returnmap.simulation import PythonNumericalIntegrator
from returnmap.verification import check_tangent

P = dict(E=200000.0, nu=0.3, sigma_y0=250.0, H=2000.0, S=0.5, s=1.0, p_D=0.01)
DIRS = {6: [1.0, -0.5, -0.5, 0.6, 0.0, 0.3], 3: [1.0, 0.3, 0.4], 1: [1.0]}
PAIRS = [(J2IsotropicDamageUnilateral3D, J2IsotropicDamage3D),
         (J2IsotropicDamageUnilateralPS, J2IsotropicDamagePS),
         (J2IsotropicDamageUnilateral1D, J2IsotropicDamage1D)]


def _advance(model, deps, n, stress=None, state=None):
    # Tight NR tolerance: the default stress-scaled tol leaves ~1e-7 MPa of
    # solution noise at σ ~ 1e3 MPa, which swamps a 1e-8 FD strain step.
    integ = PythonNumericalIntegrator(model, tol=1e-14)
    stress = np.zeros(model.ntens) if stress is None else stress
    state = model.initial_state() if state is None else state
    for _ in range(n):
        r = integ.stress_update(deps, stress, state)
        stress, state = np.asarray(r.stress), r.state
    return integ, stress, state


@pytest.mark.parametrize("uni_cls,base_cls", PAIRS)
def test_h1_reproduces_unsplit_damage_model(uni_cls, base_cls):
    """h = 1 removes the split: identical to J2IsotropicDamage through load/unload."""
    uni, base = uni_cls(**P, h=1.0), base_cls(**P)
    d = np.array(DIRS[uni.ntens]) * 2e-3
    _, su, stu = _advance(uni, d, 15)
    _, sb, stb = _advance(base, d, 15)
    _, su, stu = _advance(uni, -0.3 * d, 2, su, stu)
    _, sb, stb = _advance(base, -0.3 * d, 2, sb, stb)
    np.testing.assert_allclose(su, sb, atol=1e-8)
    assert float(stu["D"]) == pytest.approx(float(stb["D"]), rel=1e-10)


def test_1d_compression_damage_and_stiffness():
    """1D compression: Y matches the stage-1 crack-closure model; σ = (1−hD) σ̃."""
    h = 0.2
    _, s, st = _advance(J2IsotropicDamageUnilateral1D(**P, h=h), np.array([-2e-3]), 20)
    _, _, st_cc = _advance(J2IsotropicDamageCrackClosure1D(**P, h=h), np.array([-2e-3]), 20)
    assert float(st["D"]) == pytest.approx(float(st_cc["D"]), rel=1e-10)
    assert float(s[0]) == pytest.approx((1 - h * float(st["D"])) * float(st["stress_eff"][0]),
                                        rel=1e-10)


def test_1d_stiffness_recovers_on_crack_closure():
    """Damage in tension, then one elastic step from tension into compression."""
    h = 0.2
    m = J2IsotropicDamageUnilateral1D(**P, h=h)
    integ, s, st = _advance(m, np.array([2e-3]), 20)
    D = float(st["D"])
    s_eff_n = float(st["stress_eff"][0])
    deps = -2.0 * s_eff_n / P["E"]            # σ̃: +s → −s, elastic
    r = integ.stress_update(np.array([deps]), s, st)
    assert not r.is_plastic
    assert float(r.stress[0]) == pytest.approx((1 - h * D) * -s_eff_n, rel=1e-10)
    assert float(np.asarray(r.ddsdde)[0, 0]) == pytest.approx((1 - h * D) * P["E"], rel=1e-10)


@pytest.mark.parametrize("cls,direction", [
    (J2IsotropicDamageUnilateral3D, DIRS[6]),                    # tr σ ≈ 0: on the kink
    (J2IsotropicDamageUnilateral3D, [1.0, -0.4, -0.4, 0.6, 0.0, 0.3]),
    (J2IsotropicDamageUnilateralPS, DIRS[3]),
    (J2IsotropicDamageUnilateral1D, DIRS[1]),
])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_plastic_tangent_matches_fd(cls, direction, sign):
    m = cls(**P, h=0.2)
    d = sign * np.array(direction) * 2e-3
    integ, s, st = _advance(m, d, 15)
    assert float(st["D"]) > 1e-3
    res = check_tangent(integ, d, s, st, eps=1e-6)
    assert res.passed, f"max rel err = {res.max_rel_err:.3e}"


def _mixed_state(m, D=0.3):
    st = {**m.initial_state(), "D": np.array(D),
          "stress_eff": np.array([100.0, -80.0, 20.0, 30.0, 0.0, 10.0])}
    st["stress"] = np.asarray(m.nominal_from_effective(st["stress_eff"], st["D"]))
    return st


def test_elastic_tangent_symmetric_and_matches_fd():
    m = J2IsotropicDamageUnilateral3D(**P, h=0.2)
    st = _mixed_state(m)
    integ = PythonNumericalIntegrator(m)
    deps = np.array([1e-6, 0, 0, 0, 0, 0])
    r = integ.stress_update(deps, st["stress"], st)
    assert not r.is_plastic
    K = np.asarray(r.ddsdde)
    mf = np.asarray(m.dimension.mandel_factors_np)
    Km = np.diag(mf) @ K @ np.diag(mf)
    np.testing.assert_allclose(Km, Km.T, atol=1e-12 * np.max(np.abs(Km)))
    res = check_tangent(integ, deps, st["stress"], st, eps=1e-9)
    assert res.passed, f"max rel err = {res.max_rel_err:.3e}"


def test_closed_elastic_loop_does_no_work():
    """A potential exists: ∮σ:dε over a closed elastic strain loop vanishes.

    The loop crosses principal-sign boundaries (mixed tension/compression).
    """
    m = J2IsotropicDamageUnilateral3D(**P, h=0.2)
    st = _mixed_state(m)
    integ = PythonNumericalIntegrator(m)
    n, r = 400, 2e-4
    th = np.linspace(0.0, 2 * np.pi, n + 1)
    path = r * np.stack([np.cos(th), np.sin(th), 0 * th, 0.5 * np.sin(th), 0 * th, 0 * th], 1)
    stress, state, W = st["stress"], st, 0.0
    for k in range(n):
        res = integ.stress_update(path[k + 1] - path[k], stress, state)
        assert not res.is_plastic
        W += 0.5 * (stress + np.asarray(res.stress)) @ (path[k + 1] - path[k])
        stress, state = np.asarray(res.stress), res.state
    energy = 0.5 * stress @ np.linalg.solve(np.asarray(m.undamaged_stiffness()), stress)
    assert abs(W) / energy < 1e-5
    np.testing.assert_allclose(stress, st["stress"], atol=1e-8)
