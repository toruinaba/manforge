"""Pure-logic tests for J2IsotropicDamage (no integrator)."""

import numpy as np
import pytest

import manforge  # noqa: F401 — enables float64
from manforge.core.dimension import SOLID_3D
from manforge.models import (
    J2Isotropic, J2IsotropicDamage, J2IsotropicDamage3D, J2IsotropicDamagePS,
    J2IsotropicDamage1D,
)

P = dict(E=200000.0, nu=0.3, sigma_y0=250.0, H=2000.0)
DP = dict(S=0.5, s=1.5, p_D=0.01)


@pytest.fixture
def rng():
    return np.random.default_rng(7)


@pytest.mark.parametrize("cls", [J2IsotropicDamage3D, J2IsotropicDamagePS, J2IsotropicDamage1D])
def test_energy_release_rate_is_complementary_energy(cls, rng):
    """Y in triaxiality-free form equals σ̃ : C0⁻¹ : σ̃ / 2 (engineering-shear C0)."""
    m = cls(**P, **DP)
    sig = 100.0 * rng.standard_normal(m.ntens)
    C0 = np.asarray(m.undamaged_stiffness())
    expected = 0.5 * sig @ np.linalg.solve(C0, sig)
    assert float(m.energy_release_rate(sig)) == pytest.approx(expected, rel=1e-12)


def test_undamaged_state_reduces_to_j2(rng):
    m = J2IsotropicDamage3D(**P, **DP)
    ref = J2Isotropic(dimension=SOLID_3D, **P)
    sig = 300.0 * rng.standard_normal(6)
    st = {"stress": sig, "ep": 0.02, "D": 0.0}
    assert float(m.yield_function(st)) == pytest.approx(float(ref.yield_function(st)))
    np.testing.assert_allclose(np.asarray(m.elastic_stiffness(st)),
                               np.asarray(ref.elastic_stiffness(st)))


def test_yield_and_flow_live_in_effective_space(rng):
    """f(σ, D) = f_J2(σ/(1−D)); flow is ∂f/∂σ̃, not ∂f/∂σ."""
    m = J2IsotropicDamage3D(**P, **DP)
    ref = J2Isotropic(dimension=SOLID_3D, **P)
    sig, D = 300.0 * rng.standard_normal(6), 0.3
    st = {"stress": sig, "ep": 0.02, "D": D}
    st_eff = {"stress": sig / (1 - D), "ep": 0.02}
    assert float(m.yield_function(st)) == pytest.approx(float(ref.yield_function(st_eff)))
    np.testing.assert_allclose(np.asarray(m.flow(st).strain_like),
                               np.asarray(ref.flow(st_eff).strain_like), rtol=1e-12)


def test_damaged_stiffness_scales_c0():
    m = J2IsotropicDamage3D(**P, **DP)
    C0 = np.asarray(m.elastic_stiffness(None))
    np.testing.assert_allclose(np.asarray(m.elastic_stiffness({"D": 0.25})), 0.75 * C0)


def test_damage_residual_below_threshold_is_zero(rng):
    """ep_n, ep_{n+1} < p_D → R_D = D − D_n (no growth)."""
    m = J2IsotropicDamage3D(**P, **DP)
    sig = 300.0 * rng.standard_normal(6)
    st_n = {"stress": sig, "ep": 0.002, "D": 0.0}
    st_1 = {"stress": sig, "ep": 0.005, "D": 0.0}
    items = {i.name: i.value for i in m.state_residual(
        st_1, 0.003, st_n, stress_trial=sig)}
    assert abs(float(items["D"])) < 1e-12


def test_state_declaration():
    assert J2IsotropicDamage.implicit_state_names == ["stress", "D"]
    assert J2IsotropicDamage.state_names == ["stress", "ep", "D"]
