"""Tests for FlowVector and the MaterialModel.flow hook.

Covers:
- FlowVector rejects raw arithmetic (the shear convention must be named)
- strain_like / stress_like differ by [1,..,1, 2,..,2] on shear
- strain_flow / stress_flow round-trip
- Default flow() is autodiff ∂f/∂σ, tagged strain-like
- flow() must return a FlowVector (raw ndarray raises TypeError)
- Model flow matches the textbook (3/2)s/‖s‖ under shear loading
"""

import numpy as np
import pytest
import autograd
import autograd.numpy as anp

from manforge.core.state import FlowVector, Explicit, NTENS, SCALAR, _state_with_stress
from manforge.core.material import MaterialModel
from manforge.core.dimension import SOLID_3D, PLANE_STRESS, UNIAXIAL_1D
from manforge.models import (
    AFKinematic3D, AFKinematicPS, OWKinematic3D, J2Isotropic3D,
)
from manforge.models.yu_kinematic import YUKinematic3D


# A state with nonzero shear — the only regime where the 2× error shows up.
SIG_3D = np.array([300.0, 50.0, -20.0, 40.0, 10.0, 5.0])
ALPHA_3D = np.array([10.0, -4.0, -6.0, 3.0, 1.0, 2.0])


def _af(**kw):
    p = dict(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=20000.0, gamma=100.0)
    p.update(kw)
    return AFKinematic3D(**p)


# ---------------------------------------------------------------------------
# FlowVector mechanics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op,expected", [
    (lambda n: n * 2.0, "multiplication"),
    (lambda n: 2.0 * n, "multiplication"),
    (lambda n: n / 2.0, "division"),
    (lambda n: 2.0 / n, "division"),
    (lambda n: n + 1.0, "addition"),
    (lambda n: 1.0 + n, "addition"),
    (lambda n: n - 1.0, "subtraction"),
    (lambda n: 1.0 - n, "subtraction"),
    (lambda n: -n, "negation"),
    (lambda n: n @ np.eye(6), "matrix multiplication"),
    # numpy tries __array__ before __rmatmul__, so C @ n trips the ndarray guard.
    (lambda n: np.eye(6) @ n, "conversion to ndarray"),
    (lambda n: n[0], "indexing"),
    (lambda n: len(n), "len()"),
    (lambda n: list(n), "iteration"),
    (lambda n: np.asarray(n), "conversion to ndarray"),
])
def test_flow_vector_rejects_raw_arithmetic(op, expected):
    """Ambiguous uses must fail loudly, naming the operation the caller wrote."""
    n = FlowVector(SIG_3D, SOLID_3D.eng_to_phys_strain_factors_np)
    with pytest.raises(TypeError, match="shear convention") as excinfo:
        op(n)
    assert expected in str(excinfo.value)


def test_flow_vector_shear_factor_is_two():
    n = FlowVector(SIG_3D, SOLID_3D.eng_to_phys_strain_factors_np)
    ratio = n.strain_like / n.stress_like
    np.testing.assert_allclose(ratio, [1.0, 1.0, 1.0, 2.0, 2.0, 2.0])


def test_strain_flow_preserves_value():
    m = _af()
    n = m.strain_flow(SIG_3D)
    np.testing.assert_allclose(n.strain_like, SIG_3D)


def test_stress_flow_preserves_value():
    m = _af()
    n = m.stress_flow(SIG_3D)
    np.testing.assert_allclose(n.stress_like, SIG_3D)


def test_strain_flow_and_stress_flow_round_trip():
    """Tagging the same physical direction either way yields the same object."""
    m = _af()
    via_stress = m.stress_flow(SIG_3D)
    via_strain = m.strain_flow(m.dimension.to_strain_like(SIG_3D))
    np.testing.assert_allclose(via_stress.strain_like, via_strain.strain_like)
    np.testing.assert_allclose(via_stress.stress_like, via_strain.stress_like)


@pytest.mark.parametrize("dim", [SOLID_3D, PLANE_STRESS, UNIAXIAL_1D])
def test_dimension_conversions_are_mutually_inverse(dim):
    v = np.arange(1.0, dim.ntens + 1.0)
    np.testing.assert_allclose(dim.to_stress_like(dim.to_strain_like(v)), v)
    np.testing.assert_allclose(dim.to_strain_like(dim.to_stress_like(v)), v)


# ---------------------------------------------------------------------------
# Default flow() == autodiff ∂f/∂σ, strain-like
# ---------------------------------------------------------------------------

def test_default_flow_matches_autodiff_gradient():
    m = _af()
    state = m.make_state(stress=SIG_3D, alpha=ALPHA_3D, ep=0.0)
    expected = autograd.grad(
        lambda s: m.yield_function(_state_with_stress(state, s))
    )(SIG_3D)
    np.testing.assert_allclose(m.flow(state).strain_like, expected)


def test_default_flow_returns_flow_vector():
    m = _af()
    state = m.make_state(stress=SIG_3D, alpha=ALPHA_3D, ep=0.0)
    assert isinstance(m.flow(state), FlowVector)


@pytest.mark.parametrize("model,state_kw", [
    (_af(), dict(alpha=ALPHA_3D, ep=0.0)),
    (OWKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=20000.0, gamma=100.0),
     dict(alpha=ALPHA_3D, ep=0.0)),
])
def test_stress_like_flow_matches_textbook_formula(model, state_kw):
    """n̂ = (3/2)·s_ξ/‖s_ξ‖ — the form written in the literature."""
    state = model.make_state(stress=SIG_3D, **state_kw)
    s_xi = model.dev(SIG_3D) - state_kw["alpha"]
    textbook = 1.5 * s_xi / model.vonmises_norm(s_xi)
    np.testing.assert_allclose(model.flow(state).stress_like, textbook, atol=1e-15)


def test_j2_stress_like_flow_matches_textbook_formula():
    m = J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
    state = m.make_state(stress=SIG_3D, ep=0.0)
    s = m.dev(SIG_3D)
    textbook = 1.5 * s / m.vonmises(SIG_3D)
    np.testing.assert_allclose(m.flow(state).stress_like, textbook, atol=1e-15)


def test_plane_stress_flow_matches_textbook_formula():
    m = AFKinematicPS(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=20000.0, gamma=100.0)
    sig = np.array([300.0, 50.0, 40.0])
    alpha = np.array([10.0, -4.0, 3.0])
    state = m.make_state(stress=sig, alpha=alpha, ep=0.0)
    s_xi = m.dev(sig) - alpha
    textbook = 1.5 * s_xi / m.vonmises_norm(s_xi)
    np.testing.assert_allclose(m.flow(state).stress_like, textbook, atol=1e-15)


# ---------------------------------------------------------------------------
# YU3D overrides flow with its hand-derived expression
# ---------------------------------------------------------------------------

def test_yu3d_flow_matches_calc_norm_n_flow():
    m = YUKinematic3D(E=210000.0, nu=0.3, Y=150.0, C_1=300.0, C_2=200.0, B=200.0,
                      Rsat=100.0, k=20.0, b=30.0, h=0.5, Ea=180000.0, xi=20.0)
    theta = np.array([10.0, -4.0, -6.0, 3.0, 1.0, 2.0])
    beta = np.array([5.0, -2.0, -3.0, 1.0, 0.5, 0.5])
    state = m.make_state(stress=SIG_3D, theta=theta, beta=beta, R=10.0,
                         q=np.zeros(6), r=0.0, eps_eq=0.01, theta_max=12.0)
    xi = m.dev(SIG_3D) - theta - beta
    _, expected = m.calc_norm_n_flow(xi)
    np.testing.assert_allclose(m.flow(state).strain_like, expected)


# ---------------------------------------------------------------------------
# Return-contract enforcement
# ---------------------------------------------------------------------------

def test_unwrapped_flow_return_raises():
    """Returning a bare ndarray from flow() must fail at the framework boundary."""

    class _BadFlowModel(MaterialModel):
        param_names = ["sigma_y0"]
        ep = Explicit(shape=SCALAR, doc="eq plastic strain")

        def __init__(self, *, sigma_y0):
            super().__init__(dimension=SOLID_3D)
            self.E = 210000.0
            self.nu = 0.3
            self.sigma_y0 = sigma_y0

        def yield_function(self, state):
            return self.vonmises(state["stress"]) - self.sigma_y0

        def update_state(self, dlambda, state_new, state_n, *,
                         stress_trial=None, strain_inc=None):
            return [self.ep(state_n["ep"] + dlambda)]

        def flow(self, state):
            s = self.dev(state["stress"])
            return 1.5 * s / self.vonmises(state["stress"])  # missing the wrapper

    m = _BadFlowModel(sigma_y0=250.0)
    state = m.make_state(stress=SIG_3D, ep=0.0)

    # The check lives on the override itself, so it fires wherever flow is
    # reached — not only via default_stress_residual.
    with pytest.raises(TypeError, match="must return a FlowVector"):
        m.flow(state)
    with pytest.raises(TypeError, match="must return a FlowVector"):
        m.default_stress_residual(state, anp.array(0.0), SIG_3D)


def test_unwrapped_flow_in_update_state_raises():
    """A bare ndarray must not surface as AttributeError at a .stress_like use site."""

    class _BadFlowAF(AFKinematic3D):
        def flow(self, state):
            s_xi = self.dev(state["stress"]) - state["alpha"]
            return 1.5 * s_xi / self.vonmises_norm(s_xi)  # missing the wrapper

    m = _BadFlowAF(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=20000.0, gamma=100.0)
    state_n = m.make_state(stress=np.zeros(6), alpha=np.zeros(6), ep=0.0)
    state_new = m.make_state(stress=SIG_3D, alpha=np.zeros(6), ep=0.0)
    with pytest.raises(TypeError, match="must return a FlowVector"):
        m.update_state(anp.array(1e-3), state_new, state_n)


def test_valid_override_is_not_double_wrapped():
    """Re-deriving from a class with a checked flow must not stack wrappers."""

    class _Child(AFKinematic3D):
        pass

    class _GrandChild(_Child):
        def flow(self, state):
            return self.stress_flow(np.ones(6))

    m = _GrandChild(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=20000.0, gamma=100.0)
    state = m.make_state(stress=SIG_3D, alpha=ALPHA_3D, ep=0.0)
    np.testing.assert_allclose(m.flow(state).stress_like, np.ones(6))
