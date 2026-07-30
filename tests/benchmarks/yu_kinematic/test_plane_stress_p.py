"""Numerical equivalence harness for YUKinematicPS under the P-metric convention.

YUKinematicPS stores in-plane tensor components with the 33 component
identically zero and routes deviatoric contractions through PLANE_STRESS_P's
P metric.  It is solved purely by the framework autograd path.

Three independent axes, none of which needs a hand-derived Jacobian:

  1. Uniaxial equivalence — under uniaxial loading (σ22 = σ12 = 0) the model
     must reproduce YUKinematic1D and YUKinematic3D exactly.  This validates
     the convention physically rather than algebraically.
  2. Tangent vs finite differences — the AD consistent tangent must match
     central differences.
  3. Cyclic convergence — every step of a reversed history must converge.
     Guards the θ → 0 singularity of the √(a/θ̄) term, whose Jacobian grows
     like 1/√θ̄ and is left to smooth_sqrt's eps regularisation.
"""

import numpy as np
import pytest

from manforge.core import PLANE_STRESS_P
from manforge.core.dimension import P_PLANE_STRESS
from manforge.models import YUKinematic1D, YUKinematic3D, YUKinematicPS
from manforge.simulation.driver import MixedDriver, StrainDriver
from manforge.simulation.integrator import PythonNumericalIntegrator
from manforge.simulation.types import FieldHistory
from manforge.verification import JacobianChecker
from manforge.verification.tangent import TangentChecker

from .conftest import PARAMS


def _initial_state(model):
    return dict(
        stress=np.zeros(model.ntens),
        theta=np.zeros(model.ntens),
        beta=np.zeros(model.ntens),
        R=0.0,
        q=np.zeros(model.ntens),
        r=0.0,
        eps_eq=0.0,
        theta_max=0.0,
    )


# ---------------------------------------------------------------------------
# Convention wiring
# ---------------------------------------------------------------------------

def test_uses_plane_stress_p_dimension():
    model = YUKinematicPS(**PARAMS)
    assert model.dimension is PLANE_STRESS_P
    np.testing.assert_allclose(model.P, P_PLANE_STRESS)


def test_dev_is_identity():
    """P carries the deviatoric projection, so dev must not project again."""
    model = YUKinematicPS(**PARAMS)
    stress = np.array([120.0, -45.0, 30.0])
    np.testing.assert_allclose(model.dev(stress), stress)


def test_p_metric_equals_3d_deviatoric_contraction():
    """sᵀPt reproduces dev₃D(s):dev₃D(t) for tensors whose 33 component is zero."""
    model = YUKinematicPS(**PARAMS)
    rng = np.random.default_rng(0)
    s = rng.normal(size=3) * 50.0
    t = rng.normal(size=3) * 50.0

    def dev_3d(v):
        full = np.array([v[0], v[1], 0.0])
        p = full.sum() / 3.0
        return np.array([full[0] - p, full[1] - p, -p, v[2]])

    ds, dt = dev_3d(s), dev_3d(t)
    expected = ds[0] * dt[0] + ds[1] * dt[1] + ds[2] * dt[2] + 2.0 * ds[3] * dt[3]
    np.testing.assert_allclose(
        model.deviatoric_inner_product(s, t), expected, rtol=1e-12
    )


def test_yield_function_is_quadratic_form():
    model = YUKinematicPS(**PARAMS)
    Y = PARAMS["Y"]
    state = model.make_state(**_initial_state(model))
    assert model.yield_function(state) == pytest.approx(-Y * Y / 3.0)


def test_strain_norm_uses_3d_deviatoric_convention():
    """Strain has ε33 = −(ε11 + ε22) ≠ 0, so the P metric must not be used here."""
    model = YUKinematicPS(**PARAMS)
    e = 1e-2
    eps_p = np.array([e, -e / 2.0, 0.0])
    np.testing.assert_allclose(model.strain_norm(eps_p), e, rtol=1e-10)


# ---------------------------------------------------------------------------
# Axis 1: uniaxial equivalence against 1D and 3D
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("peaks,n_per_segment", [
    ([5e-3], 40),
    pytest.param([0.02, -0.02, 0.02], 25, marks=pytest.mark.slow),
])
def test_uniaxial_matches_1d(peaks, n_per_segment):
    """Uniaxial plane stress must reproduce the 1D model step for step."""
    history = FieldHistory.cyclic_strain(peaks, n_per_segment=n_per_segment, ntens=1)

    result_1d = StrainDriver(
        PythonNumericalIntegrator(YUKinematic1D(**PARAMS))
    ).run(history)
    result_ps = MixedDriver(
        PythonNumericalIntegrator(YUKinematicPS(**PARAMS)),
        prescribed_strain_idx=[0],
    ).run(history)

    # The stress-controlled components must be driven to zero.
    np.testing.assert_allclose(result_ps.stress[:, 1:], 0.0, atol=1e-8)
    np.testing.assert_allclose(
        result_ps.stress[:, 0], result_1d.stress[:, 0], rtol=1e-9, atol=1e-8
    )


def test_uniaxial_matches_3d():
    """Uniaxial plane stress must agree with the 3D model under the same BCs."""
    history = FieldHistory.cyclic_strain([5e-3], n_per_segment=40, ntens=1)

    result_3d = MixedDriver(
        PythonNumericalIntegrator(YUKinematic3D(**PARAMS)),
        prescribed_strain_idx=[0],
    ).run(history)
    result_ps = MixedDriver(
        PythonNumericalIntegrator(YUKinematicPS(**PARAMS)),
        prescribed_strain_idx=[0],
    ).run(history)

    np.testing.assert_allclose(
        result_ps.stress[:, 0], result_3d.stress[:, 0], rtol=1e-9, atol=1e-8
    )


# ---------------------------------------------------------------------------
# Axis 2: AD consistent tangent vs central differences
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strain_inc", [
    np.array([1e-4, 0.0, 0.0]),
    np.array([0.0, 0.0, 1e-4]),
    np.array([8e-5, -3e-5, 5e-5]),
])
def test_tangent_vs_finite_difference_elastic(strain_inc):
    model = YUKinematicPS(**PARAMS)
    integrator = PythonNumericalIntegrator(model)
    result = TangentChecker(integrator).check(
        np.zeros(3), _initial_state(model), strain_inc
    )
    assert result.passed, f"max_rel_err={result.max_rel_err:.3e}"


def test_tangent_vs_finite_difference_plastic():
    """Check the tangent at a plastically loaded state, not just the origin."""
    model = YUKinematicPS(**PARAMS)
    integrator = PythonNumericalIntegrator(model)

    stress = np.zeros(3)
    state = _initial_state(model)
    strain_inc = np.array([2e-4, 0.0, 0.0])
    for _ in range(20):
        step = integrator.stress_update(strain_inc, stress, state)
        stress, state = step.stress, step.state
    assert step.is_plastic, "setup did not reach the plastic regime"

    result = TangentChecker(integrator).check(stress, state, strain_inc)
    assert result.passed, f"max_rel_err={result.max_rel_err:.3e}"


# ---------------------------------------------------------------------------
# Axis 3: NR convergence over a reversed history
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("peaks", [
    [5e-3],
    [1e-3, -1e-3, 1e-3],
    pytest.param([0.02, -0.02, 0.02, -0.02], marks=pytest.mark.slow),
])
def test_cyclic_convergence(peaks):
    """Every step must converge, including first yield where θ = 0.

    The √(a/θ̄) term in R_theta has a Jacobian that diverges like 1/√θ̄ as
    θ → 0; this test is what establishes that smooth_sqrt's eps regularisation
    is sufficient in practice.
    """
    model = YUKinematicPS(**PARAMS)
    history = FieldHistory.cyclic_strain(peaks, n_per_segment=25, ntens=1)
    driver = MixedDriver(
        PythonNumericalIntegrator(model), prescribed_strain_idx=[0]
    )

    n_steps = 0
    for step in driver.iter_run(history):
        n_steps += 1
        assert step.converged, f"outer NR failed at step {step.i}"
        if step.result.is_plastic:
            assert step.result.return_mapping.converged, (
                f"return mapping failed at step {step.i}: "
                f"residual_history={step.result.residual_history}"
            )
    assert n_steps == len(history.data)


def test_plastic_step_is_on_yield_surface():
    """The converged state must satisfy f = 0 to solver tolerance."""
    model = YUKinematicPS(**PARAMS)
    integrator = PythonNumericalIntegrator(model)

    stress = np.zeros(3)
    state = _initial_state(model)
    strain_inc = np.array([2e-4, 0.0, 0.0])
    for _ in range(20):
        step = integrator.stress_update(strain_inc, stress, state)
        stress, state = step.stress, step.state
    assert step.is_plastic

    # f is quadratic with magnitude ~Y², so scale the tolerance accordingly.
    f = float(model.yield_function(model.make_state(**state)))
    assert abs(f) < 1e-6 * PARAMS["Y"] ** 2, f"f = {f:.6e}"


# ---------------------------------------------------------------------------
# Jacobian structure: rows the P formulation predicts in closed form
# ---------------------------------------------------------------------------

def test_jacobian_rows_match_closed_form():
    """The θ-free Jacobian rows follow directly from the quadratic P form.

    ∂f/∂σ = Pξ and the β row is linear in Δλ, so these blocks are exact
    closed forms.  The σ rows are deliberately excluded: C depends on eps_eq,
    which depends on σ through Δλ·√(2/3·g), so ∂R_σ/∂σ carries an extra ∂C/∂σ
    term beyond I + Δλ·C@P.
    """
    model = YUKinematicPS(**PARAMS)
    integrator = PythonNumericalIntegrator(model)
    P = model.P
    identity = np.eye(3)

    stress = np.zeros(3)
    state = _initial_state(model)
    strain_inc = np.array([2e-4, 0.0, 0.0])
    for _ in range(20):
        state_n = state
        step = integrator.stress_update(strain_inc, stress, state)
        stress, state = step.stress, step.state
    assert step.is_plastic

    dlambda = step.dlambda
    xi = model.dev(state["stress"]) - state["theta"] - state["beta"]
    blocks = JacobianChecker(model).compute(step, state_n)

    flow = P @ xi
    np.testing.assert_allclose(blocks.part["dlambda"]["stress"], flow, rtol=1e-9)
    np.testing.assert_allclose(blocks.part["dlambda"]["theta"], -flow, rtol=1e-9)
    np.testing.assert_allclose(blocks.part["dlambda"]["beta"], -flow, rtol=1e-9)

    k, b, Y = model.k, model.b, model.Y
    np.testing.assert_allclose(
        blocks.part["beta"]["stress"], -2.0 / 3.0 * k * b * dlambda * identity,
        rtol=1e-9, atol=1e-12,
    )
    np.testing.assert_allclose(
        blocks.part["beta"]["theta"], 2.0 / 3.0 * k * b * dlambda * identity,
        rtol=1e-9, atol=1e-12,
    )
    np.testing.assert_allclose(
        blocks.part["beta"]["beta"],
        (1.0 + 2.0 / 3.0 * k * b * dlambda + 2.0 / 3.0 * k * Y * dlambda) * identity,
        rtol=1e-9, atol=1e-12,
    )


def test_stress_row_matches_closed_form_when_stiffness_is_constant():
    """With Ea = E the E-degradation vanishes, so C is constant and the
    σ rows reduce to the closed forms I + Δλ·C@P and −Δλ·C@P."""
    params = dict(PARAMS, Ea=PARAMS["E"])
    model = YUKinematicPS(**params)
    integrator = PythonNumericalIntegrator(model)
    P = model.P

    stress = np.zeros(3)
    state = _initial_state(model)
    strain_inc = np.array([2e-4, 0.0, 0.0])
    for _ in range(20):
        state_n = state
        step = integrator.stress_update(strain_inc, stress, state)
        stress, state = step.stress, step.state
    assert step.is_plastic

    dlambda = step.dlambda
    C = model.elastic_stiffness(state)
    blocks = JacobianChecker(model).compute(step, state_n)

    np.testing.assert_allclose(
        blocks.part["stress"]["stress"], np.eye(3) + dlambda * (C @ P), rtol=1e-9
    )
    np.testing.assert_allclose(
        blocks.part["stress"]["theta"], -dlambda * (C @ P), rtol=1e-9
    )
    np.testing.assert_allclose(
        blocks.part["stress"]["beta"], -dlambda * (C @ P), rtol=1e-9
    )
