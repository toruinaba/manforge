"""Numerical equivalence harness for YUKinematicPS under the P-metric convention.

YUKinematicPS stores in-plane tensor components with the 33 component
identically zero and routes deviatoric contractions through PLANE_STRESS_P's
P metric.  It is solved purely by the framework autograd path.

Four independent axes, none of which needs a hand-derived Jacobian:

  1. Uniaxial equivalence — under uniaxial loading (σ22 = σ12 = 0) the model
     must reproduce YUKinematic1D and YUKinematic3D exactly.  This validates
     the convention physically rather than algebraically.
  1b. P-metric covariance — under multiaxial loading the internal variables
     must map onto the 3D model's through M(x) = dev₃D(x), which is what pins
     down where P belongs in the formulation.  Axis 1 is blind to this because
     P's ×2 shear weighting vanishes when σ12 = 0.
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
from manforge.simulation.integrator import (
    PythonAnalyticalIntegrator,
    PythonNumericalIntegrator,
)
from manforge.simulation.types import FieldHistory, FieldType
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
# Axis 1b: P-metric covariance against 3D under multiaxial loading
#
# The P metric implies a bijection between the PS storage convention and 3D
# deviators,  M(x) = dev₃D([x11, x22, 0, x12, 0, 0]),  satisfying xᵀPy = M(x):M(y).
# Every stress-like state (θ, β, q) is the M-preimage of its 3D counterpart, and
# because dev is linear the 3D evolution law keeps its form under M — which is
# why no P appears in R_theta / R_beta, only in the norms and in ∂f/∂σ.  Adding
# one would double the shear component and drop the 33 component, turning θ into
# a strain-like quantity.
#
# The uniaxial tests above cannot catch that: σ22 = σ12 = 0 makes P's ×2 shear
# weighting invisible.  Neither can the autograd-vs-closed-form Jacobian tests
# below, since both sides differentiate the same residual.
# ---------------------------------------------------------------------------

def _to_3d_deviator(x):
    """M: PS stored (11, 22, 12) → 3D deviator (6,)."""
    full = np.array([x[0], x[1], 0.0, x[2], 0.0, 0.0])
    p = (full[0] + full[1]) / 3.0
    return full - np.array([p, p, p, 0.0, 0.0, 0.0])


def _to_3d_state(state_ps):
    """σ33 ≡ 0 so σ maps by embedding; θ, β, q are stress-like and map by M."""
    sig = state_ps["stress"]
    return dict(
        stress=np.array([sig[0], sig[1], 0.0, sig[2], 0.0, 0.0]),
        theta=_to_3d_deviator(state_ps["theta"]),
        beta=_to_3d_deviator(state_ps["beta"]),
        q=_to_3d_deviator(state_ps["q"]),
        R=state_ps["R"], r=state_ps["r"],
        eps_eq=state_ps["eps_eq"], theta_max=state_ps["theta_max"],
    )


def _named(items):
    return {item.name: item.value for item in items}


# B − Y = 75, so these bracket the C_1 / C_2 branch of C_k.
@pytest.mark.parametrize("theta_max", [20.0, 110.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_state_residual_is_covariant_with_3d(theta_max, seed):
    """M(R_PS) == R_3D row by row, with Δλ_3D = (2/3)·Y·Δλ_PS.

    Evaluates both models at the *same* physical state — no solver, no driver —
    so a failure localises to a single residual row rather than to an
    accumulated trajectory difference.

    The σ row is excluded on purpose: it carries C, whose plane-stress form is
    a Schur condensation of the 3D one, so it does not transform by M.
    """
    model_ps = YUKinematicPS(**PARAMS)
    model_3d = YUKinematic3D(**PARAMS)
    Y = PARAMS["Y"]

    rng = np.random.default_rng(seed)
    theta = rng.normal(size=3) * 20.0
    beta = rng.normal(size=3) * 40.0
    q = rng.normal(size=3) * 10.0
    # Put ξ on the yield surface: ξᵀPξ = (2/3)Y² ⟺ f = 0.
    xi = rng.normal(size=3)
    xi *= np.sqrt(2.0 / 3.0) * Y / np.sqrt(
        model_ps.deviatoric_inner_product(xi, xi)
    )

    state_ps = dict(stress=xi + theta + beta, theta=theta, beta=beta, q=q,
                    R=40.0, r=15.0, eps_eq=4e-3, theta_max=theta_max)
    # Step-start state pulled back so the residuals are not trivially zero.
    state_n_ps = dict(state_ps, stress=state_ps["stress"] * 0.98,
                      theta=theta * 0.9, beta=beta * 0.95, q=q * 0.9,
                      R=38.0, r=14.0, eps_eq=4e-3 - 1e-4)

    dlambda_ps = 1e-5
    dlambda_3d = 2.0 / 3.0 * Y * dlambda_ps

    args_ps = (model_ps.make_state(**state_ps), model_ps.make_state(**state_n_ps))
    args_3d = (model_3d.make_state(**_to_3d_state(state_ps)),
               model_3d.make_state(**_to_3d_state(state_n_ps)))

    res_ps = _named(model_ps.state_residual(
        args_ps[0], dlambda_ps, args_ps[1], stress_trial=np.zeros(3)))
    res_3d = _named(model_3d.state_residual(
        args_3d[0], dlambda_3d, args_3d[1], stress_trial=np.zeros(6)))
    for key in ("theta", "beta"):
        np.testing.assert_allclose(
            _to_3d_deviator(res_ps[key]), res_3d[key],
            rtol=1e-10, atol=1e-12, err_msg=f"R_{key} is not M-covariant",
        )

    up_ps = _named(model_ps.update_state(
        dlambda_ps, args_ps[0], args_ps[1], stress_trial=np.zeros(3)))
    up_3d = _named(model_3d.update_state(
        dlambda_3d, args_3d[0], args_3d[1], stress_trial=np.zeros(6)))
    np.testing.assert_allclose(
        _to_3d_deviator(up_ps["q"]), up_3d["q"], rtol=1e-10, atol=1e-12
    )
    for key in ("R", "r", "eps_eq", "theta_max"):
        assert float(up_ps[key]) == pytest.approx(float(up_3d[key]), rel=1e-10), key

    # Δλ·n is the plastic strain increment, so the in-plane components must
    # agree even though the two Δλ differ.  This is where P legitimately enters.
    dep_ps = dlambda_ps * model_ps.flow(args_ps[0]).strain_like
    dep_3d = dlambda_3d * model_3d.flow(args_3d[0]).strain_like
    np.testing.assert_allclose(dep_ps, dep_3d[[0, 1, 3]], rtol=1e-10, atol=1e-16)


def _multiaxial_path(waypoints, n_per_segment):
    """Piecewise-linear (N, 3) in-plane strain history through the waypoints."""
    rows = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        for t in np.linspace(0.0, 1.0, n_per_segment, endpoint=False)[1:]:
            rows.append(a + (b - a) * t)
        rows.append(b)
    return FieldHistory(FieldType.STRAIN, "eps", np.array(rows))


_MONOTONIC = [np.zeros(3), np.array([6e-3, -3e-3, 4e-3])]
_NON_PROPORTIONAL = [
    np.zeros(3),
    np.array([1.2e-2, -6e-3, 8e-3]),
    np.array([-1.2e-2, 6e-3, 8e-3]),   # shear held while the axial pair reverses
    np.array([1.2e-2, -6e-3, -8e-3]),
]

_COLLECT = {
    "theta": FieldType.STRESS, "beta": FieldType.STRESS, "q": FieldType.STRESS,
    "R": FieldType.STRESS, "r": FieldType.STRESS, "theta_max": FieldType.STRESS,
    "eps_eq": FieldType.STRAIN,
}


def _run_ps_and_3d(history):
    """PS under full in-plane strain control vs 3D with σ33 = σ13 = σ23 = 0."""
    res_ps = StrainDriver(
        PythonNumericalIntegrator(YUKinematicPS(**PARAMS))
    ).run(history, collect_state=_COLLECT)
    res_3d = MixedDriver(
        PythonNumericalIntegrator(YUKinematic3D(**PARAMS)),
        prescribed_strain_idx=[0, 1, 3],
    ).run(history, collect_state=_COLLECT)
    return res_ps, res_3d


@pytest.mark.slow
@pytest.mark.parametrize("waypoints,n_per_segment", [
    (_MONOTONIC, 20),
    (_NON_PROPORTIONAL, 20),
])
def test_multiaxial_state_matches_3d(waypoints, n_per_segment):
    """Internal variables must track the 3D model through M over a full path.

    Unlike the pointwise residual test this exercises the whole PS route —
    elastic condensation, flow direction, NR — under biaxial + shear loading.
    """
    history = _multiaxial_path(waypoints, n_per_segment)
    res_ps, res_3d = _run_ps_and_3d(history)

    assert any(r.is_plastic for r in res_ps.step_results), "no plastic step"
    # θ̄ starts at 0 and ends past B − Y = 75, so both C_k branches are covered.
    assert res_ps.fields["theta_max"].data.max() > PARAMS["B"] - PARAMS["Y"]
    np.testing.assert_allclose(res_3d.stress[:, 2], 0.0, atol=1e-8)
    for col, idx_3d in enumerate([0, 1, 3]):
        np.testing.assert_allclose(
            res_ps.stress[:, col], res_3d.stress[:, idx_3d], rtol=1e-8, atol=1e-8
        )

    for key in ("theta", "beta", "q"):
        mapped = np.array([_to_3d_deviator(row) for row in res_ps.fields[key].data])
        np.testing.assert_allclose(
            mapped, res_3d.fields[key].data, rtol=1e-8, atol=1e-8,
            err_msg=f"{key} diverges from the 3D model under M",
        )
    for key in ("R", "r", "eps_eq", "theta_max"):
        np.testing.assert_allclose(
            res_ps.fields[key].data, res_3d.fields[key].data, rtol=1e-8, atol=1e-10,
            err_msg=key,
        )


@pytest.mark.slow
def test_multiaxial_tangent_matches_condensed_3d():
    """The PS tangent must equal the static condensation of the 3D one.

    D_PP − D_PF·D_FF⁻¹·D_FP over P = (11, 22, 12), F = (33, 13, 23) — the same
    reduction ``isotropic_C`` applies elastically, here checked on the plastic
    consistent tangent along a multiaxial path.
    """
    p_idx, f_idx = [0, 1, 3], [2, 4, 5]
    res_ps, res_3d = _run_ps_and_3d(_multiaxial_path(_MONOTONIC, 20))

    for i, (step_ps, step_3d) in enumerate(
        zip(res_ps.step_results, res_3d.step_results)
    ):
        d3 = step_3d.ddsdde
        condensed = d3[np.ix_(p_idx, p_idx)] - d3[np.ix_(p_idx, f_idx)] @ np.linalg.solve(
            d3[np.ix_(f_idx, f_idx)], d3[np.ix_(f_idx, p_idx)]
        )
        np.testing.assert_allclose(
            step_ps.ddsdde, condensed, rtol=1e-7, atol=1e-6, err_msg=f"step {i}"
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


# ---------------------------------------------------------------------------
# Hand-derived path: the calc_* blocks, the tangent, and the trajectory
#
# YUKinematicPS carries a hand-derived route (user_defined_return_mapping /
# user_defined_tangent) alongside the autograd one.  The autograd path is the
# reference: it is itself pinned against finite differences above.
# ---------------------------------------------------------------------------

# (row, col, callable) for every block of the 10x10 PS Jacobian.
# Blocks that need theta_max take state_n and read it from there themselves,
# matching calc_residual — so these are called with the converged state as-is.
_BLOCKS = [
    ("dlambda", "stress", lambda m, s, sn, dl: m.calc_fy_fs(s)),
    ("dlambda", "theta", lambda m, s, sn, dl: m.calc_fy_ft(s)),
    ("dlambda", "beta", lambda m, s, sn, dl: m.calc_fy_fb(s)),
    ("dlambda", "dlambda", lambda m, s, sn, dl: m.calc_fy_fl(s)),
    ("stress", "stress", lambda m, s, sn, dl: m.calc_fe_fs(s, dl, sn)),
    ("stress", "theta", lambda m, s, sn, dl: m.calc_fe_ft(s, dl, sn)),
    ("stress", "beta", lambda m, s, sn, dl: m.calc_fe_fb(s, dl, sn)),
    ("stress", "dlambda", lambda m, s, sn, dl: m.calc_fe_fl(s, dl, sn)),
    ("theta", "stress", lambda m, s, sn, dl: m.calc_ft_fs(s, dl, sn)),
    ("theta", "theta", lambda m, s, sn, dl: m.calc_ft_ft(s, dl, sn)),
    ("theta", "beta", lambda m, s, sn, dl: m.calc_ft_fb(s, dl, sn)),
    ("theta", "dlambda", lambda m, s, sn, dl: m.calc_ft_fl(s, dl, sn)),
    ("beta", "stress", lambda m, s, sn, dl: m.calc_fb_fs(s, dl)),
    ("beta", "theta", lambda m, s, sn, dl: m.calc_fb_ft(s, dl)),
    ("beta", "beta", lambda m, s, sn, dl: m.calc_fb_fb(s, dl)),
    ("beta", "dlambda", lambda m, s, sn, dl: m.calc_fb_fl(s, dl)),
]


def _plastic_steps(model, peaks, n_per_segment, driven_idx=0):
    """Yield (state_new, state_n, step) for every plastic step of a history."""
    driver = MixedDriver(
        PythonNumericalIntegrator(model), prescribed_strain_idx=[driven_idx]
    )
    result = driver.run(
        FieldHistory.cyclic_strain(peaks, n_per_segment=n_per_segment, ntens=1)
    )
    for i in range(1, len(result.step_results)):
        step = result.step_results[i]
        if step.is_plastic:
            yield dict(step.state), dict(result.step_results[i - 1].state), step


@pytest.mark.parametrize("peaks,driven_idx", [
    ([5e-3], 0),
    ([1e-2], 2),
    pytest.param([0.05, -0.05, 0.05], 0, marks=pytest.mark.slow),
    pytest.param([0.05, -0.05, 0.05], 2, marks=pytest.mark.slow),
])
def test_calc_blocks_match_autograd(peaks, driven_idx):
    """Every hand-derived Jacobian block must match the autograd Jacobian.

    Errors are collected across all blocks and all steps rather than aborting
    on the first mismatch — a single wrong block otherwise hides behind
    whichever one happens to fail first.

    Driving component 2 (shear) as well as 0 matters here: under uniaxial
    loading a ×2 shear-convention error in a P-metric term stays invisible.
    """
    model = YUKinematicPS(**PARAMS)
    checker = JacobianChecker(model)
    worst = {}
    n_steps = 0

    for state, state_n, step in _plastic_steps(model, peaks, 25, driven_idx):
        n_steps += 1
        blocks = checker.compute(step, state_n)
        for row, col, fn in _BLOCKS:
            ref = np.asarray(blocks.part[row][col], dtype=float)
            got = np.asarray(fn(model, state, state_n, step.dlambda), dtype=float)
            err = np.abs(got - ref).max() / max(np.abs(ref).max(), 1.0)
            key = f"{row}::{col}"
            worst[key] = max(worst.get(key, 0.0), err)

    assert n_steps > 0, "history produced no plastic steps"
    bad = {k: v for k, v in worst.items() if v > 1e-8}
    assert not bad, (
        f"{len(bad)}/{len(_BLOCKS)} blocks disagree with autograd over "
        f"{n_steps} plastic steps: "
        + ", ".join(f"{k}={v:.3e}" for k, v in sorted(bad.items(), key=lambda kv: -kv[1]))
    )


def test_calc_ft_fl_gates_on_stagnation():
    """calc_ft_fl's da/dΔλ must vanish while the stagnation surface is inactive.

    R is frozen unless the stagnation surface is active, so differentiating its
    evolution law unconditionally is wrong on exactly those steps.  Without the
    gate this reached 18% error, and only on steps where R did not grow — which
    a whole-Jacobian check reports as a generic ``theta::dlambda`` mismatch.
    """
    model = YUKinematicPS(**PARAMS)
    checker = JacobianChecker(model)
    n_inactive = 0

    for state, state_n, step in _plastic_steps(model, [0.05, -0.05, 0.05], 25):
        if abs(float(state["R"]) - float(state_n["R"])) > 1e-15:
            continue
        n_inactive += 1
        ref = np.asarray(checker.compute(step, state_n).part["theta"]["dlambda"], float)
        got = np.asarray(model.calc_ft_fl(state, step.dlambda, state_n), float)
        np.testing.assert_allclose(
            got, ref, rtol=1e-8, atol=1e-10,
            err_msg="calc_ft_fl disagrees on a step where R is frozen",
        )

    assert n_inactive > 0, "history never held R frozen -- test proves nothing"


@pytest.mark.parametrize("n_preload", [3, 10, 20])
def test_user_defined_tangent_vs_finite_difference(n_preload):
    """user_defined_tangent must match central differences of dσ/dΔε.

    C is evaluated at eps_eq, so the rhs of the consistent-tangent solve must
    use C(state_n) — σ_trial is built with the step-start stiffness.  Using
    C(state_new) here left a 0.5–1.2% error that no Jacobian-block check sees,
    because it enters after the Jacobian.
    """
    model = YUKinematicPS(**PARAMS)
    numerical = PythonNumericalIntegrator(model)
    analytical = PythonAnalyticalIntegrator(model)

    stress = np.zeros(3)
    state = _initial_state(model)
    strain_inc = np.array([1.0e-3, -0.3e-3, 0.0])
    for _ in range(n_preload):
        step = numerical.stress_update(strain_inc, stress, state)
        stress, state = np.asarray(step.stress), dict(step.state)
    assert numerical.stress_update(strain_inc, stress, state).is_plastic

    result = TangentChecker(analytical).check(stress, state, strain_inc)
    assert result.passed, f"max_rel_err={result.max_rel_err:.3e}"


@pytest.mark.slow
def test_user_defined_trajectory_matches_autograd():
    """The hand-derived return mapping must track the autograd one step for step.

    Covers the residual and the mu/stagnation update, which the block checks
    above do not reach.  Both routes gate on the same smooth_heaviside, so this
    is what establishes that nothing else drifts across steps.
    """
    model = YUKinematicPS(**PARAMS)
    history = FieldHistory.cyclic_strain([0.05, -0.05, 0.05], n_per_segment=25, ntens=1)
    numerical = MixedDriver(
        PythonNumericalIntegrator(model), prescribed_strain_idx=[0]
    ).run(history)
    analytical = MixedDriver(
        PythonAnalyticalIntegrator(model), prescribed_strain_idx=[0]
    ).run(history)

    np.testing.assert_allclose(
        analytical.stress, numerical.stress, rtol=1e-6, atol=1e-6
    )
    for key in ("theta", "beta", "R", "q", "r", "eps_eq", "theta_max"):
        got = np.array([np.asarray(s.state[key]) for s in analytical.step_results])
        ref = np.array([np.asarray(s.state[key]) for s in numerical.step_results])
        np.testing.assert_allclose(
            got, ref, rtol=1e-6, atol=1e-6, err_msg=f"state[{key!r}] diverged"
        )
