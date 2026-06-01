"""Path A benchmark: YU analytical return mapping vs framework Newton-Raphson.

Verifies that PythonAnalyticalIntegrator (closed-form) and
PythonNumericalIntegrator (autodiff NR) produce identical results for
YUKinematic3D across a variety of strain histories.

YUKinematic1D has no analytical path (user_defined_return_mapping is 3D-only),
so 1D coverage is smoke-only (autograd convergence + state monotonicity).

Tolerance policy (evidence from _diagnose_jacobian.py, 2026-06-01):

  stress  : atol=5e-3
    Remaining structural difference between the two paths: the analytical
    outer loop uses a hard g_flag=1/0 branch for the stagnation-surface
    state update (user_defined_return_mapping), while the autograd path
    uses smooth_heaviside throughout (update_state). For multiaxial loading
    this produces ~2.7e-3 divergence in accumulated stress.

  state   : per-key relative error (|Δ| / (|v| + 1))
    R, q, r : atol=1e-2  — hard g_flag vs smooth_heaviside in stagnation
              update; structural O(1e-2).
    others  : atol=1e-4  — eps_eq, theta, beta, theta_max stay tight.

  tangent : max_rel_err < 2e-2
    After fixing calc_jacobian (2026-06-01 bugs A+B), the structural
    mismatch is now O(1e-3)–O(1e-2). Observed worst case: 1.4e-2 for
    multiaxial loading. Tolerance 2e-2 provides a 1.5× margin.

  NR iterations: diff <= 5
    For most scenarios the counts match exactly (diff=0). Multiaxial
    loading can differ by up to 4 due to the hard/smooth g_flag difference
    leading to slightly different convergence paths.
"""

import numpy as np
import pytest

from manforge.simulation.integrator import (
    PythonAnalyticalIntegrator,
    PythonNumericalIntegrator,
)
from manforge.simulation.driver import MixedDriver, StrainDriver
from manforge.simulation.types import FieldHistory, FieldType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_driver(integrator, ntens):
    if ntens == 1:
        return StrainDriver(integrator)
    return MixedDriver(integrator, prescribed_strain_idx=[0])


def _wrap_history(model, history_data):
    if model.ntens > 1:
        # MixedDriver expects (N, len(P)) — P=[0] so extract column 0
        return FieldHistory(FieldType.STRAIN, "Strain", history_data[:, [0]])
    return FieldHistory(FieldType.STRAIN, "Strain", history_data)


def _run_and_compare(model, history):
    """Step through history with both integrators; return max errors.

    Both integrators receive the same (deps, stress_n, state_n) at each step,
    and state_n is advanced from the numerical (ground-truth) result — the
    j2_isotropic pattern. This prevents smooth/if-branch divergence from
    accumulating across steps.

    For 3D scenarios the full 6-component strain is used (uniaxial strain,
    not uniaxial stress), matching the j2_isotropic benchmark convention.
    """
    import autograd.numpy as anp

    numerical = PythonNumericalIntegrator(model)
    analytical = PythonAnalyticalIntegrator(model)

    stress_n = np.zeros(model.ntens)
    state_n = model.initial_state()
    eps_prev = np.zeros(model.ntens)

    max_stress_err = 0.0
    max_iter_diff = 0
    max_tangent_err = 0.0
    # state keys present in both paths; R/q/r track stagnation-surface state
    state_keys = ("theta", "beta", "R", "q", "r", "eps_eq", "theta_max")
    max_state_err = {k: 0.0 for k in state_keys}

    for eps in history:
        deps = eps - eps_prev
        eps_prev = eps.copy()

        r_num = numerical.stress_update(anp.array(deps), anp.array(stress_n), state_n)
        r_an = analytical.stress_update(anp.array(deps), anp.array(stress_n), state_n)

        s_num = np.asarray(r_num.stress)
        s_an = np.asarray(r_an.stress)
        max_stress_err = max(max_stress_err, float(np.max(np.abs(s_an - s_num))))

        for k in state_keys:
            if k in r_num.state:
                v_n = np.asarray(r_num.state[k])
                v_a = np.asarray(r_an.state[k])
                rel = np.abs(v_a - v_n) / (np.abs(v_n) + 1.0)
                max_state_err[k] = max(max_state_err[k], float(np.max(rel)))

        if r_num.is_plastic:
            max_iter_diff = max(
                max_iter_diff,
                abs((r_an.n_iterations or 0) - (r_num.n_iterations or 0)),
            )

            if r_an.is_plastic:
                D_n = np.asarray(r_num.ddsdde)
                D_a = np.asarray(r_an.ddsdde)
                rel = np.abs(D_a - D_n) / (np.abs(D_n) + 1.0)
                max_tangent_err = max(max_tangent_err, float(np.max(rel)))

        # advance from numerical ground truth
        stress_n = np.asarray(r_num.stress)
        state_n = r_num.state

    return {
        "stress": max_stress_err,
        "state": max_state_err,
        "iter_diff": max_iter_diff,
        "tangent": max_tangent_err,
    }


# ---------------------------------------------------------------------------
# B-S1 + B-S2: smoke — autograd converges and state is monotone
# ---------------------------------------------------------------------------

def test_smoke_convergence_and_monotonicity(yu_smoke_scenario):
    """Autograd integrator converges every step; eps_eq and theta_max are
    monotone non-decreasing across the full history."""
    model, history = yu_smoke_scenario
    drv = _make_driver(PythonNumericalIntegrator(model), model.ntens)
    load = _wrap_history(model, history)
    res = drv.run(load)

    assert all(s.converged for s in res.step_results), "non-converged step found"

    eps_eq = np.array([s.state["eps_eq"] for s in res.step_results])
    theta_max = np.array([s.state["theta_max"] for s in res.step_results])
    assert np.all(np.diff(eps_eq) >= -1e-12), "eps_eq decreased"
    assert np.all(np.diff(theta_max) >= -1e-12), "theta_max decreased"


# ---------------------------------------------------------------------------
# B-A1 to B-A5: analytical vs autograd comparison (3D only)
# ---------------------------------------------------------------------------

def test_analytical_matches_numerical(yu_3d_scenario):
    """Analytical and autograd integrators agree over the full strain history.

    Tolerances are set from _diagnose_jacobian.py empirical data (2026-06-01).

    B-A1: max stress error < 5e-3
          (multiaxial reaches 2.7e-3; hard g_flag vs smooth_heaviside in the
          outer stagnation-surface state update drives a structural divergence)
    B-A2: per-state-key relative error
          R/q/r < 1e-2, theta < 5e-4, others < 1e-4
    B-A4: ddsdde max_rel_err < 2e-2 (improved after Jacobian bug fix)
    B-A5: NR iteration count difference <= 5 (multiaxial can differ)
    """
    model, history = yu_3d_scenario
    errs = _run_and_compare(model, history)

    # B-A1: stress trajectory
    assert errs["stress"] < 5e-3, f"stress err = {errs['stress']:.3e}"

    # B-A2: state variables (relative error)
    stagnation_keys = {"R", "q", "r"}
    # theta can reach ~1.2e-4 for multiaxial after the Jacobian fix (hard/smooth g_flag)
    loose_keys = {"theta"}
    for k, v in errs["state"].items():
        if k in stagnation_keys:
            atol = 1e-2
        elif k in loose_keys:
            atol = 5e-4
        else:
            atol = 1e-4
        assert v < atol, f"state[{k!r}] rel_err = {v:.3e} (atol={atol:.0e})"

    # B-A4: ddsdde — structural difference from hard/smooth g_flag after Jacobian fix
    assert errs["tangent"] < 2e-2, f"ddsdde rel err = {errs['tangent']:.3e}"

    # B-A5: iteration count — exact match for most scenarios; multiaxial can differ
    assert errs["iter_diff"] <= 5, f"NR iter count diff = {errs['iter_diff']}"
