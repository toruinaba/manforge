"""Permanent regression test: YU calc_jacobian vs full-perturbation FD.

Verifies that the analytical Jacobian (calc_jacobian) used in the internal
Newton-Raphson solver is consistent with the true total derivative of
calc_residual — including the stagnation-surface state recomputation that
happens each NR iteration.

Regression targets (bugs fixed 2026-06-01):
  Bug A: dRstress/d{theta,beta} had spurious I_dev projection (rel-err ~0.73)
  Bug B: _prepare_Rtheta used hard C_k branch vs smooth_heaviside in residual
         and read theta_max from state_new instead of state_n (rel-err ~9)

Both bugs are caught by the J_analytic vs FD_full < 1e-6 check below.
"""

import numpy as np
import pytest
from copy import deepcopy

from manforge.models import YUKinematic3D
from manforge.simulation.integrator import PythonNumericalIntegrator
from manforge.utils.smooth import smooth_sqrt, smooth_heaviside, smooth_max

from .conftest import PARAMS


# ---------------------------------------------------------------------------
# Stagnation-surface recomputation (mirrors user_defined_return_mapping loop)
# ---------------------------------------------------------------------------

def _recompute_stagnation(model, state_n, stress, theta, beta, dlambda):
    s = 1.0 / (1.0 + model.k * dlambda)
    d_beta = beta - state_n["beta"]
    g_xi = beta - state_n["q"]
    stag_norm = model.vonmises_norm(g_xi)
    g_stag = stag_norm - state_n["r"]
    g_flag = 1.0 if g_stag > 0.0 else 0.0

    Gn = model.deviatoric_inner_product(g_xi, g_xi)
    Fn = model.deviatoric_inner_product(g_xi, d_beta)
    mu = 0.0
    r_n = state_n["r"]
    for _ in range(10):
        H_mu = smooth_sqrt(r_n * r_n + 6 * model.h * Fn / (1.0 + mu))
        F_mu = (3 * Gn - r_n * (r_n + H_mu) * (1.0 + mu) ** 2
                - 3 * model.h * Fn * (1.0 + mu))
        if F_mu < 1.0e-16:
            break
        F_mu_p = (3 * model.h * Fn / H_mu * (r_n - H_mu)
                  - 2 * r_n * (1.0 + mu) * (r_n + H_mu))
        mu -= F_mu / F_mu_p

    delta_q = mu * g_xi / (1.0 + mu)
    delta_r = 0.5 * (r_n + smooth_sqrt(r_n * r_n + 6 * model.h * Fn / (1.0 + mu))) - r_n
    delta_R = s * (state_n["R"] + model.k * model.Rsat * dlambda) - state_n["R"]
    theta_norm = model.vonmises_norm(theta)

    return {
        "q":         state_n["q"] + g_flag * delta_q,
        "r":         float(state_n["r"] + g_flag * delta_r),
        "R":         float(state_n["R"] + g_flag * delta_R),
        "eps_eq":    float(state_n["eps_eq"] + dlambda),
        "theta_max": float(smooth_max(state_n["theta_max"], theta_norm)),
    }


def _residual_full(model, x, state_n, stress_trial):
    """calc_residual with stagnation state fully re-computed for each x."""
    stress = x[0:6]
    theta  = x[7:13]
    beta   = x[13:19]
    dlambda = float(x[6])
    stag = _recompute_stagnation(model, state_n, stress, theta, beta, dlambda)
    state_new = deepcopy(state_n)
    state_new["stress"] = stress.copy()
    state_new["theta"]  = theta.copy()
    state_new["beta"]   = beta.copy()
    state_new.update(stag)
    return np.asarray(model.calc_residual(state_new, state_n, stress_trial, dlambda))


def _fd_jacobian(residual_fn, x0, eps=1e-6):
    n = len(x0)
    J = np.zeros((len(residual_fn(x0)), n))
    for j in range(n):
        xp = x0.copy(); xp[j] += eps
        xm = x0.copy(); xm[j] -= eps
        J[:, j] = (residual_fn(xp) - residual_fn(xm)) / (2.0 * eps)
    return J


# ---------------------------------------------------------------------------
# State collection
# ---------------------------------------------------------------------------

def _collect_plastic_steps(model, history):
    integrator = PythonNumericalIntegrator(model)
    stress_n = np.zeros(model.ntens)
    state_n  = model.initial_state()
    eps_prev = np.zeros(model.ntens)
    steps = []
    for eps in history:
        deps = eps - eps_prev
        eps_prev = eps.copy()
        res = integrator.stress_update(np.array(deps), np.array(stress_n), state_n)
        if res.is_plastic:
            rm = res.return_mapping
            state_c = rm.state
            dlambda = float(rm.dlambda)
            x_conv = np.concatenate([
                np.asarray(state_c["stress"]),
                [dlambda],
                np.asarray(state_c["theta"]),
                np.asarray(state_c["beta"]),
            ])
            steps.append({
                "state_n":      state_n,
                "stress_trial": np.asarray(res.stress_trial),
                "x_conv":       x_conv,
                "dlambda":      dlambda,
            })
        stress_n = np.asarray(res.stress)
        state_n  = res.state
    return steps


# ---------------------------------------------------------------------------
# Helpers for building state_new from converged x
# ---------------------------------------------------------------------------

def _build_state_new(x, state_n, model):
    stress  = x[0:6]
    theta   = x[7:13]
    beta    = x[13:19]
    dlambda = float(x[6])
    stag = _recompute_stagnation(model, state_n, stress, theta, beta, dlambda)
    state_new = deepcopy(state_n)
    state_new["stress"] = stress.copy()
    state_new["theta"]  = theta.copy()
    state_new["beta"]   = beta.copy()
    state_new.update(stag)
    return state_new, dlambda


# Block layout (row = residual, col = unknown)
_ROW = [("Rstress", slice(0,6)),  ("Ryield", slice(6,7)),
        ("Rtheta",  slice(7,13)), ("Rbeta",  slice(13,19))]
_COL = [("stress",  slice(0,6)),  ("dlambda", slice(6,7)),
        ("theta",   slice(7,13)), ("beta",    slice(13,19))]

TOL = 1e-6


def _check_jacobian(model, steps):
    for step in steps:
        state_n      = step["state_n"]
        stress_trial = step["stress_trial"]
        x0           = step["x_conv"]
        dlambda      = step["dlambda"]

        state_new, _ = _build_state_new(x0, state_n, model)
        J_analytic = model.calc_jacobian(state_new, state_n, stress_trial, dlambda)

        def res_full(x):
            return _residual_full(model, x, state_n, stress_trial)

        J_full = _fd_jacobian(res_full, x0)

        rel = np.abs(J_analytic - J_full) / (np.abs(J_analytic) + 1.0)
        for rname, rslice in _ROW:
            for cname, cslice in _COL:
                block_err = float(np.max(rel[rslice, cslice]))
                assert block_err < TOL, (
                    f"Jacobian mismatch at block [{rname}][{cname}]: "
                    f"rel-err={block_err:.3e} >= {TOL:.0e}\n"
                    f"  dlambda={dlambda:.4e}, theta_max_n={float(state_n['theta_max']):.2f}"
                )


def _cyclic_strain(amplitudes, n_per_segment):
    segments = []
    prev = 0.0
    for amp in amplitudes:
        t = np.linspace(prev, amp, n_per_segment + 1)[1:]
        prev = amp
        segments.append(t)
    eps1d = np.concatenate(segments)
    data = np.zeros((len(eps1d), 6))
    data[:, 0] = eps1d
    return data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_jacobian_fd_uniaxial_cyclic():
    """calc_jacobian consistent with full-perturbation FD: uniaxial cyclic (±5%)."""
    model = YUKinematic3D(**PARAMS)
    history = _cyclic_strain([0.05, -0.05, 0.05, -0.05], 50)
    steps = _collect_plastic_steps(model, history)
    assert steps, "no plastic steps collected"
    _check_jacobian(model, steps)


@pytest.mark.slow
def test_jacobian_fd_branch_c2():
    """calc_jacobian consistent with FD at theta_max transition (±2%, C1/C2 branch)."""
    model = YUKinematic3D(**PARAMS)
    history = _cyclic_strain([0.02, -0.02, 0.02, -0.02], 30)
    steps = _collect_plastic_steps(model, history)
    assert steps, "no plastic steps collected"
    _check_jacobian(model, steps)
