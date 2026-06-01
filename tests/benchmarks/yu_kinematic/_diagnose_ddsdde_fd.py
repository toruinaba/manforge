"""Stage 1 diagnostic: compare calc_ddsdde variants vs FD truth (dσ/dε).

Computes the consistent tangent (ddsdde) three ways and compares each against
the finite-difference ground truth (central differences of stress_update).

Variants:
  (a) no_correction — fe6f449 equivalent: chain-rule block skipped entirely.
  (b) head           — HEAD current state: as returned by result.ddsdde.
  (c) fixed          — A1 coefficient 25→12.5 and A2 factor 1.5 added.

Run:
    uv run python tests/benchmarks/yu_kinematic/_diagnose_ddsdde_fd.py 2>&1 | tee /tmp/yu_diag_ddsdde.txt
"""

import math
import sys

import numpy as np
import autograd.numpy as anp

from manforge.models import YUKinematic3D
from manforge.simulation.integrator import PythonAnalyticalIntegrator
from manforge.simulation.types import FieldHistory
from manforge.utils.smooth import smooth_heaviside

PARAMS = dict(
    E=206_000, nu=0.3, Y=360.0, C_1=2000.0, C_2=200.0,
    B=435.0, Rsat=255.0, k=26.0, b=66.0, h=0.4, Ea=159_000, xi=61.0,
)
B_MINUS_Y = PARAMS["B"] - PARAMS["Y"]  # 75.0 MPa
FD_H = 1e-5  # central-difference step (verified stable for YU)


def _build_scenarios():
    mono = np.zeros((50, 6))
    mono[:, 0] = np.linspace(0.0, 5e-3, 50)

    rev = FieldHistory.cyclic_strain([0.01, -0.01], n_per_segment=30, ntens=6).data

    cyc = FieldHistory.cyclic_strain([0.05, -0.05, 0.05, -0.05], n_per_segment=50, ntens=6).data

    large = FieldHistory.cyclic_strain([0.04, -0.04], n_per_segment=30, ntens=6).data
    small = FieldHistory.cyclic_strain([0.005, -0.005, 0.005], n_per_segment=15, ntens=6).data
    stag = np.vstack([large, small + large[-1]])

    shear_raw = FieldHistory.cyclic_strain([5e-3, -5e-3, 5e-3], n_per_segment=30, ntens=6).data
    shear = np.zeros_like(shear_raw); shear[:, 3] = shear_raw[:, 0]

    large_shear = np.zeros((60, 6))
    large_shear[:, 3] = np.linspace(0.0, 0.04, 60)

    return {
        "uniaxial_monotonic":  mono,
        "load_reversal":       rev,
        "uniaxial_cyclic_big": cyc,
        "stagnation_crossing": stag,
        "pure_shear":          shear,
        "pure_shear_large":    large_shear,
    }


def _fd_ddsdde(integrator, deps, stress_n, state_n, h=FD_H):
    """Central-difference dσ/dε at (stress_n, state_n) for strain increment deps."""
    ntens = len(deps)
    D_fd = np.zeros((ntens, ntens))
    for j in range(ntens):
        eps_p = deps.copy(); eps_p[j] += h
        eps_m = deps.copy(); eps_m[j] -= h
        rp = integrator.stress_update(eps_p, stress_n, state_n)
        rm = integrator.stress_update(eps_m, stress_n, state_n)
        D_fd[:, j] = (np.asarray(rp.stress) - np.asarray(rm.stress)) / (2.0 * h)
    return D_fd


def _calc_ddsdde_variant(model, state_new, state_n, stress_trial, dlambda, variant):
    """Re-compute calc_ddsdde with a specific variant of the chain-rule block.

    This is a local copy of calc_ddsdde (yu_kinematic.py:481-540) with the
    correction block (L506-531) controlled by `variant`.

    variant:
        'no_correction' — skip the chain-rule block entirely (fe6f449 equivalent)
        'head'          — apply block with coefficient 25.0 (HEAD current)
        'fixed'         — apply block with A1 fix (12.5) and A2 fix (1.5*)
    """
    C = model.elastic_stiffness(state_new)
    C_inv = anp.linalg.inv(C)
    xi = model.dev(state_new["stress"]) - state_new["theta"] - state_new["beta"]
    g_xi   = state_new["beta"] - state_n["q"]
    g_stag = model.vonmises_norm(g_xi) - state_n["r"]
    g_flag = float(smooth_heaviside(g_stag))
    theta  = state_new["theta"]
    t_max  = state_n["theta_max"]
    R_new  = state_new["R"]
    R_n    = state_n["R"]

    Rs_s = C_inv @ model.dRstress_dstress(C, xi, dlambda)
    Rs_b = C_inv @ model.dRstress_dbeta(C, xi, dlambda)
    Rs_t = C_inv @ model.dRstress_dtheta(C, xi, dlambda)
    Rs_l = C_inv @ model.dRstress_dlambda(C, xi, state_new["eps_eq"], dlambda)
    Rs = np.hstack((Rs_s, Rs_l[:, np.newaxis], Rs_t, Rs_b))

    Rb_s = model.dRbeta_dstress(dlambda)
    Rb_b = model.dRbeta_dbeta(dlambda)
    Rb_t = model.dRbeta_dtheta(dlambda)
    Rb_l = model.dRbeta_dlambda(xi, state_new["beta"], dlambda)
    Rb = np.hstack((Rb_s, Rb_l[:, np.newaxis], Rb_t, Rb_b))

    Rt_s = model.dRtheta_dstress(theta, t_max, R_new, R_n, dlambda, g_flag)
    Rt_b = model.dRtheta_dbeta(theta, t_max, R_new, R_n, dlambda, g_flag)
    Rt_t = model.dRtheta_dtheta(theta, t_max, R_new, R_n, dlambda, g_flag)
    Rt_l = model.dRtheta_dlambda(xi, theta, t_max, R_new, R_n, dlambda, g_flag)

    if variant != "no_correction":
        stag_norm_f = float(model.vonmises_norm(g_xi))
        if stag_norm_f > 1e-15:
            g_stag_f = float(g_stag)
            # A1: coefficient for smooth_heaviside'
            # HEAD uses 25.0;  fixed uses 12.5  (correct: 0.5*tanh*25 → deriv = 12.5*sech²)
            coeff = 12.5 if variant == "fixed" else 25.0
            t_val = math.tanh(coeff / 12.5 * 12.5 * g_stag_f)  # tanh(25*x) either way
            t_val = math.tanh(25.0 * g_stag_f)
            dg_flag_dgstag = coeff * (1.0 - t_val * t_val)
            if abs(dg_flag_dgstag) > 1e-15:
                s_val = 1.0 / (1.0 + model.k * float(dlambda))
                delta_R = s_val * (float(R_n) + model.k * model.Rsat * float(dlambda)) - float(R_n)
                T_vec = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
                g_xi_arr = np.asarray(g_xi)
                # A2: missing 1.5 factor in da_dbeta for vonmises_norm derivative
                # HEAD: no factor;  fixed: 1.5 *
                a2_factor = 1.5 if variant == "fixed" else 1.0
                da_dbeta = (delta_R * dg_flag_dgstag * a2_factor
                            * T_vec * g_xi_arr / stag_norm_f)
                theta_bar_v, _, C_k_v, _, a_v, _ = model._prepare_Rtheta(
                    theta, t_max, R_new, R_n, dlambda, g_flag)
                if float(theta_bar_v) > 1e-14 and float(a_v) > 1e-14:
                    dRt_da = -(C_k_v / model.Y * xi
                               - C_k_v * np.sqrt(1.0 / (a_v * float(theta_bar_v))) / 2.0 * theta) * dlambda
                else:
                    dRt_da = -C_k_v / model.Y * xi * dlambda
                Rt_b = Rt_b + np.outer(dRt_da, da_dbeta)

    Rt = np.hstack((Rt_s, Rt_l[:, np.newaxis], Rt_t, Rt_b))

    Rl_s = model.dRyield_dstress(xi)
    Rl_b = model.dRyield_dbeta(xi)
    Rl_t = model.dRyield_dtheta(xi)
    Rl_l = model.dRyield_dlambda()
    Rl = np.hstack((Rl_s, Rl_l, Rl_t, Rl_b))

    jac = np.vstack((Rs, Rl.reshape(1, -1), Rt, Rb))
    jac_inv = anp.linalg.inv(jac)
    return np.array(jac_inv[:6, :6])


def _err(D_candidate, D_ref):
    return float(np.max(np.abs(D_candidate - D_ref) / (np.abs(D_ref) + 1.0)))


def _worst_block(D_candidate, D_ref):
    ratio = np.abs(D_candidate - D_ref) / (np.abs(D_ref) + 1.0)
    idx = np.unravel_index(np.argmax(ratio), ratio.shape)
    return idx, float(ratio[idx])


def _diagnose_scenario(name, history, *, top_k=12):
    model     = YUKinematic3D(**PARAMS)
    integrator = PythonAnalyticalIntegrator(model)

    stress_n  = np.zeros(6)
    state_n   = model.initial_state()
    eps_prev  = np.zeros(6)

    rows = []
    for i, eps in enumerate(history):
        deps      = eps - eps_prev
        eps_prev  = eps.copy()
        result    = integrator.stress_update(deps, stress_n, state_n)

        if result.is_plastic:
            rm     = result.return_mapping
            state_new   = rm.state
            dlambda     = float(rm.dlambda)
            stress_trial = result.stress_trial

            D_fd  = _fd_ddsdde(integrator, deps, stress_n, state_n)
            D_a   = _calc_ddsdde_variant(model, state_new, state_n, stress_trial, dlambda, "no_correction")
            D_b   = np.asarray(result.ddsdde)   # HEAD (sanity: should match variant b)
            D_b2  = _calc_ddsdde_variant(model, state_new, state_n, stress_trial, dlambda, "head")
            D_c   = _calc_ddsdde_variant(model, state_new, state_n, stress_trial, dlambda, "fixed")

            g_xi   = np.asarray(state_new["beta"]) - np.asarray(state_n["q"])
            g_stag = float(model.vonmises_norm(g_xi)) - float(state_n["r"])
            t_max  = float(state_n["theta_max"])

            rows.append({
                "i":          i,
                "g_stag":     g_stag,
                "theta_max":  t_max,
                "dist_to_75": t_max - B_MINUS_Y,
                "err_a":      _err(D_a,  D_fd),
                "err_b":      _err(D_b,  D_fd),
                "err_b2":     _err(D_b2, D_fd),   # sanity vs result.ddsdde
                "err_c":      _err(D_c,  D_fd),
                "worst_a":    _worst_block(D_a,  D_fd),
                "worst_c":    _worst_block(D_c,  D_fd),
                "dlambda":    dlambda,
            })

        stress_n = np.asarray(result.stress)
        state_n  = result.state

    print(f"\n{'='*75}")
    print(f"Scenario: {name}  ({len(rows)} plastic steps)")
    print(f"  B-Y = {B_MINUS_Y:.1f} MPa  FD_H={FD_H:.0e}")
    print(f"{'='*75}")

    if not rows:
        print("  (no plastic steps)")
        return

    for key, label in [("err_a", "no_correction (fe6f449)"),
                       ("err_b", "head (result.ddsdde)"),
                       ("err_b2","head variant local"),
                       ("err_c", "fixed (A1+A2)")]:
        vals = [r[key] for r in rows]
        print(f"  max {label:28s}: {max(vals):.3e}  mean: {np.mean(vals):.3e}")

    sorted_rows = sorted(rows, key=lambda r: -r["err_b"])
    print(f"\n  Top {top_k} plastic steps by err_b (HEAD vs FD):")
    hdr = (f"  {'step':>4}  {'g_stag':>9}  {'dist_75':>8}  "
           f"{'err_a':>9}  {'err_b':>9}  {'err_c':>9}  "
           f"{'best':>6}  worst_block(a)   worst_block(c)")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in sorted_rows[:top_k]:
        # which variant wins (lowest err)
        best = min([("a", r["err_a"]), ("b", r["err_b"]), ("c", r["err_c"])],
                   key=lambda x: x[1])[0]
        wa_idx, wa_val = r["worst_a"]
        wc_idx, wc_val = r["worst_c"]
        print(
            f"  {r['i']:4d}  {r['g_stag']:+9.4f}  {r['dist_to_75']:+8.3f}  "
            f"{r['err_a']:9.3e}  {r['err_b']:9.3e}  {r['err_c']:9.3e}  "
            f"  {best:>4}   [{wa_idx[0]},{wa_idx[1]}]={wa_val:.2e}  [{wc_idx[0]},{wc_idx[1]}]={wc_val:.2e}"
        )

    # Distribution analysis: top-k steps by err_b — where are they in g_stag?
    print(f"\n  Distribution of top-{top_k} (by err_b) in |g_stag| bands:")
    for band in (0.01, 0.05, 0.1, 0.5, 2.0):
        inside = sum(1 for r in sorted_rows[:top_k] if abs(r["g_stag"]) < band)
        print(f"    |g_stag| < {band:.2f}: {inside}/{min(top_k, len(sorted_rows))}")

    # Verdict
    max_a = max(r["err_a"] for r in rows)
    max_b = max(r["err_b"] for r in rows)
    max_c = max(r["err_c"] for r in rows)
    print(f"\n  VERDICT for {name}:")
    if max_a < max_b * 0.5:
        print("    >> (a) no_correction is BETTER than HEAD  — correction HURTS → 容疑1 有罪")
    else:
        print("    >> (a) and (b) are comparable — correction is neutral/beneficial")
    if max_c < max_b * 0.9:
        print("    >> (c) fixed is BETTER than HEAD         — A1+A2 修正が効く")
    elif max_c < max_a * 0.9:
        print("    >> (c) fixed is better than (a)          — A1/A2 修正で改善")
    else:
        print("    >> (c) fixed shows no clear improvement")


def main():
    scenarios = _build_scenarios()
    for name, history in scenarios.items():
        _diagnose_scenario(name, history)

    print(f"\n{'='*75}")
    print("OVERALL INTERPRETATION")
    print(f"{'='*75}")
    print("  容疑1 確定条件: err_a < err_b * 0.5 (補正項が悪化要因)")
    print("  A1+A2 修正確定: err_c < err_b * 0.9 (12.5 + 1.5 ファクタで改善)")
    print("  g_stag 集中確認: top-k が |g_stag| < 0.05 に偏る → 遷移帯バグ")


if __name__ == "__main__":
    main()
