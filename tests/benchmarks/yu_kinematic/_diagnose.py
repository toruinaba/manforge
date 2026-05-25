"""Diagnostic script — not a pytest test.

Runs each YU 3D scenario and prints per-step error breakdown sorted by
ddsdde error. Used to confirm that ddsdde errors concentrate near the
stagnation-surface transition zone (theta_max ≈ B-Y = 75 MPa).

Key question: Is ddsdde error localised to |theta_max - 75| < ε (Pattern A:
_prepare_Rtheta hard-if is the cause), or spread across all steps (Pattern B:
smooth_heaviside β=50 structural difference)?

Run with:
    uv run python tests/benchmarks/yu_kinematic/_diagnose.py 2>&1 | tee /tmp/yu_diag.txt
"""

import numpy as np
import autograd.numpy as anp

from manforge.models import YUKinematic3D
from manforge.simulation.integrator import (
    PythonNumericalIntegrator,
    PythonAnalyticalIntegrator,
)
from manforge.simulation.types import FieldHistory

PARAMS = dict(
    E=206_000, nu=0.3, Y=360.0, C_1=2000.0, C_2=200.0,
    B=435.0, Rsat=255.0, k=26.0, b=66.0, h=0.4, Ea=159_000, xi=61.0,
)
B_MINUS_Y = PARAMS["B"] - PARAMS["Y"]  # 75.0 MPa


def _build_scenarios():
    n = 50
    mono = np.zeros((n, 6))
    mono[:, 0] = np.linspace(0.0, 5e-3, n)

    small_cyc = FieldHistory.cyclic_strain(
        [1e-3, -1e-3, 1e-3], n_per_segment=15, ntens=6,
    ).data

    large_cyc = FieldHistory.cyclic_strain(
        [0.05, -0.05, 0.05, -0.05], n_per_segment=50, ntens=6,
    ).data

    return {
        "uniaxial_monotonic":     mono,
        "small_amplitude_cyclic": small_cyc,
        "uniaxial_cyclic":        large_cyc,
    }


def _diagnose_scenario(name, history, *, top_k=10):
    model = YUKinematic3D(**PARAMS)
    int_n = PythonNumericalIntegrator(model)
    int_a = PythonAnalyticalIntegrator(model)

    stress_n = np.zeros(6)
    state_n = model.initial_state()
    eps_prev = np.zeros(6)

    state_keys = ("theta", "beta", "R", "q", "r", "eps_eq", "theta_max")
    rows = []

    for i, eps in enumerate(history):
        deps = eps - eps_prev
        eps_prev = eps.copy()

        r_n = int_n.stress_update(anp.array(deps), anp.array(stress_n), state_n)
        r_a = int_a.stress_update(anp.array(deps), anp.array(stress_n), state_n)

        s_n = np.asarray(r_n.stress)
        s_a = np.asarray(r_a.stress)
        stress_err = float(np.max(np.abs(s_a - s_n)))

        ddsdde_err = 0.0
        if r_n.is_plastic and r_a.is_plastic:
            D_n = np.asarray(r_n.ddsdde)
            D_a = np.asarray(r_a.ddsdde)
            ddsdde_err = float(np.max(np.abs(D_a - D_n) / (np.abs(D_n) + 1.0)))

        theta_max = float(r_n.state["theta_max"])

        state_errs = {}
        for k in state_keys:
            if k in r_n.state:
                v_n = np.asarray(r_n.state[k])
                v_a = np.asarray(r_a.state[k])
                state_errs[k] = float(np.max(np.abs(v_a - v_n) / (np.abs(v_n) + 1.0)))

        rows.append({
            "i": i,
            "theta_max": theta_max,
            "dist_to_transition": theta_max - B_MINUS_Y,
            "stress_err": stress_err,
            "ddsdde_err": ddsdde_err,
            "state_errs": state_errs,
            "is_plastic": r_n.is_plastic,
            "iter_n": r_n.n_iterations or 0,
            "iter_a": r_a.n_iterations or 0,
        })

        stress_n = np.asarray(r_n.stress)
        state_n = r_n.state

    plastic_rows = [r for r in rows if r["is_plastic"]]
    total_steps = len(rows)
    plastic_steps = len(plastic_rows)

    print(f"\n{'='*70}")
    print(f"Scenario: {name}  ({total_steps} steps, {plastic_steps} plastic)")
    print(f"B-Y = {B_MINUS_Y:.1f} MPa  (transition zone threshold)")
    print(f"{'='*70}")

    if not plastic_rows:
        print("  (no plastic steps)")
        return

    max_stress = max(r["stress_err"] for r in plastic_rows)
    max_ddsdde = max(r["ddsdde_err"] for r in plastic_rows)
    max_iter_diff = max(abs(r["iter_a"] - r["iter_n"]) for r in plastic_rows)
    print(f"  max stress_err  = {max_stress:.3e}")
    print(f"  max ddsdde_err  = {max_ddsdde:.3e}")
    print(f"  max iter_diff   = {max_iter_diff}")

    # per-state-key max
    for k in state_keys:
        vals = [r["state_errs"].get(k, 0.0) for r in plastic_rows]
        print(f"  max state[{k:9s}]_err = {max(vals):.3e}")

    sorted_rows = sorted(plastic_rows, key=lambda r: -r["ddsdde_err"])
    print(f"\n  Top {top_k} plastic steps by ddsdde_err:")
    hdr = (f"  {'step':>4}  {'theta_max':>9}  {'dist_to_75':>10}  "
           f"{'stress_err':>10}  {'ddsdde_err':>10}  "
           f"{'st[R]':>8}  {'st[q]':>8}  {'plastic':>7}  {'itn':>3}  {'ita':>3}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in sorted_rows[:top_k]:
        st = r["state_errs"]
        print(
            f"  {r['i']:4d}  {r['theta_max']:9.4f}  {r['dist_to_transition']:+10.4f}  "
            f"{r['stress_err']:10.3e}  {r['ddsdde_err']:10.3e}  "
            f"{st.get('R', 0.0):8.2e}  {st.get('q', 0.0):8.2e}  "
            f"{str(r['is_plastic']):>7}  {r['iter_n']:3d}  {r['iter_a']:3d}"
        )

    # distribution check: how many of top-10 are inside |dist| < 1.0?
    for band in (0.1, 0.5, 1.0, 2.0, 5.0):
        inside = sum(1 for r in sorted_rows[:top_k]
                     if abs(r["dist_to_transition"]) < band)
        print(f"  Top-{top_k} inside |dist| < {band:.1f}: {inside}/{min(top_k, len(sorted_rows))}")


def main():
    scenarios = _build_scenarios()
    for name, history in scenarios.items():
        _diagnose_scenario(name, history)

    print(f"\n{'='*70}")
    print("INTERPRETATION GUIDE")
    print(f"{'='*70}")
    print("  Pattern A: top ddsdde_err concentrated near |dist_to_75| < ~1.0")
    print("             → _prepare_Rtheta hard-if is the cause (Jacobian only)")
    print("             → split assert: strict outside band, relaxed inside")
    print()
    print("  Pattern B: ddsdde_err spread evenly, no dist_to_75 correlation")
    print("             → smooth_heaviside β=50 structural difference")
    print("             → single tolerance ~1e-3 for all steps")


if __name__ == "__main__":
    main()
