"""Stage 1 diagnostic: compare FortranIntegrator ddsdde vs FD truth (dσ/dε).

Checks whether the ddsdde error seen in _diagnose_ddsdde_fd.py is shared by
the Fortran implementation (i.e. both share the same formula).  If Python and
Fortran show the same error pattern the bug is in the shared algorithm, not a
translation artefact.

Requires the compiled yu_kinematic_3d extension:
    make fortran-build-yu

Run:
    uv run python tests/benchmarks/yu_kinematic/_diagnose_ddsdde_fortran.py 2>&1 | tee /tmp/yu_diag_fortran.txt
"""

import os
import sys

# Add fortran/ to sys.path so compiled .so modules are importable (mirrors tests/conftest.py)
_HERE = os.path.dirname(os.path.abspath(__file__))
_FORTRAN_DIR = os.path.join(_HERE, "..", "..", "..", "fortran")
sys.path.insert(0, os.path.abspath(_FORTRAN_DIR))

import numpy as np

try:
    import yu_kinematic_3d  # noqa: F401
except ImportError:
    print("ERROR: yu_kinematic_3d not compiled — run: make fortran-build-yu", file=sys.stderr)
    sys.exit(1)

from manforge.models import YUKinematic3D
from manforge.simulation.integrator import (
    FortranModule,
    FortranIntegrator,
    PythonAnalyticalIntegrator,
)
from manforge.simulation.types import FieldHistory

PARAMS = dict(
    E=206_000, nu=0.3, Y=360.0, C_1=2000.0, C_2=200.0,
    B=435.0, Rsat=255.0, k=26.0, b=66.0, h=0.4, Ea=159_000, xi=61.0,
)
B_MINUS_Y = PARAMS["B"] - PARAMS["Y"]  # 75.0 MPa
FD_H = 1e-5


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
    """Central-difference dσ/dε using the given integrator."""
    ntens = len(deps)
    D_fd = np.zeros((ntens, ntens))
    for j in range(ntens):
        eps_p = deps.copy(); eps_p[j] += h
        eps_m = deps.copy(); eps_m[j] -= h
        rp = integrator.stress_update(eps_p, stress_n, state_n)
        rm = integrator.stress_update(eps_m, stress_n, state_n)
        D_fd[:, j] = (np.asarray(rp.stress) - np.asarray(rm.stress)) / (2.0 * h)
    return D_fd


def _err(D_candidate, D_ref):
    return float(np.max(np.abs(D_candidate - D_ref) / (np.abs(D_ref) + 1.0)))


def _diagnose_scenario(name, history, *, fc_int, py_int, model, top_k=12):
    stress_n  = np.zeros(6)
    state_n   = model.initial_state()
    eps_prev  = np.zeros(6)

    rows = []
    for i, eps in enumerate(history):
        deps = eps - eps_prev
        eps_prev = eps.copy()

        # Run both integrators from same (stress_n, state_n)
        r_py = py_int.stress_update(deps, stress_n, state_n)
        r_fc = fc_int.stress_update(deps, stress_n, state_n)

        if r_py.is_plastic:  # FortranIntegrator has is_plastic=None; use Python to detect
            D_fd_py = _fd_ddsdde(py_int, deps, stress_n, state_n)
            D_fd_fc = _fd_ddsdde(fc_int, deps, stress_n, state_n)
            D_py    = np.asarray(r_py.ddsdde)
            D_fc    = np.asarray(r_fc.ddsdde)

            state_new = r_py.return_mapping.state
            g_xi   = np.asarray(state_new["beta"]) - np.asarray(state_n["q"])
            g_stag = float(model.vonmises_norm(g_xi)) - float(state_n["r"])
            t_max  = float(state_n["theta_max"])

            rows.append({
                "i":           i,
                "g_stag":      g_stag,
                "theta_max":   t_max,
                "dist_to_75":  t_max - B_MINUS_Y,
                # Python analytical vs its own FD
                "err_py":      _err(D_py, D_fd_py),
                # Fortran vs Python FD (cross-check)
                "err_fc_vs_pyfd": _err(D_fc, D_fd_py),
                # Fortran vs its own FD
                "err_fc":      _err(D_fc, D_fd_fc),
                # Python vs Fortran (existing cross-check)
                "err_py_fc":   _err(D_py, D_fc),
            })

        # Advance using Python integrator for consistent state trajectory
        stress_n = np.asarray(r_py.stress)
        state_n  = r_py.state

    print(f"\n{'='*75}")
    print(f"Scenario: {name}  ({len(rows)} plastic steps)")
    print(f"  B-Y = {B_MINUS_Y:.1f} MPa  FD_H={FD_H:.0e}")
    print(f"{'='*75}")

    if not rows:
        print("  (no plastic steps)")
        return

    for key, label in [
        ("err_py",         "Python analytical vs Python FD"),
        ("err_fc",         "Fortran          vs Fortran FD"),
        ("err_fc_vs_pyfd", "Fortran          vs Python  FD"),
        ("err_py_fc",      "Python           vs Fortran"),
    ]:
        vals = [r[key] for r in rows]
        print(f"  max {label:38s}: {max(vals):.3e}  mean: {np.mean(vals):.3e}")

    sorted_rows = sorted(rows, key=lambda r: -r["err_fc_vs_pyfd"])
    print(f"\n  Top {top_k} plastic steps by err_fc_vs_pyfd:")
    hdr = (f"  {'step':>4}  {'g_stag':>9}  {'dist_75':>8}  "
           f"{'err_py':>9}  {'err_fc':>9}  {'fc_vs_pyfd':>10}  {'py_fc':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in sorted_rows[:top_k]:
        print(
            f"  {r['i']:4d}  {r['g_stag']:+9.4f}  {r['dist_to_75']:+8.3f}  "
            f"{r['err_py']:9.3e}  {r['err_fc']:9.3e}  {r['err_fc_vs_pyfd']:10.3e}  {r['err_py_fc']:9.3e}"
        )

    # Is the Fortran error pattern the same as Python?
    corr = np.corrcoef([r["err_py"] for r in rows],
                       [r["err_fc_vs_pyfd"] for r in rows])[0, 1]
    print(f"\n  Correlation(err_py, err_fc_vs_pyfd) = {corr:.4f}")
    if corr > 0.9:
        print("    >> High correlation — Fortran and Python share the same error pattern")
        print("       Fixing Python formula should fix Fortran too.")
    else:
        print("    >> Low correlation — errors differ between Python and Fortran")
        print("       May be a translation artefact, inspect Fortran separately.")


def main():
    model  = YUKinematic3D(**PARAMS)
    fm     = FortranModule("yu_kinematic_3d")
    fc_int = FortranIntegrator.from_model(fm, "yu_kinematic_3d", model)
    py_int = PythonAnalyticalIntegrator(model)

    scenarios = _build_scenarios()
    for name, history in scenarios.items():
        _diagnose_scenario(name, history, fc_int=fc_int, py_int=py_int, model=model)

    print(f"\n{'='*75}")
    print("OVERALL INTERPRETATION")
    print(f"{'='*75}")
    print("  共有バグ確認: err_py と err_fc_vs_pyfd の相関が高い (>0.9) → 同一アルゴリズムのバグ")
    print("  Python 修正だけで Fortran も直る可能性が高い → Stage 2 で Python 修正後 Fortran 再検証")


if __name__ == "__main__":
    main()
