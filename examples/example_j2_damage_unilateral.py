"""Crack closure in the stress: stiffness recovers in compression.

Demonstrates:
- J2IsotropicDamageUnilateral1D (stage 2): Lemaitre's quasi-unilateral
  complementary energy — stiffness (1 − D)E in tension, (1 − hD)E in compression
- Against J2IsotropicDamageCrackClosure1D (stage 1, same h): in 1D both have
  the *same* damage driving force, so the D history is identical; only the
  compressive stress differs
- The elastic slope switches at σ = 0 inside one unloading branch (the
  hallmark of crack closure), read off the autodiff elastic tangent

Usage
-----
    uv run --extra examples python examples/example_j2_damage_unilateral.py
"""

import numpy as np

from manforge.models import (
    J2IsotropicDamageCrackClosure1D, J2IsotropicDamageUnilateral1D,
)
from manforge.simulation import StrainDriver, PythonNumericalIntegrator, FieldHistory
from manforge.verification import check_tangent

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ---------------------------------------------------------------------------
# Models (MPa) — same parameters, same h; stage 1 vs stage 2
# ---------------------------------------------------------------------------
P = dict(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_500.0, S=0.6, s=1.0, p_D=0.0, h=0.2)
MODELS = {
    "stage 2 (stress split)": J2IsotropicDamageUnilateral1D(**P),
    "stage 1 (Y split only)": J2IsotropicDamageCrackClosure1D(**P),
}

AMP, N_CYCLES, N_SEG = 0.01, 15, 80
load = FieldHistory.cyclic_strain([AMP, -AMP] * N_CYCLES + [AMP], n_per_segment=N_SEG)
strain = np.asarray(load.data).reshape(-1)
pk_t = [i for i in range(len(strain)) if np.isclose(strain[i], AMP)]
pk_c = [i for i in range(len(strain)) if np.isclose(strain[i], -AMP)]

results = {}
for name, model in MODELS.items():
    res = StrainDriver(PythonNumericalIntegrator(model)).run(load)
    stress = np.asarray(res.stress)[:, 0]
    D = np.array([float(r.state["D"]) for r in res.step_results])
    # Elastic slope on every elastic step (autodiff tangent of the predictor).
    slope = np.array([float(np.asarray(r.ddsdde)[0, 0]) if not r.is_plastic else np.nan
                      for r in res.step_results])
    results[name] = dict(stress=stress, D=D, slope=slope, res=res)

r2, r1 = results["stage 2 (stress split)"], results["stage 1 (Y split only)"]
print(f"Strain cycling ±{AMP * 100:.0f} %, {N_CYCLES} cycles, h = {P['h']}")
print(f"  max |D_stage2 − D_stage1| = {np.max(np.abs(r2['D'] - r1['D'])):.1e}  (same Y in 1D)")
for name, r in results.items():
    print(f"  {name}:  D_end = {r['D'][-1]:.3f},  last peaks σ = "
          f"{r['stress'][pk_t[-1]]:+.0f} / {r['stress'][pk_c[-1]]:+.0f} MPa")

# Tangent check on a late elastic step crossing σ = 0 (stage 2)
m = MODELS["stage 2 (stress split)"]
integ = PythonNumericalIntegrator(m)
steps = r2["res"].step_results
k = next(i for i in range(pk_t[-2], pk_c[-1]) if not steps[i].is_plastic
         and steps[i - 1].stress[0] > 0 > steps[i].stress[0])
tc = check_tangent(
    integ,
    np.array([strain[k] - strain[k - 1]]),
    np.asarray(steps[k - 1].stress),
    steps[k - 1].state,
    eps=1e-9,
)
print(f"Tangent check, elastic step across σ = 0: [{'PASS' if tc.passed else 'FAIL'}] "
      f"max rel err = {tc.max_rel_err:.1e}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
if HAS_MATPLOTLIB:
    COLORS = {"stage 2 (stress split)": "#2a78d6", "stage 1 (Y split only)": "#eb6834"}
    INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
    GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    })
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.4))

    # (a) first vs last cycle
    ax = axs[0, 0]
    first = slice(N_SEG, 3 * N_SEG + 1)
    last = slice(len(strain) - 2 * N_SEG - 1, len(strain))
    ax.plot(strain[first] * 100, r2["stress"][first], color=MUTED, lw=1.2, ls="--",
            label="cycle 1 (both)")
    for name, r in results.items():
        ax.plot(strain[last] * 100, r["stress"][last], color=COLORS[name], lw=2,
                label=f"cycle {N_CYCLES}, {name}")
    ax.axhline(0.0, color=AXIS, lw=0.8)
    ax.set_title(f"Stress–strain, cycle 1 vs cycle {N_CYCLES}", loc="left", pad=16)
    ax.text(0.0, 1.0, "same D history; stage 2 keeps the compressive stiffness",
            transform=ax.transAxes, color=INK2, ha="left", va="bottom", fontsize=8)
    ax.set_xlabel("Axial strain ε₁₁ [%]")
    ax.set_ylabel("Nominal stress σ₁₁ [MPa]")
    ax.legend(frameon=False, loc="upper left")

    # (b) zoom: last unloading from +peak through σ = 0
    ax = axs[0, 1]
    zoom = slice(pk_t[-2], pk_t[-2] + N_SEG // 2)
    for name, r in results.items():
        ax.plot(strain[zoom] * 100, r["stress"][zoom], color=COLORS[name], lw=2,
                marker="o", ms=3, label=name)
    ax.axhline(0.0, color=AXIS, lw=0.8)
    D_z = r2["D"][pk_t[-2]]
    ax.set_title("Unloading through σ = 0 (last tensile peak)", loc="left", pad=16)
    ax.text(0.0, 1.0, f"stage 2: slope (1−D)E above 0, (1−hD)E below (D = {D_z:.2f})",
            transform=ax.transAxes, color=INK2, ha="left", va="bottom", fontsize=8)
    ax.set_xlabel("Axial strain ε₁₁ [%]")
    ax.set_ylabel("Nominal stress σ₁₁ [MPa]")
    ax.legend(frameon=False, loc="lower right")

    # (c) peak stresses per cycle
    ax = axs[1, 0]
    cyc_t, cyc_c = np.arange(1, len(pk_t) + 1), np.arange(1, len(pk_c) + 1)
    for name, r in results.items():
        ax.plot(cyc_t, r["stress"][pk_t], color=COLORS[name], lw=2, marker="o", ms=5,
                mec=SURFACE, mew=1.2, label=f"{name}, tension")
        ax.plot(cyc_c, -r["stress"][pk_c], color=COLORS[name], lw=2, ls="--", marker="s",
                ms=5, mec=SURFACE, mew=1.2, label=f"{name}, |compression|")
    ax.set_title("Peak stress per cycle (tension solid, |compression| dashed)", loc="left",
                 pad=16)
    ax.text(0.0, 1.0, "tension peaks of both models coincide (same D, same (1−D))",
            transform=ax.transAxes, color=INK2, ha="left", va="bottom", fontsize=8)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("|σ₁₁| at peak [MPa]")
    ax.legend(frameon=False, loc="lower center", fontsize=8)

    # (d) elastic modulus on each side vs D
    ax = axs[1, 1]
    s2 = r2["stress"]
    tens = ~np.isnan(r2["slope"]) & (s2 > 0)
    comp = ~np.isnan(r2["slope"]) & (s2 < 0)
    Dg = np.linspace(0.0, r2["D"].max(), 50)
    ax.plot(Dg, 1 - Dg, color=MUTED, lw=1.2, ls="--", label="1 − D")
    ax.plot(Dg, 1 - P["h"] * Dg, color=MUTED, lw=1.2, ls=":", label="1 − hD")
    ax.plot(r2["D"][tens], r2["slope"][tens] / P["E"], ls="none", marker="o", ms=4,
            mfc=COLORS["stage 2 (stress split)"], mec=SURFACE, label="stage 2, σ > 0")
    # Both series are stage 2, so both wear its colour; the side is encoded
    # by marker (filled circle vs hollow square) and the reference line.
    ax.plot(r2["D"][comp], r2["slope"][comp] / P["E"], ls="none", marker="s", ms=5,
            mfc=SURFACE, mec=COLORS["stage 2 (stress split)"], mew=1.3,
            label="stage 2, σ < 0")
    ax.set_title("Elastic tangent of stage 2 (autodiff, elastic steps)", loc="left")
    ax.set_xlabel("D [-]")
    ax.set_ylabel("E_tangent / E [-]")
    ax.legend(frameon=False, loc="lower left")

    fig.suptitle("J2IsotropicDamageUnilateral1D — crack closure in the stress "
                 f"(±{AMP * 100:.0f} % cycling, h={P['h']}, S={P['S']})",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    out = "examples/output_j2_damage_unilateral.png"
    fig.savefig(out, dpi=150)
    print(f"  Plot saved → {out}")
else:
    print("  matplotlib not installed — skipping plot.")
