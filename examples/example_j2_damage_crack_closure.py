"""Tension/compression split of damage: J2 + Lemaitre crack-closure parameter h.

Demonstrates:
- J2IsotropicDamageCrackClosure1D: the damage driving force Y is split on the
  principal effective stresses, so compression drives damage h times as hard
  as tension (h = 1: no split, h = 0: compression never damages)
- Symmetric strain cycling: with h < 1, D climbs in a staircase — steep on
  the tensile half-cycles, shallow on the compressive ones
- The tangent stays exact through the split (checked against FD at the end)

Scope: only Y is split.  The nominal stress is σ = (1 − D) σ̃ in both
tension and compression — the stiffness does not recover on crack closure.

Usage
-----
    uv run --extra examples python examples/example_j2_damage_crack_closure.py
"""

import numpy as np

from returnmap.models import J2IsotropicDamageCrackClosure1D
from returnmap.simulation import StrainDriver, PythonNumericalIntegrator, FieldHistory
from returnmap.verification import check_tangent

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ---------------------------------------------------------------------------
# Models (MPa) — identical except for the crack-closure parameter h
# ---------------------------------------------------------------------------
P = dict(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_500.0, S=0.6, s=1.0, p_D=0.0)
H_VALUES = [1.0, 0.2, 0.0]

# Symmetric strain cycling ±1 %, starting in tension.  Isotropic hardening is
# unbounded under cycling, so σ̃ — and with it Y — keeps rising; larger
# amplitudes or more cycles drive D to 1 (no failure criterion in the model).
AMP, N_CYCLES, N_SEG = 0.01, 15, 80
load = FieldHistory.cyclic_strain([AMP, -AMP] * N_CYCLES + [AMP], n_per_segment=N_SEG)
strain = np.asarray(load.data).reshape(-1)

results = {}
for h in H_VALUES:
    model = J2IsotropicDamageCrackClosure1D(**P, h=h)
    res = StrainDriver(PythonNumericalIntegrator(model)).run(load)
    stress = np.asarray(res.stress)[:, 0]
    ep = np.array([float(r.state["ep"]) for r in res.step_results])
    D = np.array([float(r.state["D"]) for r in res.step_results])
    # peak tensile stress of each cycle (at each +AMP reversal)
    peaks_idx = [i for i in range(len(strain)) if np.isclose(strain[i], AMP)]
    results[h] = dict(stress=stress, ep=ep, D=D, peak=stress[peaks_idx])

print(f"Strain cycling ±{AMP * 100:.0f} %, {N_CYCLES} cycles")
for h, r in results.items():
    print(f"  h = {h:.1f}:  D_end = {r['D'][-1]:.3f},  "
          f"peak σ  cycle 1 → last = {r['peak'][0]:.0f} → {r['peak'][-1]:.0f} MPa")

# Tangent check at the end of the last compressive half-cycle (h = 0.2)
model = J2IsotropicDamageCrackClosure1D(**P, h=0.2)
integ = PythonNumericalIntegrator(model)
res = StrainDriver(integ).run(FieldHistory(load.type, load.name, load.data[:-N_SEG - 5]))
last = res.step_results[-1]
deps = np.array([strain[len(res.step_results)] - strain[len(res.step_results) - 1]])
tc = check_tangent(integ, deps, np.asarray(last.stress), last.state, eps=1e-8)
print(f"Tangent check (compressive plastic step, h = 0.2): "
      f"[{'PASS' if tc.passed else 'FAIL'}] max rel err = {tc.max_rel_err:.1e}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
if HAS_MATPLOTLIB:
    COLORS = {1.0: "#2a78d6", 0.2: "#eb6834", 0.0: "#1baf7a"}  # categorical slots 1–3
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
    axes = [axs[0, 1], axs[1, 0], axs[1, 1]]

    # (0) stress–strain: first vs last cycle.  Drawing all 15 loops per h
    # buries one model under the other; the first cycle is the same for every
    # h (D is still tiny), so it is drawn once as a muted reference.
    ax = axs[0, 0]
    first = slice(N_SEG, 3 * N_SEG + 1)            # +A → −A → +A, cycle 1
    last = slice(len(strain) - 2 * N_SEG - 1, len(strain))
    ax.plot(strain[first] * 100, results[1.0]["stress"][first], color=MUTED, lw=1.2,
            ls="--", label="cycle 1 (all h)")
    for h in H_VALUES:
        ax.plot(strain[last] * 100, results[h]["stress"][last], color=COLORS[h], lw=2,
                label=f"cycle {N_CYCLES}, h = {h:.1f}")
    ax.axhline(0.0, color=AXIS, lw=0.8)
    ax.set_title(f"Stress–strain, cycle 1 vs cycle {N_CYCLES}", loc="left", pad=16)
    ax.text(0.0, 1.0, "σ = (1 − D) σ̃ in tension and compression "
            "(stage 1: no stiffness recovery)",
            transform=ax.transAxes, color=INK2, ha="left", va="bottom", fontsize=8,
            )
    ax.set_xlabel("Axial strain ε₁₁ [%]")
    ax.set_ylabel("Nominal stress σ₁₁ [MPa]")
    ax.legend(frameon=False, loc="upper left")

    # (a) staircase: D vs accumulated plastic strain
    ax = axes[0]
    for h, r in results.items():
        ax.plot(r["ep"] * 100, r["D"], color=COLORS[h], lw=2, label=f"h = {h:.1f}")
        ax.annotate(f"h = {h:.1f}", (r["ep"][-1] * 100, r["D"][-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", color=INK2)
    ax.set_title("Damage vs accumulated plastic strain", loc="left")
    ax.set_xlabel("Accumulated plastic strain p [%]")
    ax.set_ylabel("D [-]")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.08)
    ax.legend(frameon=False, loc="upper left")

    # (b) one cycle zoom of the staircase (h = 0.2): tension steep, compression flat
    ax = axes[1]
    r = results[0.2]
    seg = slice(4 * N_SEG, 6 * N_SEG + 1)          # 3rd cycle: + → − → +
    x = np.arange(len(strain))[seg]
    tens = r["stress"][seg] > 0
    ax.plot(x, r["D"][seg], color=COLORS[0.2], lw=2)
    ax.fill_between(x, 0, 1, where=tens, transform=ax.get_xaxis_transform(),
                    color=GRID, alpha=0.6, lw=0)
    ax.text(0.02, 0.96, "shaded: σ > 0 (tension)", transform=ax.transAxes,
            va="top", color=INK2)
    ax.set_title("h = 0.2, one cycle: D grows mainly in tension", loc="left")
    ax.set_xlabel("Increment")
    ax.set_ylabel("D [-]")

    # (c) cyclic softening: peak tensile stress per cycle
    ax = axes[2]
    for h, r in results.items():
        cyc = np.arange(1, len(r["peak"]) + 1)
        ax.plot(cyc, r["peak"], color=COLORS[h], lw=2, marker="o", ms=6,
                mec=SURFACE, mew=1.5, label=f"h = {h:.1f}")
        ax.annotate(f"h = {h:.1f}", (cyc[-1], r["peak"][-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK2)
    ax.set_title("Peak tensile stress per cycle", loc="left")
    ax.set_xlabel("Cycle")
    ax.set_ylabel(f"σ₁₁ at +{AMP * 100:.0f} % [MPa]")
    ax.set_xlim(right=len(results[1.0]["peak"]) + 1.5)
    ax.legend(frameon=False, loc="lower left")

    fig.suptitle("J2IsotropicDamageCrackClosure1D — principal split of Y "
                 f"(±{AMP * 100:.0f} % strain cycling, S={P['S']}, s={P['s']})",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    out = "examples/output_j2_damage_crack_closure.png"
    fig.savefig(out, dpi=150)
    print(f"  Plot saved → {out}")
else:
    print("  matplotlib not installed — skipping plot.")
