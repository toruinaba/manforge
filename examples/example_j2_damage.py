"""Uniaxial tension with unload/reload cycles: J2 + weakly coupled Lemaitre damage.

Demonstrates:
- J2IsotropicDamage1D next to the undamaged J2Isotropic1D under the same strain
- Stiffness degradation: the unloading slope measured from the stress–strain
  curve equals (1 − D)·E
- Weak coupling: the damaged model's ep history is identical to the undamaged one

Usage
-----
    uv run --extra examples python examples/example_j2_damage.py
"""

import numpy as np

from manforge.models import J2Isotropic1D, J2IsotropicDamage1D
from manforge.simulation import PythonNumericalIntegrator

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ---------------------------------------------------------------------------
# Models (MPa)
# ---------------------------------------------------------------------------
P = dict(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_500.0)
damaged = J2IsotropicDamage1D(**P, S=0.3, s=1.0, p_D=0.02)
reference = J2Isotropic1D(**P)

# ---------------------------------------------------------------------------
# Strain-controlled loading: tension to ε_peak, unload to σ ≈ 0, reload, ...
# The unload target is computed from the current damaged modulus so every
# unload ends near zero stress instead of yielding in compression.  After the
# last cycle, load monotonically until D reaches D_STOP: the model has no
# critical damage / element deletion, and its σ row turns singular as D → 1.
# ---------------------------------------------------------------------------
PEAKS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
DEPS = 5e-4            # loading strain step
N_UNLOAD = 10
D_STOP, EPS_MAX = 0.98, 0.6

integ = PythonNumericalIntegrator(damaged)
integ_ref = PythonNumericalIntegrator(reference)
stress, state = np.zeros(1), damaged.initial_state()
stress_r, state_r = np.zeros(1), reference.initial_state()

eps_hist, sig_hist, sig_ref_hist, D_hist, ep_err = [0.0], [0.0], [0.0], [0.0], []
unloads = []  # (ε at peak, measured slope / E, 1 − D at peak)
eps = 0.0


def step(target):
    global eps, stress, state, stress_r, state_r
    deps = np.array([target - eps])
    r = integ.stress_update(deps, stress, state)
    rr = integ_ref.stress_update(deps, stress_r, state_r)
    stress, state = np.asarray(r.stress), r.state
    stress_r, state_r = np.asarray(rr.stress), rr.state
    eps = target
    eps_hist.append(eps)
    sig_hist.append(float(stress[0]))
    sig_ref_hist.append(float(stress_r[0]))
    D_hist.append(float(state["D"]))
    ep_err.append(abs(float(state["ep"]) - float(state_r["ep"])))


for peak in PEAKS:
    n_load = int(round((peak - eps) / DEPS))
    for e in np.linspace(eps, peak, n_load + 1)[1:]:
        step(e)
    sig_peak, eps_peak, D_peak = float(stress[0]), eps, float(state["D"])
    eps_zero = eps - 0.98 * sig_peak / ((1.0 - D_peak) * P["E"])
    for e in np.linspace(eps, eps_zero, N_UNLOAD + 1)[1:]:
        step(e)
    slope = (sig_peak - float(stress[0])) / (eps_peak - eps)
    unloads.append((eps_peak, slope / P["E"], 1.0 - D_peak))

while D_hist[-1] < D_STOP and eps < EPS_MAX:
    step(eps + DEPS)

eps_hist = np.array(eps_hist)
print(f"Final: ε = {eps:.3f}, D = {D_hist[-1]:.3f}, σ = {sig_hist[-1]:.1f} MPa "
      f"(peak {max(sig_hist):.1f} MPa, undamaged {sig_ref_hist[-1]:.1f} MPa)")
print(f"max |ep_damaged − ep_undamaged| = {max(ep_err):.1e}  (weak coupling)")
print("Unloading modulus vs (1 − D):")
for e, ratio, one_minus_d in unloads:
    print(f"  ε = {e:.2f}:  E_unload/E = {ratio:.4f},  1 − D = {one_minus_d:.4f}")

# ---------------------------------------------------------------------------
# Plot — three small multiples sharing the strain axis
# ---------------------------------------------------------------------------
if HAS_MATPLOTLIB:
    BLUE, ORANGE = "#2a78d6", "#eb6834"            # categorical slots 1, 2
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
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True)
    x = eps_hist * 100

    ax = axes[0]
    ax.plot(x, sig_ref_hist, color=INK2, lw=1.5, ls="--", label="Undamaged J2")
    ax.plot(x, sig_hist, color=BLUE, lw=2, label=f"J2 + Lemaitre damage (final σ = {sig_hist[-1]:.0f} MPa)")
    ax.annotate("undamaged", (x[-1], sig_ref_hist[-1]), xytext=(-6, 4),
                textcoords="offset points", ha="right", color=INK2)
    ax.set_title("Stress–strain (softening to failure)", loc="left")
    ax.set_xlabel("Axial strain ε₁₁ [%]")
    ax.set_ylabel("Nominal stress σ₁₁ [MPa]")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    ax.plot(x, D_hist, color=BLUE, lw=2)
    ax.axvline(damaged.p_D * 100, color=MUTED, lw=1, ls=":")
    ax.text(damaged.p_D * 100, max(D_hist) * 0.95, "  ep = p_D", color=INK2, va="top")
    ax.text(x[-1], D_hist[-1], f"D = {D_hist[-1]:.2f}", color=INK, va="bottom", ha="right")
    ax.set_title("Damage D (frozen during unloading)", loc="left")
    ax.set_xlabel("Axial strain ε₁₁ [%]")
    ax.set_ylabel("D [-]")

    ax = axes[2]
    pe = np.array([u[0] for u in unloads]) * 100
    ax.plot(x, 1.0 - np.array(D_hist), color=BLUE, lw=2, label="1 − D (model state)")
    ax.plot(pe, [u[1] for u in unloads], ls="none", marker="o", ms=8,
            mfc=ORANGE, mec=SURFACE, mew=2, label="Measured unload slope / E")
    ax.set_title("Elastic stiffness ratio", loc="left")
    ax.set_xlabel("Axial strain ε₁₁ [%]")
    ax.set_ylabel("E_eff / E [-]")
    ax.legend(frameon=False, loc="lower left")

    fig.suptitle("J2IsotropicDamage1D — weakly coupled ductile damage "
                 f"(S={damaged.S}, s={damaged.s}, p_D={damaged.p_D})",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    out = "examples/output_j2_damage.png"
    fig.savefig(out, dpi=150)
    print(f"  Plot saved → {out}")
else:
    print("  matplotlib not installed — skipping plot.")
