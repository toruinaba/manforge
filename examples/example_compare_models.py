"""Comparing constitutive models on one loading history.

Demonstrates compare_models, which runs several models through the same
history and reports, for each:
- the response (stress, strain and any state history, for plotting)
- the cost (Newton iterations per plastic step, time per step)
- the goodness of fit against experimental curves, when targets are given
- where it stopped, if it could not complete the history

Usage
-----
    uv run python examples/example_compare_models.py
"""

import numpy as np

from returnmap.fitting import FitTarget, ParameterSpace, fit_params
from returnmap.models import (
    AFKinematic1D, AFKinematic3D, J2Isotropic1D, OWKinematic1D,
)
from returnmap.simulation import (
    FieldHistory, MixedDriver, PythonIntegrator, StrainDriver, compare_models,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def banner(title):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Models (steel, MPa units) and one cyclic strain history
# ---------------------------------------------------------------------------
ELASTIC = dict(E=206_000.0, nu=0.3)

j2 = J2Isotropic1D(**ELASTIC, sigma_y0=250.0, H=2_000.0)
af = AFKinematic1D(**ELASTIC, sigma_y0=250.0, C_k=20_000.0, gamma=100.0)
ow = OWKinematic1D(**ELASTIC, sigma_y0=250.0, C_k=20_000.0, gamma=100.0)

load = FieldHistory.cyclic_strain([0.015, -0.015, 0.015], n_per_segment=40)

# ---------------------------------------------------------------------------
# 1. Response and cost — the loading history is all it takes
# ---------------------------------------------------------------------------
comparison = compare_models({"J2": j2, "AF": af, "OW": ow}, load)

banner("1. Response and cost")
print(comparison.summary())
# Each run reads like a DriverResult:
#   comparison.runs["AF"].stress / .strain / .history("alpha")

# ---------------------------------------------------------------------------
# 2. Goodness of fit — score the same models against an experimental curve
#    The "experiment" is the AF response with noise.  J2 is fitted to it
#    first; isotropic hardening has no Bauschinger effect, so the fit improves
#    it but cannot close the gap.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
stress_exp = comparison.runs["AF"].stress[:, 0] + rng.normal(0.0, 3.0, len(load.data))
target = FitTarget.stress_strain(load.data, stress_exp)

fit = fit_params(
    j2, [target],
    ParameterSpace(
        free={"sigma_y0": (250.0, (100.0, 500.0)), "H": (2_000.0, (0.0, 50_000.0))},
        fixed=ELASTIC,
    ),
)
scored = compare_models(
    {"J2": j2, "J2 (fitted)": fit.model, "AF": af, "OW": ow}, load, targets=[target],
)

banner("2. Goodness of fit (experiment = AF + noise)")
print(scored.summary())
print(f"\n  fit_params reported {fit.residual:.4g} for the fitted J2 — the same number.")

# ---------------------------------------------------------------------------
# 3. A model that cannot complete the history is a row, not an exception
# ---------------------------------------------------------------------------
absurd = OWKinematic1D(**ELASTIC, sigma_y0=250.0, C_k=1e9, gamma=1e12)
partial = compare_models({"AF": af, "OW (absurd recovery)": absurd}, load)

banner("3. Non-convergence")
print(partial.summary())
print(f"\n  completed steps are still there: "
      f"{partial.runs['OW (absurd recovery)'].stress.shape[0]} rows of stress")

# ---------------------------------------------------------------------------
# 4. Same model, different solvers — pass integrators instead of models.
#    A 3D model under uniaxial stress needs the mixed driver; the factory
#    sees the integrator, so one axial history serves every dimension.
# ---------------------------------------------------------------------------
af_3d = AFKinematic3D(**ELASTIC, sigma_y0=250.0, C_k=20_000.0, gamma=100.0)


def uniaxial_stress(integrator):
    if integrator.ntens == 1:
        return StrainDriver(integrator)
    return MixedDriver(integrator, prescribed_strain_idx=[0])


solvers = compare_models(
    {
        "AF 1D": af,
        "AF 3D, exact Jacobian": PythonIntegrator(af_3d),
        "AF 3D, Broyden": PythonIntegrator(af_3d, nr_jacobian="broyden"),
    },
    load,
    driver_factory=uniaxial_stress,
)

banner("4. Solver cost (uniaxial stress)")
print(solvers.summary())

# ---------------------------------------------------------------------------
# Plot — the three responses against the experiment
# ---------------------------------------------------------------------------
if HAS_MATPLOTLIB:
    INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
    # One hue per model, in a fixed order; the experiment is neutral.
    COLORS = {"J2 (fitted)": "#2a78d6", "AF": "#eb6834", "OW": "#1baf7a"}

    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.scatter(load.data * 100, stress_exp, s=10, color=MUTED, alpha=0.6,
               linewidths=0, label="experiment (AF + noise)", zorder=2)
    for name, color in COLORS.items():
        run = scored.runs[name]
        ax.plot(run.strain[:, 0] * 100, run.stress[:, 0], color=color,
                linewidth=2, label=name, zorder=3)

    ax.axhline(0.0, color="#c3c2b7", linewidth=0.8, zorder=1)
    ax.axvline(0.0, color="#c3c2b7", linewidth=0.8, zorder=1)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelcolor="#52514e")
    ax.set_xlabel("Axial strain ε₁₁  [%]", color="#52514e")
    ax.set_ylabel("Axial stress σ₁₁  [MPa]", color="#52514e")
    ax.set_title("Three models on one cyclic history", color=INK, loc="left")
    # Below the axes: the loops fill the plot area, so any corner collides.
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, labelcolor=INK,
               loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    out = "examples/output_compare_models.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"\n  Plot saved → {out}")
else:
    print("\n  matplotlib not installed — skipping plot.")
    print("  uv sync --extra examples")
