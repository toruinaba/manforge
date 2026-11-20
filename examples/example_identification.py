"""From a test record to model parameters: tune by hand, then fit.

The walk-through of docs/guide/tutorial.md as one script:
1. Read a stress-strain record from CSV
2. Look up what an existing model needs
3. Tune by hand — a few candidates side by side, scored against the record
4. Fit from the best hand-tuned candidate
5. Check against the full record, and finish the fit on it
6. Sweep one parameter around the result

The record is synthetic (examples/data/generate.py): an Armstrong-Frederick
response with measurement noise, standing in for a testing machine's export.

Takes about 50 seconds: the AF model has no hand-written return mapping, so
each evaluation of a fit is a run solved by automatic differentiation.

Usage
-----
    uv run python examples/example_identification.py
"""

import warnings
from pathlib import Path

import numpy as np

from returnmap.fitting import FitTarget, ParameterSpace, SlowFitWarning, fit_params
from returnmap.models import AFKinematic1D, J2Isotropic1D
from returnmap.simulation import FieldHistory, FieldType, compare_models

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


def rms(run):
    """Root-mean-square stress error of a scored run, in MPa."""
    return np.sqrt(run.fit_residual)


# ---------------------------------------------------------------------------
# 1. The test record: strain and stress, one row per sample
# ---------------------------------------------------------------------------
record = np.loadtxt(Path(__file__).parent / "data" / "cyclic_tension_compression.csv",
                    delimiter=",", skiprows=1)
strain_all, stress_all = record[:, 0], record[:, 1]

# Work on every 3rd sample while tuning: a run costs a few ms per step, and
# the full record is only needed at the end.
strain, stress = strain_all[::3], stress_all[::3]
load = FieldHistory(FieldType.STRAIN, strain)
target = FitTarget.stress_strain(strain, stress)     # "this strain should give this stress"

banner("1. Test record")
print(f"  {len(strain_all)} samples, strain {strain_all.min():+.4f} .. {strain_all.max():+.4f}, "
      f"stress {stress_all.min():+.0f} .. {stress_all.max():+.0f} MPa")
print(f"  tuning on every 3rd sample: {len(strain)} steps")

# ---------------------------------------------------------------------------
# 2. An existing model — what it needs is written on the class
# ---------------------------------------------------------------------------
banner("2. AFKinematic1D parameters")
for name, meta in AFKinematic1D.param_meta.items():
    unit = f" [{meta.unit}]" if meta.unit else ""
    print(f"  {name:9s} {meta.doc}{unit}; typical {meta.default:g}, range {meta.bounds}")

ELASTIC = dict(E=206_000.0, nu=0.3)           # read off the elastic slope of the record
start = AFKinematic1D(**ELASTIC, sigma_y0=250.0, C_k=20_000.0, gamma=100.0)

# ---------------------------------------------------------------------------
# 3. Tune by hand: change parameters, compare candidates against the record.
#    with_params makes a variant; targets=[...] scores each one.
# ---------------------------------------------------------------------------
candidates = {
    "J2, H=5000": J2Isotropic1D(**ELASTIC, sigma_y0=300.0, H=5_000.0),
    "AF start": start,
    "AF yield up": start.with_params(sigma_y0=290.0),
    "AF yield up, C_k up": start.with_params(sigma_y0=290.0, C_k=26_000.0),
    "AF by hand": start.with_params(sigma_y0=290.0, C_k=26_000.0, gamma=135.0),
}
tuned = compare_models(candidates, load, targets=[target])

banner("3. Candidates tuned by hand")
print(tuned.summary())
for name, run in tuned.runs.items():
    print(f"  {name:22s} RMS stress error {rms(run):6.1f} MPa")
best = min(tuned.runs.values(), key=lambda run: run.fit_residual)
print(f"\n  best by hand: {best.name}  (the noise in the record is about 3 MPa)")

# ---------------------------------------------------------------------------
# 4. Fit, starting from the hand-tuned values
# ---------------------------------------------------------------------------
by_hand = candidates["AF by hand"]


def space_around(model):
    """Search the three AF parameters, starting from ``model``'s values."""
    return ParameterSpace(
        free={                                 # name: (start, (lower, upper))
            "sigma_y0": (model.sigma_y0, (150.0, 450.0)),
            "C_k": (model.C_k, (5_000.0, 80_000.0)),
            "gamma": (model.gamma, (20.0, 500.0)),
        },
        fixed=ELASTIC,
    )


def report(n, residual, best):
    if n % 20 == 0:
        print(f"  evaluation {n:3d}: RMS {np.sqrt(best):6.2f} MPa")


banner("4. Fit on the samples used for tuning")
# fit_params warns here that one evaluation takes a fraction of a second,
# because AF is solved by automatic differentiation.  That is expected in
# this script, so the warning is silenced; run a fit without this to read it.
warnings.simplefilter("ignore", SlowFitWarning)
fit = fit_params(
    by_hand, [target], space_around(by_hand),
    nr_jacobian="broyden",                     # faster for models solved by autodiff
    tol=1e-6,                                  # the record is noisy: 1e-12 buys nothing
    progress=report,
)
print(f"  {fit.n_evaluations} evaluations, converged: {fit.success}")
print("  fitted:", {k: round(v, 1) for k, v in fit.model_params.items()})

# ---------------------------------------------------------------------------
# 5. Check against the full record, then finish on it.
#    Every 3rd sample means strain increments three times larger, and the
#    integration error of a larger increment is absorbed into the parameters:
#    they fit the thinned record better than they fit the test.
# ---------------------------------------------------------------------------
load_all = FieldHistory(FieldType.STRAIN, strain_all)
target_all = FitTarget.stress_strain(strain_all, stress_all)

banner("5. Against all samples")
finalists = {"AF by hand": by_hand, "AF fitted (thinned)": fit.model}
thinned = compare_models(finalists, load, targets=[target])
full = compare_models(finalists, load_all, targets=[target_all])
for name in finalists:
    print(f"  {name:20s} RMS stress error {rms(thinned.runs[name]):5.1f} MPa on the "
          f"{len(strain)} samples tuned on, {rms(full.runs[name]):5.1f} MPa on all {len(strain_all)}")

print("\n  finishing on all samples, from the thinned fit:")
final = fit_params(
    fit.model, [target_all], space_around(fit.model),
    nr_jacobian="broyden", tol=1e-6, progress=report,
)
print(f"  {final.n_evaluations} evaluations, RMS {np.sqrt(final.residual):.1f} MPa "
      "(the noise in the record is about 3 MPa)")
print("  fitted:", {k: round(v, 1) for k, v in final.model_params.items()})
print("  (the record was generated with sigma_y0=285, C_k=26000, gamma=140)")

# ---------------------------------------------------------------------------
# 6. A parameter study around the result is the same call
# ---------------------------------------------------------------------------
study = compare_models(
    {f"gamma={g:g}": final.model.with_params(gamma=g) for g in (80.0, 110.0, 140.0, 200.0)},
    load_all, targets=[target_all],
)

banner("6. Parameter study: gamma")
for name, run in study.runs.items():
    print(f"  {name:10s} peak stress {run.stress[:, 0].max():6.1f} MPa   RMS error {rms(run):6.1f} MPa")

# ---------------------------------------------------------------------------
# Plot — the record, the hand-tuned start and the fit
# ---------------------------------------------------------------------------
if HAS_MATPLOTLIB:
    INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = (
        "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb")
    COLORS = {"AF start": "#2a78d6", "AF fitted": "#eb6834"}

    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.scatter(strain_all * 100, stress_all, s=8, color=MUTED, alpha=0.6, linewidths=0,
               label="test record", zorder=2)
    curves = {
        "AF start": tuned.runs["AF start"],
        "AF fitted": compare_models({"AF fitted": final.model}, load_all,
                                    targets=[target_all]).runs["AF fitted"],
    }
    for name, run in curves.items():
        ax.plot(run.strain[:, 0] * 100, run.stress[:, 0], color=COLORS[name],
                linewidth=2, label=f"{name} (RMS {rms(run):.0f} MPa)", zorder=3)

    ax.axhline(0.0, color=AXIS, linewidth=0.8, zorder=1)
    ax.axvline(0.0, color=AXIS, linewidth=0.8, zorder=1)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=SECONDARY)
    ax.set_xlabel("Axial strain ε₁₁  [%]", color=SECONDARY)
    ax.set_ylabel("Axial stress σ₁₁  [MPa]", color=SECONDARY)
    ax.set_title("Armstrong-Frederick model against the test record", color=INK, loc="left")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, labelcolor=INK,
               loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    out = "examples/output_identification.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"\n  Plot saved → {out}")
else:
    print("\n  matplotlib not installed — skipping plot.")
    print("  uv sync --extra examples")
