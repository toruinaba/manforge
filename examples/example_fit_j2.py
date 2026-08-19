"""Parameter fitting example for J2 isotropic hardening — cyclic loading.

Demonstrates:
- Building a cyclic strain history with FieldHistory.cyclic_strain
- Generating synthetic cyclic stress-strain data from known parameters
- Adding measurement noise to simulate experimental data
- Fitting sigma_y0 and H using fit_params (L-BFGS-B)
- Reparametrising H as a ratio of sigma_y0 via ParameterSpace
- Comparing true vs fitted parameters on the hysteresis loop
- Plotting the fit quality (saved as PNG)

Usage
-----
    uv run python examples/example_fit_j2.py
"""

import numpy as np

import manforge  # noqa: F401 — enables JAX float64
from manforge.models.j2_isotropic import J2Isotropic1D
from manforge.simulation.driver import StrainDriver
from manforge.simulation.integrator import PythonIntegrator
from manforge.simulation.types import FieldHistory
from manforge.fitting import FitTarget, Observation, ParameterSpace, fit_params

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ---------------------------------------------------------------------------
# True model
# ---------------------------------------------------------------------------
true_model = J2Isotropic1D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)
driver_factory = lambda i: StrainDriver(i)

# ---------------------------------------------------------------------------
# Generate synthetic "experimental" data (cyclic: 4 half-cycles)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
load = FieldHistory.cyclic_strain(
    peaks=[5e-3, -5e-3, 5e-3, -5e-3],
    n_per_segment=12,
    ntens=1,
)
stress_clean = driver_factory(PythonIntegrator(true_model)).run(load).stress[:, 0]
strain_exp = load.data[:, 0]

# Add small Gaussian noise (~0.5 MPa std) to simulate measurement scatter
noise_std = 0.5
stress_exp = stress_clean + rng.normal(0.0, noise_std, size=len(strain_exp))

exp_data = {"strain": strain_exp, "stress": stress_exp}

# ---------------------------------------------------------------------------
# Fit sigma_y0 and H (E and nu are fixed)
# ---------------------------------------------------------------------------
fit_config = {
    "sigma_y0": (180.0, (50.0, 600.0)),   # initial guess = 180 MPa
    "H":        (500.0, (0.0, 10_000.0)), # initial guess = 500 MPa
}
fixed_params = {"E": true_model.E, "nu": true_model.nu}

print("=" * 55)
print("  J2 Parameter Fitting — L-BFGS-B")
print("=" * 55)
print(f"  True  sigma_y0 = {true_model.sigma_y0:.1f} MPa")
print(f"  True  H        = {true_model.H:.1f} MPa")
print(f"  Init  sigma_y0 = {fit_config['sigma_y0'][0]:.1f} MPa")
print(f"  Init  H        = {fit_config['H'][0]:.1f} MPa")
print()
print("  Running optimisation …")

result = fit_params(
    true_model,
    driver_factory,
    exp_data,
    fit_config,
    fixed_params=fixed_params,
    method="L-BFGS-B",
    max_iter=200,
)

fitted_sigma_y0 = result.params["sigma_y0"]
fitted_H        = result.params["H"]
err_sy = abs(fitted_sigma_y0 - true_model.sigma_y0) / true_model.sigma_y0 * 100
err_H  = abs(fitted_H - true_model.H) / true_model.H * 100

print()
print("  Results:")
print(f"  Fitted sigma_y0 = {fitted_sigma_y0:.2f} MPa   (err {err_sy:.1f}%)")
print(f"  Fitted H        = {fitted_H:.2f} MPa   (err {err_H:.1f}%)")
print(f"  Residual        = {result.residual:.4f}")
print(f"  Converged       = {result.success}  ({result.n_iter} func. evals)")
print()

# ---------------------------------------------------------------------------
# Same fit, reparametrised: optimise H/sigma_y0 instead of H.
# A ratio bounded below by 1.0 enforces H >= sigma_y0 without an inequality
# constraint — the pattern for correlated parameters such as YU's Y < B.
# ---------------------------------------------------------------------------
space = ParameterSpace(
    free={
        "sigma_y0": (180.0, (50.0, 600.0)),
        "H_ratio":  (2.0,   (1.0, 20.0)),   # H/sigma_y0 >= 1
    },
    fixed={"E": true_model.E, "nu": true_model.nu},
    derived={"H": lambda p: p["sigma_y0"] * p["H_ratio"]},
)
result_ratio = fit_params(
    true_model, driver_factory, exp_data, space,
    method="L-BFGS-B", max_iter=200,
)

print("  Reparametrised (H = sigma_y0 x H_ratio):")
print(f"  Fitted sigma_y0 = {result_ratio.params['sigma_y0']:.2f} MPa")
print(f"  Fitted H_ratio  = {result_ratio.params['H_ratio']:.3f}")
print(f"  Derived H       = {result_ratio.params['H']:.2f} MPa")
print(f"  Residual        = {result_ratio.residual:.4f}")
print()

# ---------------------------------------------------------------------------
# Multi-target fit: two strain amplitudes measured as separate experiments.
# Each FitTarget is one simulation run; the residuals are normalised by data
# point count so the longer history does not dominate.
# ---------------------------------------------------------------------------
loads_multi = [
    FieldHistory.cyclic_strain(peaks=[3e-3, -3e-3], n_per_segment=10, ntens=1),
    FieldHistory.cyclic_strain(peaks=[8e-3, -8e-3, 8e-3], n_per_segment=15, ntens=1),
]
targets = [
    FitTarget(
        load=load_i,
        observations=[
            Observation(
                values=driver_factory(PythonIntegrator(true_model))
                .run(load_i)
                .stress[:, 0]
                + rng.normal(0.0, noise_std, size=len(load_i.data)),
                component=0,
            )
        ],
        name=f"amplitude-{i}",
    )
    for i, load_i in enumerate(loads_multi)
]

result_multi = fit_params(
    true_model, driver_factory, targets, fit_config,
    fixed_params=fixed_params, method="L-BFGS-B", max_iter=200,
)

print("  Multi-target (2 strain amplitudes):")
print(f"  Fitted sigma_y0 = {result_multi.params['sigma_y0']:.2f} MPa")
print(f"  Fitted H        = {result_multi.params['H']:.2f} MPa")
print(f"  Residual        = {result_multi.residual:.4f}  (mean sq. per curve)")
print()

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
if HAS_MATPLOTLIB:
    fitted_model = J2Isotropic1D(
        E=true_model.E, nu=true_model.nu,
        sigma_y0=fitted_sigma_y0, H=fitted_H,
    )
    stress_fitted = driver_factory(PythonIntegrator(fitted_model)).run(load).stress[:, 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: hysteresis loop comparison
    ax = axes[0]
    ax.scatter(strain_exp * 100, stress_exp, s=8, color="gray", alpha=0.5,
               label="Synthetic exp. data (noisy)")
    ax.plot(strain_exp * 100, stress_clean, color="black", linewidth=1.5,
            linestyle="--", label=f"True  (σ_y0={true_model.sigma_y0:.0f}, H={true_model.H:.0f})")
    ax.plot(strain_exp * 100, stress_fitted, color="steelblue", linewidth=1.5,
            label=f"Fitted (σ_y0={fitted_sigma_y0:.1f}, H={fitted_H:.1f})")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.axvline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel("Axial strain ε₁₁  [%]")
    ax.set_ylabel("Axial stress σ₁₁  [MPa]")
    ax.set_title("Fit quality — hysteresis loop")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: convergence history
    ax = axes[1]
    if result.history:
        iters = range(len(result.history))
        sy_hist = [h["sigma_y0"] for h in result.history]
        H_hist  = [h["H"]        for h in result.history]
        ax.plot(iters, sy_hist, label="σ_y0", color="steelblue")
        ax.plot(iters, H_hist,  label="H",     color="tomato")
        ax.axhline(true_model.sigma_y0, color="steelblue", linestyle="--",
                   linewidth=0.8, alpha=0.6)
        ax.axhline(true_model.H, color="tomato", linestyle="--",
                   linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Optimiser iteration")
        ax.set_ylabel("Parameter value [MPa]")
        ax.set_title("Convergence history")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No history available", ha="center", va="center",
                transform=ax.transAxes)

    fig.tight_layout()
    out = "examples/output_fit_j2.png"
    fig.savefig(out, dpi=150)
    print(f"  Plot saved → {out}")
else:
    print("  matplotlib not installed — skipping plot.")
