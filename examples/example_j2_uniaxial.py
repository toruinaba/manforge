"""Cyclic uniaxial simulation with J2 isotropic hardening.

Demonstrates:
- Defining a J2Isotropic1D model with constructor parameters
- Building a cyclic strain history with FieldHistory.cyclic_strain
- Running a cyclic strain history with StrainDriver (hysteresis loop)
- Verifying the consistent tangent with check_tangent
- Plotting the stress-strain hysteresis loop (saved as PNG)

Usage
-----
    uv run python examples/example_j2_uniaxial.py
"""

import numpy as np

import manforge  # noqa: F401 — enables JAX float64
from manforge.models.j2_isotropic import J2Isotropic1D
from manforge.simulation.driver import StrainDriver
from manforge.simulation.integrator import PythonIntegrator
from manforge.simulation.types import FieldHistory, FieldType
from manforge.verification import check_tangent

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ---------------------------------------------------------------------------
# Model (steel, MPa units)
# ---------------------------------------------------------------------------
model = J2Isotropic1D(
    E=210_000.0,    # Young's modulus [MPa]
    nu=0.3,         # Poisson's ratio
    sigma_y0=250.0, # Initial yield stress [MPa]
    H=1_000.0,      # Linear isotropic hardening modulus [MPa]
)

# ---------------------------------------------------------------------------
# Cyclic simulation: 4 half-cycles (0 → +5e-3 → -5e-3 → +5e-3 → -5e-3)
# ---------------------------------------------------------------------------
load = FieldHistory.cyclic_strain(
    peaks=[5e-3, -5e-3, 5e-3, -5e-3],
    n_per_segment=25,
    ntens=1,
)
result = StrainDriver(PythonIntegrator(model)).run(load)
strain_history = result.strain[:, 0]   # ε11
stress_history = result.stress[:, 0]   # σ11

# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------
eps_yield_approx = model.sigma_y0 / model.E
plastic_steps = sum(1 for r in result.step_results if r.is_plastic)

print("=" * 50)
print("  J2 Cyclic Uniaxial Simulation")
print("=" * 50)
print(f"  E          = {model.E:.0f} MPa")
print(f"  nu         = {model.nu}")
print(f"  sigma_y0   = {model.sigma_y0:.1f} MPa")
print(f"  H          = {model.H:.0f} MPa")
print(f"  Strain peaks: ±{5e-3:.4f}  (4 half-cycles, 25 steps each)")
print(f"  Approx. yield strain: {eps_yield_approx:.5f}")
print(f"  Total steps: {len(result.step_results)}  (plastic: {plastic_steps})")
print(f"  Max stress: {stress_history.max():.2f} MPa")
print()

# ---------------------------------------------------------------------------
# Tangent verification
# ---------------------------------------------------------------------------
print("Tangent check — elastic domain:")
result_e = check_tangent(
    PythonIntegrator(model),
    np.zeros(1),
    model.initial_state(),
    np.array([1e-5]),
)
status = "PASS" if result_e.passed else "FAIL"
print(f"  [{status}]  max rel err = {result_e.max_rel_err:.2e}")

print("Tangent check — plastic domain (uniaxial):")
result_p = check_tangent(
    PythonIntegrator(model),
    np.zeros(1),
    model.initial_state(),
    np.array([2e-3]),
)
status = "PASS" if result_p.passed else "FAIL"
print(f"  [{status}]  max rel err = {result_p.max_rel_err:.2e}")
print()

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
if HAS_MATPLOTLIB:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(strain_history * 100, stress_history, color="steelblue", linewidth=1.5,
            label="J2 isotropic hardening")
    ax.axhline( model.sigma_y0, color="gray", linestyle="--", linewidth=1,
               label=f"$\\sigma_{{y0}}$ = {model.sigma_y0:.0f} MPa")
    ax.axhline(-model.sigma_y0, color="gray", linestyle="--", linewidth=1)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.axvline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel("Axial strain ε₁₁  [%]")
    ax.set_ylabel("Axial stress σ₁₁  [MPa]")
    ax.set_title("Cyclic Uniaxial — J2 Isotropic Hardening (hysteresis)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = "examples/output_j2_uniaxial.png"
    fig.savefig(out, dpi=150)
    print(f"  Plot saved → {out}")
else:
    print("  matplotlib not installed — skipping plot.")
