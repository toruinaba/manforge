"""Uniaxial tension simulation with J2 isotropic hardening.

Demonstrates:
- Defining material parameters for J2IsotropicHardening
- Running a uniaxial strain history with UniaxialDriver
- Verifying the consistent tangent with check_tangent
- Plotting the stress-strain curve (saved as PNG)

Usage
-----
    uv run python examples/example_j2_uniaxial.py
"""

import numpy as np

import manforge  # noqa: F401 — enables JAX float64
from manforge.models.j2_isotropic import J2IsotropicHardening
from manforge.simulation.driver import UniaxialDriver
from manforge.verification.fd_check import check_tangent

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Material parameters (steel, MPa units)
# ---------------------------------------------------------------------------
params = {
    "E": 210_000.0,   # Young's modulus [MPa]
    "nu": 0.3,        # Poisson's ratio
    "sigma_y0": 250.0, # Initial yield stress [MPa]
    "H": 1_000.0,     # Linear isotropic hardening modulus [MPa]
}

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
model = J2IsotropicHardening()
driver = UniaxialDriver()

N = 100
strain_history = np.linspace(0.0, 5e-3, N)   # cumulative ε11
stress_history = driver.run(model, strain_history, params)

# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------
sigma_y0 = params["sigma_y0"]
sigma_final = stress_history[-1]
eps_yield_approx = sigma_y0 / params["E"]

print("=" * 50)
print("  J2 Uniaxial Tension Simulation")
print("=" * 50)
print(f"  E          = {params['E']:.0f} MPa")
print(f"  nu         = {params['nu']}")
print(f"  sigma_y0   = {params['sigma_y0']:.1f} MPa")
print(f"  H          = {params['H']:.0f} MPa")
print(f"  Strain range: 0 → {strain_history[-1]:.4f}")
print(f"  Approx. yield strain: {eps_yield_approx:.5f}")
print(f"  Final stress (sigma11): {sigma_final:.2f} MPa")
print()

# ---------------------------------------------------------------------------
# Tangent verification
# ---------------------------------------------------------------------------
print("Tangent check — elastic domain:")
result_e = check_tangent(
    model,
    jnp.zeros(6),
    model.initial_state(),
    params,
    jnp.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0]),
)
status = "PASS" if result_e.passed else "FAIL"
print(f"  [{status}]  max rel err = {result_e.max_rel_err:.2e}")

print("Tangent check — plastic domain (uniaxial):")
result_p = check_tangent(
    model,
    jnp.zeros(6),
    model.initial_state(),
    params,
    jnp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0]),
)
status = "PASS" if result_p.passed else "FAIL"
print(f"  [{status}]  max rel err = {result_p.max_rel_err:.2e}")
print()

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
if HAS_MATPLOTLIB:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(strain_history * 100, stress_history, color="steelblue", linewidth=2,
            label="J2 isotropic hardening")
    ax.axhline(sigma_y0, color="gray", linestyle="--", linewidth=1,
               label=f"$\\sigma_{{y0}}$ = {sigma_y0:.0f} MPa")
    ax.set_xlabel("Axial strain ε₁₁  [%]")
    ax.set_ylabel("Axial stress σ₁₁  [MPa]")
    ax.set_title("Uniaxial Tension — J2 Isotropic Hardening")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = "examples/output_j2_uniaxial.png"
    fig.savefig(out, dpi=150)
    print(f"  Plot saved → {out}")
else:
    print("  matplotlib not installed — skipping plot.")
