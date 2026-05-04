"""Uniaxial stress state with a Fortran UMAT via FortranIntegrator.

Demonstrates the production workflow:
  1. Define a Python model (J2Isotropic3D).
  2. Load a compiled Fortran UMAT with FortranModule.
  3. Build a FortranIntegrator with from_model() — no manual param_fn needed.
  4. Drive both Python and Fortran integrators with StressDriver and compare strains.

σ11 is prescribed from 0 to 1.5 σ_y0 (yielding at σ_y0).  All other stress
components are zero; the lateral strains (ε22, ε33) are determined by the
Newton–Raphson loop inside StressDriver.

Usage
-----
    # Build the Fortran module first (requires gfortran):
    uv run manforge build fortran/abaqus_stubs.f90 fortran/j2_isotropic_3d.f90 \\
        --name j2_isotropic_3d

    uv run python examples/example_fortran_uniaxial.py
"""

import sys
import os

import numpy as np

_fortran_dir = os.path.join(os.path.dirname(__file__), "..", "fortran")
sys.path.insert(0, os.path.abspath(_fortran_dir))

import manforge  # noqa: F401 — enables JAX float64
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.simulation import FortranIntegrator, StressDriver
from manforge.simulation.integrator import PythonIntegrator
from manforge.simulation.types import FieldHistory, FieldType
from manforge.verification import FortranModule

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
model = J2Isotropic3D(
    E=210_000.0,    # Young's modulus [MPa]
    nu=0.3,         # Poisson's ratio
    sigma_y0=250.0, # Initial yield stress [MPa]
    H=1_000.0,      # Linear isotropic hardening modulus [MPa]
)

# ---------------------------------------------------------------------------
# Load compiled Fortran UMAT
# ---------------------------------------------------------------------------
fortran = FortranModule("j2_isotropic_3d")

# ---------------------------------------------------------------------------
# Integrators
#   from_model() auto-fills param_fn, state_names, initial_state,
#   elastic_stiffness, and stress_state from model attributes.
#   Pass param_fn explicitly only when the Fortran argument order differs.
# ---------------------------------------------------------------------------
py_int = PythonIntegrator(model)
fc_int = FortranIntegrator.from_model(fortran, "j2_isotropic_3d", model)

# ---------------------------------------------------------------------------
# Stress loading history (uniaxial: σ11 prescribed, other components = 0)
# ---------------------------------------------------------------------------
N = 100
stress_data = np.zeros((N, model.ntens))
stress_data[:, 0] = np.linspace(0.0, 1.5 * model.sigma_y0, N)  # σ11 target
load = FieldHistory(FieldType.STRESS, "sigma", stress_data)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
result_py = StressDriver(py_int).run(load)
result_fc = StressDriver(fc_int).run(load)

strain_py = result_py.strain[:, 0]   # ε11 (Python)
strain_fc = result_fc.strain[:, 0]   # ε11 (Fortran)

# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------
max_diff = float(np.max(np.abs(strain_py - strain_fc)))
max_ref  = float(np.max(np.abs(strain_py)))
max_rel  = max_diff / max_ref if max_ref > 0 else 0.0

print("=" * 55)
print("  Uniaxial Stress State — Python vs Fortran UMAT")
print("=" * 55)
print(f"  Model        : J2Isotropic3D  (SOLID_3D, ntens=6)")
print(f"  E            = {model.E:.0f} MPa")
print(f"  nu           = {model.nu}")
print(f"  sigma_y0     = {model.sigma_y0:.1f} MPa")
print(f"  H            = {model.H:.0f} MPa")
print(f"  Steps        : {N}")
print(f"  σ11 range    : 0 → {stress_data[-1, 0]:.1f} MPa")
print()
print(f"  Final ε11 (Python) : {strain_py[-1]:.6f}")
print(f"  Final ε11 (Fortran): {strain_fc[-1]:.6f}")
print(f"  Max abs diff       : {max_diff:.2e}")
print(f"  Max rel diff       : {max_rel:.2e}")
print()

if max_rel < 1e-5:
    print("  [PASS] Python and Fortran results agree within 1e-5.")
else:
    print(f"  [WARN] Max rel diff {max_rel:.2e} exceeds 1e-5.")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
if HAS_MATPLOTLIB:
    sigma_axis = stress_data[:, 0]  # σ11 [MPa]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(strain_py * 100, sigma_axis, color="steelblue", linewidth=2, label="Python (autodiff)")
    ax.plot(strain_fc * 100, sigma_axis, color="tomato",    linewidth=1.5, linestyle="--",
            label="Fortran UMAT")
    ax.axhline(model.sigma_y0, color="gray", linestyle=":", linewidth=1,
               label=f"σ_y0 = {model.sigma_y0:.0f} MPa")
    ax.set_xlabel("Axial strain ε₁₁  [%]")
    ax.set_ylabel("Axial stress σ₁₁  [MPa]")
    ax.set_title("Stress-strain curve (stress-controlled)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(sigma_axis, np.abs(strain_py - strain_fc), color="darkorange", linewidth=1.5)
    ax2.set_xlabel("Axial stress σ₁₁  [MPa]")
    ax2.set_ylabel("|ε₁₁_py − ε₁₁_fc|")
    ax2.set_title("Absolute strain difference")
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    fig.suptitle("FortranIntegrator (j2_isotropic_3d) — Python vs UMAT", fontsize=12)
    fig.tight_layout()

    out = "examples/output_fortran_uniaxial.png"
    fig.savefig(out, dpi=150)
    print(f"  Plot saved → {out}")
else:
    print("  matplotlib not installed — skipping plot.")
