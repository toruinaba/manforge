"""Uniaxial tension with a Fortran UMAT via FortranIntegrator.

Demonstrates the production workflow:
  1. Define a Python model (J2Isotropic3D).
  2. Load a compiled Fortran UMAT with FortranUMAT.
  3. Build a FortranIntegrator with from_model() — no manual param_fn needed.
  4. Drive both Python and Fortran integrators with StrainDriver and compare.

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
from manforge.simulation import FortranIntegrator, StrainDriver
from manforge.simulation.integrator import PythonIntegrator
from manforge.simulation.types import FieldHistory, FieldType
from manforge.verification import FortranUMAT

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
try:
    fortran = FortranUMAT("j2_isotropic_3d")
except ModuleNotFoundError:
    raise SystemExit(
        "Fortran module not found.  Build it first:\n"
        "  uv run manforge build fortran/abaqus_stubs.f90 "
        "fortran/j2_isotropic_3d.f90 --name j2_isotropic_3d\n"
        "  uv run manforge list   (to verify)"
    )

# ---------------------------------------------------------------------------
# Integrators
#   from_model() auto-fills param_fn, state_names, initial_state,
#   elastic_stiffness, and stress_state from model attributes.
#   Pass param_fn explicitly only when the Fortran argument order differs.
# ---------------------------------------------------------------------------
py_int = PythonIntegrator(model)
fc_int = FortranIntegrator.from_model(fortran, "j2_isotropic_3d", model)

# ---------------------------------------------------------------------------
# Strain loading history (uniaxial tension, σ11 direction)
# ---------------------------------------------------------------------------
N = 100
strain_history = np.zeros((N, model.ntens))
strain_history[:, 0] = np.linspace(0.0, 5e-3, N)  # cumulative ε11
load = FieldHistory(FieldType.STRAIN, "eps", strain_history)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
result_py = StrainDriver(py_int).run(load)
result_fc = StrainDriver(fc_int).run(load)

stress_py = result_py.stress[:, 0]   # σ11 (Python)
stress_fc = result_fc.stress[:, 0]   # σ11 (Fortran)

# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------
max_diff = float(np.max(np.abs(stress_py - stress_fc)))
max_ref  = float(np.max(np.abs(stress_py)))
max_rel  = max_diff / max_ref if max_ref > 0 else 0.0

print("=" * 55)
print("  Uniaxial Tension — Python vs Fortran UMAT")
print("=" * 55)
print(f"  Model        : J2Isotropic3D  (SOLID_3D, ntens=6)")
print(f"  E            = {model.E:.0f} MPa")
print(f"  nu           = {model.nu}")
print(f"  sigma_y0     = {model.sigma_y0:.1f} MPa")
print(f"  H            = {model.H:.0f} MPa")
print(f"  Steps        : {N}")
print(f"  Strain range : 0 → {strain_history[-1, 0]:.4f}")
print()
print(f"  Final σ11 (Python) : {stress_py[-1]:.4f} MPa")
print(f"  Final σ11 (Fortran): {stress_fc[-1]:.4f} MPa")
print(f"  Max abs diff       : {max_diff:.2e} MPa")
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
    eps_axis = strain_history[:, 0] * 100  # % strain

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(eps_axis, stress_py, color="steelblue", linewidth=2, label="Python (autodiff)")
    ax.plot(eps_axis, stress_fc, color="tomato",    linewidth=1.5, linestyle="--",
            label="Fortran UMAT")
    ax.axhline(model.sigma_y0, color="gray", linestyle=":", linewidth=1,
               label=f"σ_y0 = {model.sigma_y0:.0f} MPa")
    ax.set_xlabel("Axial strain ε₁₁  [%]")
    ax.set_ylabel("Axial stress σ₁₁  [MPa]")
    ax.set_title("Stress-strain curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(eps_axis, np.abs(stress_py - stress_fc), color="darkorange", linewidth=1.5)
    ax2.set_xlabel("Axial strain ε₁₁  [%]")
    ax2.set_ylabel("|σ_py − σ_f|  [MPa]")
    ax2.set_title("Absolute stress difference")
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    fig.suptitle("FortranIntegrator (j2_isotropic_3d) — Python vs UMAT", fontsize=12)
    fig.tight_layout()

    out = "examples/output_fortran_uniaxial.png"
    fig.savefig(out, dpi=150)
    print(f"  Plot saved → {out}")
else:
    print("  matplotlib not installed — skipping plot.")
