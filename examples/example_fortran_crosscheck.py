"""Fortran UMAT crosscheck: build → load → crosscheck → drive.

The complete Fortran validation workflow (README § 5.7):

1. Load a compiled UMAT with FortranModule
2. Crosscheck Python vs Fortran over a generated strain history
   (CrosscheckStrainDriver — per-step stress/ddsdde comparison)
3. Debug a failing step via iter_run + JacobianChecker.compare
4. Drive the Fortran UMAT directly with StressDriver (cyclic hysteresis, plot)

In an external project, replace J2Isotropic3D with your model and
"j2_isotropic_3d" with your compiled module/subroutine names.
FortranIntegrator.from_model() fills param_fn / state layout / dimension from
the model; pass param_fn= explicitly only if your Fortran argument order
differs from model.param_names.

Requires a compiled Fortran module (gfortran):

    uv run manforge build fortran/abaqus_stubs.f90 fortran/j2_isotropic_3d.f90 \\
        --name j2_isotropic_3d
    uv run python examples/example_fortran_crosscheck.py
"""

import os
import sys

import numpy as np

_fortran_dir = os.path.join(os.path.dirname(__file__), "..", "fortran")
sys.path.insert(0, os.path.abspath(_fortran_dir))

import manforge  # noqa: F401 — enables float64
from manforge.models import J2Isotropic3D
from manforge.simulation import (
    FortranIntegrator,
    PythonIntegrator,
    PythonNumericalIntegrator,
    StressDriver,
)
from manforge.simulation.types import FieldHistory, FieldType
from manforge.verification import (
    CrosscheckStrainDriver,
    FortranModule,
    JacobianChecker,
    generate_strain_history,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ---------------------------------------------------------------------------
# 1. Model + compiled Fortran module
# ---------------------------------------------------------------------------
model = J2Isotropic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)
fortran = FortranModule("j2_isotropic_3d")
fc_int = FortranIntegrator.from_model(fortran, "j2_isotropic_3d", model)

# ---------------------------------------------------------------------------
# 2. Crosscheck over a generated multi-pattern strain history
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Crosscheck: Python (numerical NR) vs Fortran UMAT")
print("=" * 60)

load = FieldHistory(FieldType.STRAIN, "eps", generate_strain_history(model))
cc = CrosscheckStrainDriver(PythonNumericalIntegrator(model), fc_int)
result = cc.run(load)

print(f"  passed             : {result.passed}")
print(f"  n_cases            : {result.n_cases}  (passed: {result.n_passed})")
print(f"  max stress rel err : {result.max_stress_rel_err:.2e}")
assert result.passed

# ---------------------------------------------------------------------------
# 3. Debugging pattern: stream steps, inspect Jacobian blocks on failure
# ---------------------------------------------------------------------------
checker = JacobianChecker(model)
for cr in cc.iter_run(load):
    if not cr.passed:
        cmp = checker.compare(cr.result_a, cr.result_b, cr.state_n)
        print(f"  step failed — block comparison:\n{cmp.blocks}")
        break
else:
    print("  iter_run: no failing step (as expected)")
print()

# ---------------------------------------------------------------------------
# 4. Drive the Fortran UMAT with StressDriver: cyclic ±1.5 sigma_y0
#    Lateral strains are solved by the outer NR — a uniaxial stress test
#    from the 3D UMAT.
# ---------------------------------------------------------------------------
print("=" * 60)
print("  StressDriver: cyclic hysteresis, Python vs Fortran")
print("=" * 60)

sigma_max = 1.5 * model.sigma_y0
cyc = FieldHistory.cyclic_stress(
    peaks=[sigma_max, -sigma_max, sigma_max, -sigma_max],
    n_per_segment=25, ntens=model.ntens)

strain_py = StressDriver(PythonIntegrator(model)).run(cyc).strain[:, 0]
strain_fc = StressDriver(fc_int).run(cyc).strain[:, 0]

max_rel = float(np.max(np.abs(strain_py - strain_fc)) / np.max(np.abs(strain_py)))
print(f"  Final eps11 (Python) : {strain_py[-1]:.6f}")
print(f"  Final eps11 (Fortran): {strain_fc[-1]:.6f}")
print(f"  Max rel diff         : {max_rel:.2e}")
assert max_rel < 1e-5

if HAS_MATPLOTLIB:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(strain_py * 100, cyc.data[:, 0], color="steelblue", linewidth=2,
            label="Python (autodiff)")
    ax.plot(strain_fc * 100, cyc.data[:, 0], color="tomato", linewidth=1.5,
            linestyle="--", label="Fortran UMAT")
    ax.axhline(model.sigma_y0, color="gray", linestyle=":", linewidth=1)
    ax.axhline(-model.sigma_y0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Axial strain ε₁₁  [%]")
    ax.set_ylabel("Axial stress σ₁₁  [MPa]")
    ax.set_title("Cyclic hysteresis — Python vs Fortran UMAT")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "output_fortran_crosscheck.png")
    fig.savefig(out, dpi=150)
    print(f"  Plot saved → {out}")

print()
print("All crosschecks passed.")
