"""External user example: CrosscheckStrainDriver with a custom model and UMAT.

This script demonstrates how an external user (pip-installed manforge) would
validate their own Python constitutive model against a compiled Fortran UMAT.

The example uses the library-internal J2 model and UMAT as stand-ins for
"MyModel" and "my_umat_core", so it can be run from the manforge repo once
the UMAT is compiled:

    uv run manforge build fortran/abaqus_stubs.f90 fortran/j2_isotropic_3d.f90 \
        --name j2_isotropic_3d
    uv run python examples/crosscheck_umat_external.py

In a real external project the user would replace:
    - J2Isotropic3D  →  MyModel (their Python model)
    - FortranModule("j2_isotropic_3d")  →  FortranModule("my_umat")
    - subroutine "j2_isotropic_3d"  →  "my_umat_core"
    - (optional) param_fn=...  →  only when Fortran argument order differs from
      model.param_names.  Omit if they match (the convention).
"""

import sys
import os
import numpy as np

_fortran_dir = os.path.join(os.path.dirname(__file__), "..", "fortran")
sys.path.insert(0, os.path.abspath(_fortran_dir))

from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.simulation import PythonNumericalIntegrator, FortranIntegrator
from manforge.simulation.types import FieldHistory, FieldType
from manforge.verification import FortranModule, CrosscheckStrainDriver, generate_strain_history

# ---------------------------------------------------------------------------
# 1. Define (or import) your Python model
# ---------------------------------------------------------------------------
model = J2Isotropic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)

# ---------------------------------------------------------------------------
# 2. Load your compiled Fortran UMAT module
# ---------------------------------------------------------------------------
fortran = FortranModule("j2_isotropic_3d")

# ---------------------------------------------------------------------------
# 3. Build a strain loading history
# ---------------------------------------------------------------------------
strain_history = generate_strain_history(model)
load = FieldHistory(FieldType.STRAIN, "eps", strain_history)

# ---------------------------------------------------------------------------
# 4. Create the two integrators and the crosscheck harness
#    param_fn maps your Python model attributes to the argument order that
#    your Fortran subroutine expects.  It takes no arguments — capture model
#    in a closure.
#
#    method: "numerical_newton" | "user_defined" | "auto"
# ---------------------------------------------------------------------------
py_int = PythonNumericalIntegrator(model)
# from_model() fills param_fn, state_names, initial_state, and dimension from
# model attributes.  Pass param_fn= explicitly only when your Fortran subroutine
# uses a different argument order than model.param_names.
fc_int = FortranIntegrator.from_model(fortran, "j2_isotropic_3d", model)

cc = CrosscheckStrainDriver(py_int, fc_int)

# ---------------------------------------------------------------------------
# 5. Run and inspect the result
# ---------------------------------------------------------------------------
result = cc.run(load)

print(f"passed             : {result.passed}")
print(f"n_cases            : {result.n_cases}")
print(f"n_passed           : {result.n_passed}")
print(f"max stress rel err : {result.max_stress_rel_err:.2e}")

assert result.passed, (
    f"Crosscheck failed: max_stress_rel_err = {result.max_stress_rel_err:.2e}"
)
