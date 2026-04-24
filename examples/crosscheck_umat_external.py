"""External user example: crosscheck_umat with a custom model and UMAT.

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
    - FortranUMAT("j2_isotropic_3d")  →  FortranUMAT("my_umat")
    - umat_subroutine="j2_isotropic_3d"  →  "my_umat_core"
    - param_fn=...  →  the parameter order of their own Fortran subroutine
"""

import sys
import os
import numpy as np

# Make compiled Fortran modules importable when running from the repo root.
# In an installed package the user would add their own build output to sys.path.
_fortran_dir = os.path.join(os.path.dirname(__file__), "..", "fortran")
sys.path.insert(0, os.path.abspath(_fortran_dir))

from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.simulation import StrainDriver
from manforge.simulation.types import FieldHistory, FieldType
from manforge.verification import FortranUMAT, crosscheck_umat, generate_strain_history

# ---------------------------------------------------------------------------
# 1. Define (or import) your Python model
#    Replace this with: from my_package.models import MyModel
# ---------------------------------------------------------------------------
model = J2Isotropic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)

# ---------------------------------------------------------------------------
# 2. Load your compiled Fortran UMAT module
#    Replace "j2_isotropic_3d" with the --name you passed to manforge build
# ---------------------------------------------------------------------------
try:
    fortran = FortranUMAT("j2_isotropic_3d")
except ModuleNotFoundError:
    raise SystemExit(
        "Fortran module not found.  Compile it first:\n"
        "  uv run manforge build fortran/abaqus_stubs.f90 "
        "fortran/j2_isotropic_3d.f90 --name j2_isotropic_3d"
    )

# ---------------------------------------------------------------------------
# 3. Build a strain loading history
#    generate_strain_history produces a 35-step tension-unload-compression
#    sequence tuned to the model's yield strain.
# ---------------------------------------------------------------------------
strain_history = generate_strain_history(model)
load = FieldHistory(FieldType.STRAIN, "eps", strain_history)

# ---------------------------------------------------------------------------
# 4. Run the crosscheck
#    param_fn maps your Python model attributes to the argument order that
#    your Fortran subroutine expects.
# ---------------------------------------------------------------------------
result = crosscheck_umat(
    StrainDriver(),
    model,
    fortran,
    umat_subroutine="j2_isotropic_3d",   # f2py-callable core routine name
    load=load,
    param_fn=lambda m: (m.E, m.nu, m.sigma_y0, m.H),
    # state_to_args and parse_umat_return use the default (state_names-order)
    # convention, which works when the Fortran subroutine returns state vars
    # in the same order as model.state_names.
)

# ---------------------------------------------------------------------------
# 5. Inspect the result
# ---------------------------------------------------------------------------
print(f"passed             : {result.passed}")
print(f"max stress rel err : {result.max_stress_rel_err:.2e}")
print(f"stress_py shape    : {result.stress_py.shape}")
print(f"stress_f  shape    : {result.stress_f.shape}")

assert result.passed, (
    f"Crosscheck failed: max_stress_rel_err = {result.max_stress_rel_err:.2e}"
)
