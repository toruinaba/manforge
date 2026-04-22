"""Registry cross-validation: J2 Python methods vs Fortran subroutines.

Requires the compiled j2_isotropic_3d module:

    make fortran-build-umat

If the module is not available, all tests in this file are skipped.
"""

import pytest

pytest.importorskip(
    "j2_isotropic_3d",
    reason="j2_isotropic_3d not compiled -- run: make fortran-build-umat",
)

pytestmark = pytest.mark.fortran

from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.verification import FortranUMAT, check_bindings


def test_elastic_stiffness_binding_registered():
    bindings = J2Isotropic3D._fortran_bindings
    assert "elastic_stiffness" in bindings
    assert bindings["elastic_stiffness"].subroutine == "j2_isotropic_3d_elastic_stiffness"


def test_check_bindings_elastic_stiffness(model):
    fortran = FortranUMAT("j2_isotropic_3d")
    cases = {
        "elastic_stiffness": ((), (model.E, model.nu)),
    }
    results = check_bindings(model, fortran, cases, rtol=1e-10)

    ok, max_err = results["elastic_stiffness"]
    assert ok, f"elastic_stiffness mismatch: max_rel_err={max_err:.2e}"
    assert max_err < 1e-10
