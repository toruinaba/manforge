"""Integration tests for Fortran binding verification (verification/fortran_check.py).

Covers:
- FortranModule raises ModuleNotFoundError for unknown module name
- @verified_against_fortran registry and check_bindings (requires compiled .so)

Absorbed from:
- tests/integration/test_verification.py (FortranModule bad-module test)
- tests/fortran/test_j2_bindings.py
"""

import pytest

from returnmap.simulation import FortranModule


# ---------------------------------------------------------------------------
# FortranModule error path (no Fortran required)
# ---------------------------------------------------------------------------

def test_fortran_umat_bad_module():
    """FortranModule raises ModuleNotFoundError for an unknown module name."""
    with pytest.raises(ModuleNotFoundError):
        FortranModule("nonexistent_umat_module_xyz")


# ---------------------------------------------------------------------------
# @verified_against_fortran registry + check_bindings (requires .so)
# ---------------------------------------------------------------------------

@pytest.mark.fortran
def test_elastic_stiffness_binding_registered():
    pytest.importorskip(
        "j2_isotropic_3d",
        reason="j2_isotropic_3d not compiled -- run: uv run returnmap build fortran/abaqus_stubs.f90 src/returnmap/models/j2_isotropic/fortran/j2_isotropic_3d.f90 --name j2_isotropic_3d",
    )
    from returnmap.models import J2Isotropic3D
    bindings = J2Isotropic3D._fortran_bindings
    assert "elastic_stiffness" in bindings
    assert bindings["elastic_stiffness"].subroutine == "j2_isotropic_3d_elastic_stiffness"


@pytest.mark.fortran
def test_check_bindings_elastic_stiffness(model):
    pytest.importorskip(
        "j2_isotropic_3d",
        reason="j2_isotropic_3d not compiled -- run: uv run returnmap build fortran/abaqus_stubs.f90 src/returnmap/models/j2_isotropic/fortran/j2_isotropic_3d.f90 --name j2_isotropic_3d",
    )
    from returnmap.verification import check_bindings
    fortran = FortranModule("j2_isotropic_3d")
    cases = {
        "elastic_stiffness": ((model.initial_state(),), (model.E, model.nu)),
    }
    report = check_bindings(model, fortran, cases, tol=1e-10)
    assert report.passed, report.summary()
    assert report.checks["elastic_stiffness"].rel_err <= 1e-10
    assert report.not_checked == []
