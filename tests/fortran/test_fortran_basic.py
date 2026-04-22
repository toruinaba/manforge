"""Basic f2py integration test for the Fortran elastic_stress subroutine.

Verifies that the Fortran implementation of isotropic elastic stress
computation agrees with the Python reference (J2Isotropic3D.elastic_stiffness)
to floating-point precision.

The compiled module ``manforge_test_basic`` must be built before running:

    cd fortran/
    python -m numpy.f2py -c test_basic.f90 -m manforge_test_basic

If the module is not available, all tests in this file are skipped.
"""

import numpy as np
import pytest

mod = pytest.importorskip(
    "manforge_test_basic",
    reason="manforge_test_basic not compiled — run: "
           "cd fortran && python -m numpy.f2py -c test_basic.f90 -m manforge_test_basic",
)

pytestmark = pytest.mark.fortran


def _py_stress(model, dstran):
    """Reference: Python elastic stiffness matrix-vector product."""
    import autograd.numpy as anp
    C = model.elastic_stiffness()
    return np.array(C @ anp.array(dstran))


# ---------------------------------------------------------------------------
# elastic_stress — uniaxial strain increment
# ---------------------------------------------------------------------------

def test_elastic_stress_uniaxial(model):
    """Fortran elastic_stress matches Python C @ dstran (uniaxial)."""
    dstran = np.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])

    stress_f90 = mod.elastic_stress(model.E, model.nu, dstran)
    stress_py  = _py_stress(model, dstran)

    np.testing.assert_allclose(stress_f90, stress_py, rtol=1e-12,
                               err_msg="Uniaxial: Fortran vs Python mismatch")


# ---------------------------------------------------------------------------
# elastic_stress — multiaxial strain increment
# ---------------------------------------------------------------------------

def test_elastic_stress_multiaxial(model):
    """Fortran elastic_stress matches Python C @ dstran (multiaxial)."""
    dstran = np.array([1.5e-3, -0.5e-3, -0.5e-3, 0.5e-3, 0.0, 0.0])

    stress_f90 = mod.elastic_stress(model.E, model.nu, dstran)
    stress_py  = _py_stress(model, dstran)

    np.testing.assert_allclose(stress_f90, stress_py, rtol=1e-12,
                               err_msg="Multiaxial: Fortran vs Python mismatch")


# ---------------------------------------------------------------------------
# elastic_stiffness — diagonal entries
# ---------------------------------------------------------------------------

def test_elastic_stiffness_diagonal(model):
    """Fortran stiffness diagonal: normal entries = lam+2mu, shear = mu."""
    E, nu = model.E, model.nu
    mu  = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    C_f90 = mod.elastic_stiffness(E, nu, 6)

    # Normal diagonal (indices 0,1,2 in Python / 1,2,3 in Fortran)
    for i in range(3):
        np.testing.assert_allclose(C_f90[i, i], lam + 2.0 * mu, rtol=1e-12,
                                   err_msg=f"Normal diagonal C[{i},{i}]")

    # Shear diagonal (indices 3,4,5)
    for i in range(3, 6):
        np.testing.assert_allclose(C_f90[i, i], mu, rtol=1e-12,
                                   err_msg=f"Shear diagonal C[{i},{i}]")


# ---------------------------------------------------------------------------
# elastic_stiffness — symmetry
# ---------------------------------------------------------------------------

def test_elastic_stiffness_symmetric(model):
    """Fortran stiffness matrix is symmetric."""
    C_f90 = mod.elastic_stiffness(model.E, model.nu, 6)
    np.testing.assert_allclose(C_f90, C_f90.T, atol=1e-12,
                               err_msg="Stiffness matrix not symmetric")
