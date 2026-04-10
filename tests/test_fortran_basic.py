"""Basic f2py integration test for the Fortran elastic_stress subroutine.

Verifies that the Fortran implementation of isotropic elastic stress
computation agrees with the Python reference (J2IsotropicHardening.elastic_stiffness)
to floating-point precision.

The compiled module ``manforge_test_basic`` must be built before running:

    cd fortran/
    python -m numpy.f2py -c test_basic.f90 -m manforge_test_basic

If the module is not available, all tests in this file are skipped.
"""

import sys
import os

import numpy as np
import pytest

# Add fortran/ to path so the compiled .so is importable
_FORTRAN_DIR = os.path.join(os.path.dirname(__file__), "..", "fortran")
sys.path.insert(0, os.path.abspath(_FORTRAN_DIR))

mod = pytest.importorskip(
    "manforge_test_basic",
    reason="manforge_test_basic not compiled — run: "
           "cd fortran && python -m numpy.f2py -c test_basic.f90 -m manforge_test_basic",
)

import manforge  # noqa: F401 — enables JAX float64
from manforge.models.j2_isotropic import J2IsotropicHardening


@pytest.fixture
def params():
    return {"E": 210_000.0, "nu": 0.3, "sigma_y0": 250.0, "H": 1_000.0}


@pytest.fixture
def model():
    return J2IsotropicHardening()


def _py_stress(model, params, dstran):
    """Reference: Python elastic stiffness matrix-vector product."""
    import jax.numpy as jnp
    C = model.elastic_stiffness(params)
    return np.array(C @ jnp.array(dstran))


# ---------------------------------------------------------------------------
# elastic_stress — uniaxial strain increment
# ---------------------------------------------------------------------------

def test_elastic_stress_uniaxial(model, params):
    """Fortran elastic_stress matches Python C @ dstran (uniaxial)."""
    dstran = np.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])

    stress_f90 = mod.elastic_stress(params["E"], params["nu"], dstran)
    stress_py  = _py_stress(model, params, dstran)

    np.testing.assert_allclose(stress_f90, stress_py, rtol=1e-12,
                               err_msg="Uniaxial: Fortran vs Python mismatch")


# ---------------------------------------------------------------------------
# elastic_stress — multiaxial strain increment
# ---------------------------------------------------------------------------

def test_elastic_stress_multiaxial(model, params):
    """Fortran elastic_stress matches Python C @ dstran (multiaxial)."""
    dstran = np.array([1.5e-3, -0.5e-3, -0.5e-3, 0.5e-3, 0.0, 0.0])

    stress_f90 = mod.elastic_stress(params["E"], params["nu"], dstran)
    stress_py  = _py_stress(model, params, dstran)

    np.testing.assert_allclose(stress_f90, stress_py, rtol=1e-12,
                               err_msg="Multiaxial: Fortran vs Python mismatch")


# ---------------------------------------------------------------------------
# elastic_stiffness — diagonal entries
# ---------------------------------------------------------------------------

def test_elastic_stiffness_diagonal(params):
    """Fortran stiffness diagonal: normal entries = lam+2mu, shear = mu."""
    E, nu = params["E"], params["nu"]
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

def test_elastic_stiffness_symmetric(params):
    """Fortran stiffness matrix is symmetric."""
    C_f90 = mod.elastic_stiffness(params["E"], params["nu"], 6)
    np.testing.assert_allclose(C_f90, C_f90.T, atol=1e-12,
                               err_msg="Stiffness matrix not symmetric")
