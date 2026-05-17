"""Integration tests for the Fortran module f2py wrapper (FortranModule sanity).

Absorbed from tests/fortran/test_fortran_basic.py.

Tests that the basic Fortran elastic_stress / elastic_stiffness subroutines
(compiled from fortran/test_basic.f90 as module ``manforge_test_basic``) agree
with the Python reference to floating-point precision.

Build before running:
    uv run manforge build fortran/test_basic.f90 --name manforge_test_basic
"""

import numpy as np
import pytest

mod = pytest.importorskip(
    "manforge_test_basic",
    reason="manforge_test_basic not compiled — run: "
           "uv run manforge build fortran/test_basic.f90 --name manforge_test_basic",
)

pytestmark = pytest.mark.fortran


def _py_stress(model, dstran):
    import autograd.numpy as anp
    C = model.elastic_stiffness()
    return np.array(C @ anp.array(dstran))


def test_elastic_stress_uniaxial(model):
    """Fortran elastic_stress matches Python C @ dstran (uniaxial)."""
    dstran = np.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    stress_f90 = mod.elastic_stress(model.E, model.nu, dstran)
    stress_py = _py_stress(model, dstran)
    np.testing.assert_allclose(stress_f90, stress_py, rtol=1e-12)


def test_elastic_stress_multiaxial(model):
    """Fortran elastic_stress matches Python C @ dstran (multiaxial)."""
    dstran = np.array([1.5e-3, -0.5e-3, -0.5e-3, 0.5e-3, 0.0, 0.0])
    stress_f90 = mod.elastic_stress(model.E, model.nu, dstran)
    stress_py = _py_stress(model, dstran)
    np.testing.assert_allclose(stress_f90, stress_py, rtol=1e-12)


def test_elastic_stiffness_diagonal(model):
    """Fortran stiffness diagonal: normal = lam+2mu, shear = mu."""
    E, nu = model.E, model.nu
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    C_f90 = mod.elastic_stiffness(E, nu, 6)
    for i in range(3):
        np.testing.assert_allclose(C_f90[i, i], lam + 2.0 * mu, rtol=1e-12)
    for i in range(3, 6):
        np.testing.assert_allclose(C_f90[i, i], mu, rtol=1e-12)


def test_elastic_stiffness_symmetric(model):
    """Fortran stiffness matrix is symmetric."""
    C_f90 = mod.elastic_stiffness(model.E, model.nu, 6)
    np.testing.assert_allclose(C_f90, C_f90.T, atol=1e-12)
