"""Tests for the verification module.

Verifies that :func:`check_tangent` correctly identifies matching tangents
for J2Isotropic3D in elastic and plastic domains, that the
TangentCheckResult structure is correct, and that the Fortran bridge skeleton
raises NotImplementedError as expected.
"""

import autograd.numpy as anp
import pytest

from manforge.simulation.integrator import PythonIntegrator
from manforge.verification.fd_check import check_tangent, TangentCheckResult
from manforge.verification import FortranUMAT


# ---------------------------------------------------------------------------
# Elastic domain
# ---------------------------------------------------------------------------

def test_check_tangent_elastic_passes(model):
    """Elastic domain: AD tangent matches FD tangent within tolerance."""
    strain_inc = anp.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = check_tangent(
        PythonIntegrator(model), anp.zeros(6), model.initial_state(), strain_inc
    )

    assert result.passed
    assert result.max_rel_err < 1e-5
    assert result.ddsdde_ad.shape == (6, 6)
    assert result.ddsdde_fd.shape == (6, 6)


# ---------------------------------------------------------------------------
# Plastic domain — uniaxial
# ---------------------------------------------------------------------------

def test_check_tangent_plastic_uniaxial_passes(model):
    """Plastic uniaxial domain: AD tangent matches FD tangent."""
    strain_inc = anp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = check_tangent(
        PythonIntegrator(model), anp.zeros(6), model.initial_state(), strain_inc
    )

    assert result.passed, f"Max rel err: {result.max_rel_err:.3e}"


# ---------------------------------------------------------------------------
# Plastic domain — multiaxial
# ---------------------------------------------------------------------------

def test_check_tangent_plastic_multiaxial_passes(model):
    """Plastic multiaxial domain: AD tangent matches FD tangent."""
    strain_inc = anp.array([1.5e-3, -0.5e-3, -0.5e-3, 0.5e-3, 0.0, 0.0])
    result = check_tangent(
        PythonIntegrator(model), anp.zeros(6), model.initial_state(), strain_inc
    )

    assert result.passed, f"Max rel err: {result.max_rel_err:.3e}"


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

def test_check_tangent_result_structure(model):
    """TangentCheckResult has correct types for all fields."""
    strain_inc = anp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = check_tangent(
        PythonIntegrator(model), anp.zeros(6), model.initial_state(), strain_inc
    )

    assert isinstance(result, TangentCheckResult)
    assert isinstance(result.passed, bool)
    assert isinstance(result.max_rel_err, float)
    assert result.ddsdde_ad.shape == (6, 6)
    assert result.ddsdde_fd.shape == (6, 6)
    assert result.rel_err_matrix.shape == (6, 6)


# ---------------------------------------------------------------------------
# Verify check can actually fail (FD truncation error at machine precision tol)
# ---------------------------------------------------------------------------

def test_check_tangent_tight_tol_can_fail(model):
    """Using tol=1e-15 triggers failure due to FD truncation error."""
    strain_inc = anp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = check_tangent(
        PythonIntegrator(model), anp.zeros(6), model.initial_state(), strain_inc,
        tol=1e-15,
    )

    assert not result.passed, (
        "Expected failure with tol=1e-15 (FD truncation dominates at machine precision)"
    )


# ---------------------------------------------------------------------------
# Fortran bridge -- error on bad module name
# ---------------------------------------------------------------------------

def test_fortran_umat_bad_module():
    """FortranUMAT raises ModuleNotFoundError for an unknown module name."""
    with pytest.raises(ModuleNotFoundError):
        FortranUMAT("nonexistent_umat_module_xyz")
