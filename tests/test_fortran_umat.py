"""Cross-validation: Fortran J2 UMAT vs. Python return_mapping.

Verifies that the compiled ``manforge_umat`` module passes the full
:class:`~manforge.verification.UMATVerifier` suite, including auto-generated
single-step test cases and a multi-step tension-unload-compression history.

The compiled module must be built before running:

    make fortran-build-umat

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
    "manforge_umat",
    reason="manforge_umat not compiled -- run: make fortran-build-umat",
)

import manforge  # noqa: F401 -- enables JAX float64
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.verification import UMATVerifier, FortranUMAT


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    return J2Isotropic3D()


@pytest.fixture
def params():
    return {"E": 210_000.0, "nu": 0.3, "sigma_y0": 250.0, "H": 1_000.0}


@pytest.fixture
def verifier(model):
    return UMATVerifier(model, module_name="manforge_umat")


# ---------------------------------------------------------------------------
# FortranUMAT direct interface (primary usage pattern)
# ---------------------------------------------------------------------------

def test_fortran_umat_run(params):
    """FortranUMAT.run calls _run with positional args and returns numpy arrays."""
    fortran = FortranUMAT("manforge_umat")
    dstran = np.array([1e-4, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    stress_f, ep_f, ddsdde_f = fortran.run(
        params["E"], params["nu"], params["sigma_y0"], params["H"],
        np.zeros(6), 0.0, dstran,
    )
    assert np.asarray(stress_f).shape == (6,)
    assert np.asarray(ddsdde_f).shape == (6, 6)


def test_fortran_umat_call_component():
    """FortranUMAT.call invokes a named sub-component subroutine."""
    fortran = FortranUMAT("manforge_umat")
    C = fortran.call("umat_j2_elastic_stiffness", 210_000.0, 0.3)
    assert np.asarray(C).shape == (6, 6)


def test_fortran_umat_run_matches_module_smoke(params):
    """FortranUMAT.run and the raw f2py module give identical results."""
    fortran = FortranUMAT("manforge_umat")
    dstran = np.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    stress_f, ep_f, ddsdde_f = fortran.run(
        params["E"], params["nu"], params["sigma_y0"], params["H"],
        np.zeros(6), 0.0, dstran,
    )
    stress_raw, ep_raw, ddsdde_raw = mod.umat_j2_run(
        params["E"], params["nu"], params["sigma_y0"], params["H"],
        np.zeros(6), 0.0, dstran,
    )
    np.testing.assert_array_equal(stress_f, stress_raw)
    np.testing.assert_array_equal(ddsdde_f, ddsdde_raw)


# ---------------------------------------------------------------------------
# f2py module smoke test (verifies the compiled module independent of bridge)
# ---------------------------------------------------------------------------

def test_f2py_module_smoke(params):
    """f2py module is loadable and produces output with the expected shapes."""
    dstran = np.array([1e-4, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    stress_out, ep_out, ddsdde = mod.umat_j2_run(
        params["E"], params["nu"], params["sigma_y0"], params["H"],
        np.zeros(6), 0.0, dstran,
    )
    assert np.asarray(stress_out).shape == (6,)
    assert np.asarray(ddsdde).shape == (6, 6)


# ---------------------------------------------------------------------------
# Full verification via UMATVerifier
# ---------------------------------------------------------------------------

def test_verifier_passes(verifier, params):
    """UMATVerifier: J2 UMAT passes both single-step and multi-step phases."""
    result = verifier.run(params)

    assert result.passed, (
        f"Verification failed:\n{result.summary()}"
    )
    assert result.single_step.passed
    assert result.multi_step_passed


def test_verifier_custom_strain_history(verifier, params):
    """UMATVerifier accepts a custom 1-D strain history."""
    strain = np.linspace(0.0, 5e-3, 50)
    result = verifier.run(params, strain_history=strain)

    assert result.passed, (
        f"Verification with custom history failed:\n{result.summary()}"
    )


def test_verifier_result_structure(verifier, params):
    """VerificationResult has correct fields and types."""
    result = verifier.run(params)

    assert isinstance(result.passed, bool)
    assert isinstance(result.multi_step_passed, bool)
    assert isinstance(result.multi_step_max_stress_err, float)
    assert isinstance(result.multi_step_max_tangent_err, float)
    assert isinstance(result.multi_step_max_state_err, float)
    assert result.n_multi_steps == len(result.multi_step_steps)
    assert result.n_multi_steps > 0

    step = result.multi_step_steps[0]
    assert isinstance(step.stress_rel_err, float)
    assert isinstance(step.state_rel_err, float)
    assert step.strain_inc.shape == (6,)


def test_verifier_summary_contains_pass(verifier, params):
    """summary() produces a string that reports PASS for all phases."""
    result = verifier.run(params)
    s = result.summary()

    assert "PASS" in s
    assert "Single-step" in s
    assert "Multi-step" in s
