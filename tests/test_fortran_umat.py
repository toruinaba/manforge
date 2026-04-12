"""Cross-validation: Fortran J2 UMAT vs. Python return_mapping.

Verifies that the Fortran ``umat_j2_run`` subroutine produces stress,
equivalent plastic strain, and consistent tangent that agree with the
Python reference implementation to tight tolerances.

The compiled module ``manforge_umat`` must be built before running:

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

import jax.numpy as jnp
import manforge  # noqa: F401 -- enables JAX float64
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.core.return_mapping import return_mapping
from manforge.verification.fortran_bridge import FortranUMAT
from manforge.verification.compare import compare_solvers


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def params():
    return {"E": 210_000.0, "nu": 0.3, "sigma_y0": 250.0, "H": 1_000.0}


@pytest.fixture
def model():
    return J2Isotropic3D()


@pytest.fixture
def fortran_umat(model):
    return FortranUMAT(module_name="manforge_umat", model=model)


@pytest.fixture
def zero_state():
    return {"ep": 0.0}


@pytest.fixture
def elastic_dstran():
    """Strain increment that stays in the elastic domain."""
    return np.array([1e-4, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.fixture
def plastic_dstran_uniaxial():
    """Uniaxial strain increment well into the plastic domain."""
    return np.array([5e-3, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.fixture
def plastic_dstran_multiaxial():
    """Multiaxial strain increment into the plastic domain."""
    return np.array([3e-3, -1e-3, -1e-3, 1e-3, 0.0, 0.0])


def _py_result(model, params, state_n, dstran):
    """Run Python return_mapping and return numpy arrays."""
    stress_0 = jnp.zeros(6)
    stress, state, ddsdde = return_mapping(
        model, jnp.array(dstran), stress_0, state_n, params
    )
    return np.array(stress), state, np.array(ddsdde)


# ---------------------------------------------------------------------------
# f2py module smoke test (verifies the compiled module works independently)
# ---------------------------------------------------------------------------

def test_f2py_module_smoke(params, zero_state):
    """f2py module is loadable and produces plausible output shapes."""
    dstran = np.array([1e-4, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    stress_out, ep_out, ddsdde = mod.umat_j2_run(
        params["E"], params["nu"], params["sigma_y0"], params["H"],
        np.zeros(6), float(zero_state["ep"]), dstran,
    )
    assert np.asarray(stress_out).shape == (6,)
    assert np.asarray(ddsdde).shape == (6, 6)


# ---------------------------------------------------------------------------
# Bridge API tests (FortranUMAT.call vs Python return_mapping)
# ---------------------------------------------------------------------------

def test_umat_elastic_stress(model, params, fortran_umat, zero_state, elastic_dstran):
    """Elastic step: Fortran stress matches Python to machine precision."""
    stress_py, _, _ = _py_result(model, params, zero_state, elastic_dstran)
    stress_f90, _, _ = fortran_umat.call(elastic_dstran, np.zeros(6), zero_state, params)

    np.testing.assert_allclose(
        stress_f90, stress_py, rtol=1e-12,
        err_msg="Elastic stress: Fortran vs Python mismatch",
    )


def test_umat_elastic_ddsdde(model, params, fortran_umat, zero_state, elastic_dstran):
    """Elastic step: Fortran DDSDDE equals the elastic stiffness C."""
    C_py = np.array(model.elastic_stiffness(params))
    _, _, ddsdde_f90 = fortran_umat.call(elastic_dstran, np.zeros(6), zero_state, params)

    np.testing.assert_allclose(
        ddsdde_f90, C_py, rtol=1e-12,
        err_msg="Elastic DDSDDE: Fortran vs Python (elastic C) mismatch",
    )


def test_umat_plastic_uniaxial_stress(model, params, fortran_umat, zero_state,
                                      plastic_dstran_uniaxial):
    """Uniaxial plastic step: Fortran stress matches Python reference."""
    stress_py, _, _ = _py_result(model, params, zero_state, plastic_dstran_uniaxial)
    stress_f90, _, _ = fortran_umat.call(
        plastic_dstran_uniaxial, np.zeros(6), zero_state, params
    )

    np.testing.assert_allclose(
        stress_f90, stress_py, rtol=1e-6,
        err_msg="Uniaxial plastic stress: Fortran vs Python mismatch",
    )


def test_umat_plastic_multiaxial_stress(model, params, fortran_umat, zero_state,
                                        plastic_dstran_multiaxial):
    """Multiaxial plastic step: Fortran stress matches Python reference."""
    stress_py, _, _ = _py_result(model, params, zero_state, plastic_dstran_multiaxial)
    stress_f90, _, _ = fortran_umat.call(
        plastic_dstran_multiaxial, np.zeros(6), zero_state, params
    )

    np.testing.assert_allclose(
        stress_f90, stress_py, rtol=1e-6,
        err_msg="Multiaxial plastic stress: Fortran vs Python mismatch",
    )


def test_umat_plastic_ddsdde(model, params, fortran_umat, zero_state,
                              plastic_dstran_uniaxial):
    """Plastic step: Fortran consistent tangent matches Python reference."""
    _, _, ddsdde_py = _py_result(model, params, zero_state, plastic_dstran_uniaxial)
    _, _, ddsdde_f90 = fortran_umat.call(
        plastic_dstran_uniaxial, np.zeros(6), zero_state, params
    )

    np.testing.assert_allclose(
        ddsdde_f90, ddsdde_py, rtol=1e-5,
        err_msg="Plastic DDSDDE: Fortran vs Python mismatch",
    )


def test_umat_state_update(model, params, fortran_umat, zero_state,
                           plastic_dstran_uniaxial):
    """Fortran equivalent plastic strain matches Python state_new['ep']."""
    _, state_py, _ = _py_result(model, params, zero_state, plastic_dstran_uniaxial)
    _, state_f90, _ = fortran_umat.call(
        plastic_dstran_uniaxial, np.zeros(6), zero_state, params
    )

    np.testing.assert_allclose(
        state_f90["ep"], state_py["ep"], rtol=1e-6,
        err_msg="Equivalent plastic strain: Fortran vs Python mismatch",
    )


# ---------------------------------------------------------------------------
# compare_solvers integration test
# ---------------------------------------------------------------------------

def test_compare_solvers_fortran_vs_python(model, params, fortran_umat, zero_state):
    """compare_solvers validates Fortran UMAT against Python across all regimes."""
    test_cases = [
        # elastic
        {
            "strain_inc": np.array([1e-4, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "stress_n":   np.zeros(6),
            "state_n":    {"ep": 0.0},
            "params":     params,
        },
        # uniaxial plastic
        {
            "strain_inc": np.array([5e-3, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "stress_n":   np.zeros(6),
            "state_n":    {"ep": 0.0},
            "params":     params,
        },
        # multiaxial plastic
        {
            "strain_inc": np.array([3e-3, -1e-3, -1e-3, 1e-3, 0.0, 0.0]),
            "stress_n":   np.zeros(6),
            "state_n":    {"ep": 0.0},
            "params":     params,
        },
    ]

    result = compare_solvers(
        fortran_umat.make_python_solver(),
        fortran_umat.call,
        test_cases,
        stress_tol=1e-6,
        tangent_tol=1e-5,
    )

    assert result.n_cases == 3
    assert result.n_passed == result.n_cases, (
        f"Failed cases: {[d for d in result.details if not d['passed']]}"
    )
    assert result.passed
