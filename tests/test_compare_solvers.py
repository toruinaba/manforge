"""Tests for the generic compare_solvers utility.

Covers:
- Identical solvers produce zero error and pass
- autodiff vs analytical J2 solvers agree within tolerance
- A wrong solver correctly reports failure
- compare_solvers works with mixed plastic/elastic test cases
"""

import jax.numpy as jnp
import numpy as np
import pytest

from manforge.core.return_mapping import return_mapping
from manforge.verification.compare import compare_solvers, SolverComparisonResult


def _make_solver(model, method):
    """Return a SolverFn bound to the given model and method."""
    def _solve(strain_inc, stress_n, state_n, params):
        return return_mapping(
            model,
            jnp.asarray(strain_inc),
            jnp.asarray(stress_n),
            state_n,
            params,
            method=method,
        )
    return _solve


@pytest.fixture
def test_cases(steel_params):
    """A small set of elastic and plastic test cases."""
    state_n = {"ep": jnp.array(0.0)}
    return [
        # elastic step
        {
            "strain_inc": jnp.array([1e-4, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "stress_n":   jnp.zeros(6),
            "state_n":    state_n,
            "params":     steel_params,
        },
        # plastic uniaxial
        {
            "strain_inc": jnp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "stress_n":   jnp.zeros(6),
            "state_n":    state_n,
            "params":     steel_params,
        },
        # plastic shear
        {
            "strain_inc": jnp.array([0.0, 0.0, 0.0, 3e-3, 0.0, 0.0]),
            "stress_n":   jnp.zeros(6),
            "state_n":    state_n,
            "params":     steel_params,
        },
        # plastic mixed
        {
            "strain_inc": jnp.array([1e-3, -5e-4, -5e-4, 1e-3, 0.0, 0.0]),
            "stress_n":   jnp.zeros(6),
            "state_n":    state_n,
            "params":     steel_params,
        },
    ]


# ---------------------------------------------------------------------------
# Identical solver → zero error, always passes
# ---------------------------------------------------------------------------

def test_identical_solvers_pass(model, test_cases, steel_params):
    """Comparing a solver to itself gives zero error and passes."""
    solver = _make_solver(model, "autodiff")
    result = compare_solvers(solver, solver, test_cases)

    assert isinstance(result, SolverComparisonResult)
    assert result.passed
    assert result.n_cases == len(test_cases)
    assert result.n_passed == len(test_cases)
    assert result.max_stress_rel_err == pytest.approx(0.0, abs=1e-15)
    assert result.max_tangent_rel_err == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# autodiff vs analytical — must agree
# ---------------------------------------------------------------------------

def test_autodiff_vs_analytical_pass(model, test_cases):
    """autodiff and analytical solvers must agree within default tolerances."""
    solver_ad = _make_solver(model, "autodiff")
    solver_an = _make_solver(model, "analytical")

    result = compare_solvers(solver_ad, solver_an, test_cases)

    assert result.passed, (
        f"Solvers disagree: max_stress_err={result.max_stress_rel_err:.3e}, "
        f"max_tangent_err={result.max_tangent_rel_err:.3e}\n"
        f"Details: {result.details}"
    )
    assert result.max_stress_rel_err  < 1e-6
    assert result.max_tangent_rel_err < 1e-5


# ---------------------------------------------------------------------------
# Wrong solver → correctly detected as failure
# ---------------------------------------------------------------------------

def test_wrong_solver_fails(model, test_cases):
    """A solver that returns wrong stress must be flagged as failed."""
    solver_ad = _make_solver(model, "autodiff")

    def bad_solver(strain_inc, stress_n, state_n, params):
        stress, state, ddsdde = return_mapping(
            model, jnp.asarray(strain_inc), jnp.asarray(stress_n),
            state_n, params, method="autodiff"
        )
        return stress * 1.1, state, ddsdde  # 10% error in stress

    result = compare_solvers(solver_ad, bad_solver, test_cases, stress_tol=1e-6)

    # At least the plastic cases should fail
    assert not result.passed
    assert result.n_passed < result.n_cases


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

def test_result_details_length(model, test_cases):
    """details list length equals the number of test cases."""
    solver = _make_solver(model, "autodiff")
    result = compare_solvers(solver, solver, test_cases)
    assert len(result.details) == len(test_cases)


def test_result_details_keys(model, test_cases):
    """Each detail record has the required keys."""
    solver = _make_solver(model, "autodiff")
    result = compare_solvers(solver, solver, test_cases)
    for d in result.details:
        assert "case_index"      in d
        assert "stress_rel_err"  in d
        assert "tangent_rel_err" in d
        assert "passed"          in d


def test_empty_test_cases(model):
    """compare_solvers with an empty list passes vacuously."""
    solver = _make_solver(model, "autodiff")
    result = compare_solvers(solver, solver, [])
    assert result.passed
    assert result.n_cases == 0
    assert result.n_passed == 0
