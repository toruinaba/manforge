"""Tests for the generic compare_solvers utility.

Covers:
- Identical solvers produce zero error and pass
- autodiff vs analytical J2 solvers agree within tolerance
- A wrong solver correctly reports failure
- compare_solvers works with mixed plastic/elastic test cases
"""

import dataclasses

import autograd.numpy as anp
import numpy as np
import pytest

from manforge.core.stress_update import stress_update, StressUpdateResult
from manforge.verification.compare import compare_solvers, SolverComparisonResult


def _make_solver(method):
    """Return a solver callable bound to the given method."""
    def _solve(model, strain_inc, stress_n, state_n):
        return stress_update(
            model,
            anp.asarray(strain_inc),
            anp.asarray(stress_n),
            state_n,
            method=method,
        )
    return _solve


@pytest.fixture
def test_cases(model):
    """A small set of elastic and plastic test cases."""
    state_n = model.initial_state()
    return [
        # elastic step
        {
            "strain_inc": anp.array([1e-4, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "stress_n":   anp.zeros(6),
            "state_n":    state_n,
        },
        # plastic uniaxial
        {
            "strain_inc": anp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "stress_n":   anp.zeros(6),
            "state_n":    state_n,
        },
        # plastic shear
        {
            "strain_inc": anp.array([0.0, 0.0, 0.0, 3e-3, 0.0, 0.0]),
            "stress_n":   anp.zeros(6),
            "state_n":    state_n,
        },
        # plastic mixed
        {
            "strain_inc": anp.array([1e-3, -5e-4, -5e-4, 1e-3, 0.0, 0.0]),
            "stress_n":   anp.zeros(6),
            "state_n":    state_n,
        },
    ]


# ---------------------------------------------------------------------------
# Identical solver → zero error, always passes
# ---------------------------------------------------------------------------

def test_identical_solvers_pass(model, test_cases):
    """Comparing a solver to itself gives zero error and passes."""
    solver = _make_solver("numerical_newton")
    result = compare_solvers(model, solver, solver, test_cases)

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
    solver_ad = _make_solver("numerical_newton")
    solver_an = _make_solver("user_defined")

    result = compare_solvers(model, solver_ad, solver_an, test_cases)

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
    solver_ad = _make_solver("numerical_newton")

    def bad_solver(model, strain_inc, stress_n, state_n):
        r = stress_update(
            model, anp.asarray(strain_inc), anp.asarray(stress_n),
            state_n, method="numerical_newton"
        )
        if r.return_mapping is not None:
            bad_rm = dataclasses.replace(r.return_mapping, stress=r.return_mapping.stress * 1.1)
            return dataclasses.replace(r, return_mapping=bad_rm)
        return r

    result = compare_solvers(model, solver_ad, bad_solver, test_cases, stress_tol=1e-6)

    # At least the plastic cases should fail
    assert not result.passed
    assert result.n_passed < result.n_cases


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

def test_result_details_length(model, test_cases):
    """details list length equals the number of test cases."""
    solver = _make_solver("numerical_newton")
    result = compare_solvers(model, solver, solver, test_cases)
    assert len(result.details) == len(test_cases)


def test_result_details_keys(model, test_cases):
    """Each detail record has the required keys."""
    solver = _make_solver("numerical_newton")
    result = compare_solvers(model, solver, solver, test_cases)
    required = {
        "case_index", "stress_rel_err", "tangent_rel_err",
        "state_rel_err", "trial_rel_err",
        "is_plastic_match", "elastic_branch_match", "passed",
    }
    for d in result.details:
        assert required <= d.keys()


def test_empty_test_cases(model):
    """compare_solvers with an empty list passes vacuously."""
    solver = _make_solver("numerical_newton")
    result = compare_solvers(model, solver, solver, [])
    assert result.passed
    assert result.n_cases == 0
    assert result.n_passed == 0
