"""Tests for SolverCrosscheck.

Covers:
- Identical solvers produce zero error and pass
- autodiff vs analytical J2 solvers agree within tolerance
- A wrong solver correctly reports failure
- SolverCrosscheck works with mixed plastic/elastic test cases
"""

import dataclasses

import autograd.numpy as anp
import numpy as np
import pytest

from manforge.core.stress_update import stress_update, StressUpdateResult
from manforge.verification.solver_crosscheck import SolverCrosscheck, SolverCrosscheckResult


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
    cs = SolverCrosscheck(solver, solver)
    result = cs.run(model, test_cases)

    assert isinstance(result, SolverCrosscheckResult)
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

    cs = SolverCrosscheck(solver_ad, solver_an)
    result = cs.run(model, test_cases)

    assert result.passed, (
        f"Solvers disagree: max_stress_err={result.max_stress_rel_err:.3e}, "
        f"max_tangent_err={result.max_tangent_rel_err:.3e}\n"
        f"Failed cases: {[c.index for c in result.cases if not c.passed]}"
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

    cs = SolverCrosscheck(solver_ad, bad_solver, stress_tol=1e-6)
    result = cs.run(model, test_cases)

    assert not result.passed
    assert result.n_passed < result.n_cases


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

def test_result_cases_length(model, test_cases):
    """cases list length equals the number of test cases."""
    solver = _make_solver("numerical_newton")
    cs = SolverCrosscheck(solver, solver)
    result = cs.run(model, test_cases)
    assert len(result.cases) == len(test_cases)


def test_result_case_fields(model, test_cases):
    """Each SolverCaseResult has the required attributes."""
    from manforge.verification.solver_crosscheck import SolverCaseResult
    solver = _make_solver("numerical_newton")
    cs = SolverCrosscheck(solver, solver)
    result = cs.run(model, test_cases)
    for c in result.cases:
        assert isinstance(c, SolverCaseResult)
        assert c.index >= 0
        assert c.stress_rel_err is not None
        assert c.tangent_rel_err is not None
        assert isinstance(c.state_rel_err, dict)
        assert isinstance(c.is_plastic_match, bool)
        assert isinstance(c.elastic_branch_match, bool)
        assert isinstance(c.passed, bool)


def test_empty_test_cases(model):
    """SolverCrosscheck with an empty list passes vacuously."""
    solver = _make_solver("numerical_newton")
    cs = SolverCrosscheck(solver, solver)
    result = cs.run(model, [])
    assert result.passed
    assert result.n_cases == 0
    assert result.n_passed == 0


# ---------------------------------------------------------------------------
# iter_run — step-by-step generator
# ---------------------------------------------------------------------------

def test_iter_run_early_break(model, test_cases):
    """iter_run allows early break on first failing case."""
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

    cs = SolverCrosscheck(solver_ad, bad_solver)
    found = False
    for case in cs.iter_run(model, test_cases):
        if not case.passed:
            found = True
            assert case.result_a is not None
            assert case.result_b is not None
            break
    assert found
