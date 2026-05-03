"""Tests for SolverComparison.iter_run and SolverCaseResult."""

import autograd.numpy as anp
import numpy as np
import pytest

from manforge.core.stress_update import stress_update
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.verification.compare import (
    SolverCaseResult,
    SolverComparison,
    compare_jacobians,
)


@pytest.fixture
def model():
    return J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)


def _solver(method):
    def _solve(m, strain_inc, stress_n, state_n):
        return stress_update(m, anp.asarray(strain_inc), anp.asarray(stress_n), state_n, method=method)
    return _solve


@pytest.fixture
def test_cases(model):
    state_n = model.initial_state()
    yield_strain = float(250.0 / 210000.0)
    return [
        {"strain_inc": anp.array([1e-4, 0.0, 0.0, 0.0, 0.0, 0.0]), "stress_n": anp.zeros(6), "state_n": state_n},
        {"strain_inc": anp.array([yield_strain * 3, 0.0, 0.0, 0.0, 0.0, 0.0]), "stress_n": anp.zeros(6), "state_n": state_n},
        {"strain_inc": anp.array([1e-3, 5e-4, 0.0, 2e-4, 0.0, 0.0]), "stress_n": anp.zeros(6), "state_n": state_n},
    ]


class TestSolverComparisonIterRun:
    def test_yields_solver_case_result(self, model, test_cases):
        solver = _solver("numerical_newton")
        cs = SolverComparison(solver, solver)
        for case in cs.iter_run(model, test_cases):
            assert isinstance(case, SolverCaseResult)

    def test_case_index_sequential(self, model, test_cases):
        solver = _solver("numerical_newton")
        cs = SolverComparison(solver, solver)
        indices = [c.index for c in cs.iter_run(model, test_cases)]
        assert indices == list(range(len(test_cases)))

    def test_identical_solvers_all_pass(self, model, test_cases):
        solver = _solver("numerical_newton")
        cs = SolverComparison(solver, solver)
        for case in cs.iter_run(model, test_cases):
            assert case.passed
            assert case.stress_rel_err == pytest.approx(0.0, abs=1e-14)
            assert case.tangent_rel_err == pytest.approx(0.0, abs=1e-14)

    def test_iter_run_matches_run(self, model, test_cases):
        solver_a = _solver("numerical_newton")
        solver_b = _solver("numerical_newton")
        cs = SolverComparison(solver_a, solver_b)
        batch = cs.run(model, test_cases)
        iter_cases = list(cs.iter_run(model, test_cases))

        assert len(iter_cases) == batch.n_cases
        for iter_c, batch_c in zip(iter_cases, batch.cases):
            assert iter_c.index == batch_c.index
            assert iter_c.passed == batch_c.passed
            assert iter_c.stress_rel_err == pytest.approx(batch_c.stress_rel_err, rel=1e-12)
            assert iter_c.tangent_rel_err == pytest.approx(batch_c.tangent_rel_err, rel=1e-12)

    def test_early_break_on_failure(self, model, test_cases):
        solver_a = _solver("numerical_newton")

        def bad_solver(m, strain_inc, stress_n, state_n):
            result = stress_update(m, anp.asarray(strain_inc), anp.asarray(stress_n), state_n)
            from dataclasses import replace
            return replace(result, stress_trial=result.stress_trial * 2.0)

        cs = SolverComparison(solver_a, bad_solver)
        failed = []
        for case in cs.iter_run(model, test_cases):
            if not case.passed:
                failed.append(case)
                break

        assert len(failed) >= 1
        assert not failed[0].passed

    def test_exposes_raw_results_for_jacobians(self, model, test_cases):
        solver = _solver("numerical_newton")
        cs = SolverComparison(solver, solver)
        for case in cs.iter_run(model, test_cases):
            state_n = test_cases[case.index]["state_n"]
            jac_result = compare_jacobians(model, case.result_a, case.result_b, state_n)
            assert jac_result.passed
