"""Tests for iter_compare_solvers and SolverCaseResult."""

import autograd.numpy as anp
import numpy as np
import pytest

from manforge.core.stress_update import stress_update
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.verification.compare import (
    SolverCaseResult,
    compare_jacobians,
    compare_solvers,
    iter_compare_solvers,
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


class TestIterCompareSolvers:
    def test_yields_solver_case_result(self, model, test_cases):
        solver = _solver("numerical_newton")
        for case in iter_compare_solvers(model, solver, solver, test_cases):
            assert isinstance(case, SolverCaseResult)

    def test_case_index_sequential(self, model, test_cases):
        solver = _solver("numerical_newton")
        indices = [c.case_index for c in iter_compare_solvers(model, solver, solver, test_cases)]
        assert indices == list(range(len(test_cases)))

    def test_identical_solvers_all_pass(self, model, test_cases):
        solver = _solver("numerical_newton")
        for case in iter_compare_solvers(model, solver, solver, test_cases):
            assert case.passed
            assert case.stress_rel_err == pytest.approx(0.0, abs=1e-14)
            assert case.tangent_rel_err == pytest.approx(0.0, abs=1e-14)

    def test_matches_compare_solvers(self, model, test_cases):
        solver_a = _solver("numerical_newton")
        solver_b = _solver("numerical_newton")
        batch = compare_solvers(model, solver_a, solver_b, test_cases)
        cases = list(iter_compare_solvers(model, solver_a, solver_b, test_cases))

        assert len(cases) == batch.n_cases
        for case, detail in zip(cases, batch.details):
            assert case.case_index == detail["case_index"]
            assert case.passed == detail["passed"]
            assert case.stress_rel_err == pytest.approx(detail["stress_rel_err"], rel=1e-12)
            assert case.tangent_rel_err == pytest.approx(detail["tangent_rel_err"], rel=1e-12)

    def test_early_break_on_failure(self, model, test_cases):
        solver_a = _solver("numerical_newton")

        def bad_solver(m, strain_inc, stress_n, state_n):
            result = stress_update(m, anp.asarray(strain_inc), anp.asarray(stress_n), state_n)
            from dataclasses import replace
            return replace(result, stress_trial=result.stress_trial * 2.0)

        failed = []
        for case in iter_compare_solvers(model, solver_a, bad_solver, test_cases):
            if not case.passed:
                failed.append(case)
                break

        assert len(failed) >= 1
        assert not failed[0].passed

    def test_exposes_raw_results_for_jacobians(self, model, test_cases):
        solver = _solver("numerical_newton")
        for case in iter_compare_solvers(model, solver, solver, test_cases):
            state_n = test_cases[case.case_index]["state_n"]
            jac_result = compare_jacobians(model, case.result_a, case.result_b, state_n)
            assert jac_result.passed
