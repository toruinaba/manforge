"""Tests for SolverCrosscheck.

Covers:
- Identical integrators produce zero error and pass
- autodiff vs analytical J2 integrators agree within tolerance
- A wrong integrator correctly reports failure
- SolverCrosscheck works with mixed plastic/elastic test cases
- SolverCrosscheck with FortranIntegrator
"""

import dataclasses

import autograd.numpy as anp
import numpy as np
import pytest

from manforge.core.stress_update import StressUpdateResult
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.simulation import PythonNumericalIntegrator, PythonAnalyticalIntegrator
from manforge.verification.solver_crosscheck import SolverCrosscheck, SolverCrosscheckResult


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
# Identical integrators → zero error, always passes
# ---------------------------------------------------------------------------

def test_identical_integrators_pass(model, test_cases):
    """Comparing an integrator to itself gives zero error and passes."""
    pi = PythonNumericalIntegrator(model)
    cs = SolverCrosscheck(pi, pi)
    result = cs.run(test_cases)

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
    """autodiff and analytical integrators must agree within default tolerances."""
    pi_num = PythonNumericalIntegrator(model)
    pi_ana = PythonAnalyticalIntegrator(model)

    cs = SolverCrosscheck(pi_num, pi_ana)
    result = cs.run(test_cases)

    assert result.passed, (
        f"Solvers disagree: max_stress_err={result.max_stress_rel_err:.3e}, "
        f"max_tangent_err={result.max_tangent_rel_err:.3e}\n"
        f"Failed cases: {[c.index for c in result.cases if not c.passed]}"
    )
    assert result.max_stress_rel_err  < 1e-6
    assert result.max_tangent_rel_err < 1e-5


# ---------------------------------------------------------------------------
# Wrong integrator → correctly detected as failure
# ---------------------------------------------------------------------------

class _BadIntegrator:
    """Integrator that multiplies plastic-step stress by 1.1."""

    def __init__(self, model):
        self._pi = PythonNumericalIntegrator(model)
        self.stress_state = model.stress_state
        self.ntens = model.ntens

    def initial_state(self):
        return self._pi.initial_state()

    def elastic_stiffness(self):
        return self._pi.elastic_stiffness()

    def stress_update(self, strain_inc, stress_n, state_n):
        r = self._pi.stress_update(strain_inc, stress_n, state_n)
        if r.return_mapping is not None:
            bad_rm = dataclasses.replace(r.return_mapping, stress=r.return_mapping.stress * 1.1)
            return dataclasses.replace(r, return_mapping=bad_rm)
        return r


def test_wrong_integrator_fails(model, test_cases):
    """An integrator that returns wrong stress must be flagged as failed."""
    pi_good = PythonNumericalIntegrator(model)
    pi_bad = _BadIntegrator(model)

    cs = SolverCrosscheck(pi_good, pi_bad, stress_tol=1e-6)
    result = cs.run(test_cases)

    assert not result.passed
    assert result.n_passed < result.n_cases


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

def test_result_cases_length(model, test_cases):
    """cases list length equals the number of test cases."""
    pi = PythonNumericalIntegrator(model)
    cs = SolverCrosscheck(pi, pi)
    result = cs.run(test_cases)
    assert len(result.cases) == len(test_cases)


def test_result_case_fields(model, test_cases):
    """Each SolverCaseResult has the required attributes."""
    from manforge.verification.solver_crosscheck import SolverCaseResult
    pi = PythonNumericalIntegrator(model)
    cs = SolverCrosscheck(pi, pi)
    result = cs.run(test_cases)
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
    pi = PythonNumericalIntegrator(model)
    cs = SolverCrosscheck(pi, pi)
    result = cs.run([])
    assert result.passed
    assert result.n_cases == 0
    assert result.n_passed == 0


# ---------------------------------------------------------------------------
# iter_run — step-by-step generator
# ---------------------------------------------------------------------------

def test_iter_run_early_break(model, test_cases):
    """iter_run allows early break on first failing case."""
    pi_good = PythonNumericalIntegrator(model)
    pi_bad = _BadIntegrator(model)

    cs = SolverCrosscheck(pi_good, pi_bad)
    found = False
    for case in cs.iter_run(test_cases):
        if not case.passed:
            found = True
            assert case.result_a is not None
            assert case.result_b is not None
            break
    assert found
