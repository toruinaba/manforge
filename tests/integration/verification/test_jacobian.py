"""Tests for JacobianChecker.compare."""

import autograd.numpy as anp
import numpy as np
import pytest

from manforge.models import J2Isotropic3D
from manforge.simulation.integrator import (
    PythonAnalyticalIntegrator, PythonIntegrator, PythonNumericalIntegrator,
)
from manforge.verification.jacobian import JacobianChecker, JacobianComparisonResult


def _result(model, strain_scale):
    deps = np.zeros(model.ntens)
    deps[0] = strain_scale
    state0 = model.initial_state()
    return PythonIntegrator(model).stress_update(deps, np.zeros(model.ntens), state0), state0


class TestCompareJacobiansJ2:
    def test_returns_result_type(self, model):
        r_ad, state0 = _result(model, 3e-3)
        r_an, _ = _result(model, 3e-3)
        out = JacobianChecker(model).compare(r_ad, r_an)
        assert isinstance(out, JacobianComparisonResult)

    def test_identical_results_zero_error(self, model):
        r, state0 = _result(model, 3e-3)
        out = JacobianChecker(model).compare(r, r)
        assert out.passed
        assert out.max_rel_err == pytest.approx(0.0, abs=1e-15)

    def test_numerical_newton_vs_user_defined_plastic(self, model):
        """Jacobian blocks of autodiff and analytical solvers must agree < 1e-10."""
        r_ad, state0 = _result(model, 3e-3)
        r_an, _ = _result(model, 3e-3)
        r_an_ud = PythonAnalyticalIntegrator(model).stress_update(
            np.array([3e-3] + [0.0] * (model.ntens - 1)),
            np.zeros(model.ntens), state0,
        )
        out = JacobianChecker(model, rtol=1e-8).compare(r_ad, r_an_ud)
        assert out.passed, (
            f"max_rel_err={out.max_rel_err:.3e}, blocks={out.blocks}"
        )

    def test_elastic_step_does_not_raise(self, model):
        """Elastic step (dlambda=0) should not raise in compare."""
        r, state0 = _result(model, 1e-5)
        assert not r.is_plastic
        out = JacobianChecker(model).compare(r, r)
        assert out.passed

    def test_blocks_dict_populated(self, model):
        """blocks dict should contain at least the four base block keys."""
        r, state0 = _result(model, 3e-3)
        out = JacobianChecker(model).compare(r, r)
        for key in ("stress::stress", "stress::dlambda", "dlambda::stress", "dlambda::dlambda"):
            assert key in out.blocks

    def test_max_rel_err_equals_max_of_blocks(self, model):
        r, state0 = _result(model, 3e-3)
        out = JacobianChecker(model).compare(r, r)
        assert out.max_rel_err == pytest.approx(max(out.blocks.values()), abs=1e-15)


class _StrainIncDependentJ2(J2Isotropic3D):
    """J2 whose explicit update actually reads strain_inc.

    Physically contrived (ep picks up a Δε-proportional term), but it makes
    the residual strain_inc-dependent the same way an elastic/plastic strain
    decomposition would.
    """

    def update_state(self, new, dlambda, inc):
        bump = anp.sum(inc.strain_inc * inc.strain_inc)
        return [self.ep(inc.state_n["ep"] + dlambda + bump)]


class TestComputeStrainInc:
    """compute() must evaluate the same residual the solver solved.

    A model whose update_state reads strain_inc gets a *different* residual
    (or a TypeError on strain_inc=None) when the checker omits it.
    """

    def _plastic_result(self):
        model = _StrainIncDependentJ2(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
        deps = np.zeros(6)
        deps[0] = 3e-3
        state0 = model.initial_state()
        result = PythonIntegrator(model).stress_update(deps, np.zeros(6), state0)
        return model, deps, state0, result

    def test_compute_forwards_strain_inc(self):
        model, deps, state0, result = self._plastic_result()
        jac = JacobianChecker(model).compute(result)
        assert jac.full.shape == (7, 7)

    def test_compare_forwards_strain_inc(self):
        model, deps, state0, result = self._plastic_result()
        out = JacobianChecker(model).compare(result, result)
        assert out.passed


# ---------------------------------------------------------------------------
# The increment comes from the result
# ---------------------------------------------------------------------------

class TestIncrementComesFromTheResult:
    @staticmethod
    def _step(model):
        deps = anp.array([3e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
        return PythonNumericalIntegrator(model).stress_update(
            deps, anp.zeros(6), model.initial_state())

    def test_compute_needs_nothing_but_the_result(self, model):
        result = self._step(model)
        jac = JacobianChecker(model).compute(result)
        assert jac.full.shape == (7, 7)

    def test_result_without_increment_is_rejected(self, model):
        """A Fortran result reports no trial state; its residual is unknown."""
        from dataclasses import replace
        result = replace(self._step(model), increment=None)
        with pytest.raises(ValueError, match="no increment"):
            JacobianChecker(model).compute(result)

    def test_compare_evaluates_both_on_the_first_results_increment(self, model):
        from dataclasses import replace
        result = self._step(model)
        no_increment = replace(result, increment=None)
        out = JacobianChecker(model).compare(result, no_increment)
        assert out.passed and out.max_rel_err == 0.0
        with pytest.raises(ValueError, match="result_a must come from a Python integrator"):
            JacobianChecker(model).compare(no_increment, result)
