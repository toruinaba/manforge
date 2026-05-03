"""Tests for compare_jacobians."""

import numpy as np
import pytest

from manforge.core.stress_update import stress_update
from manforge.verification.solver_crosscheck import compare_jacobians, JacobianComparisonResult


def _result(model, strain_scale):
    deps = np.zeros(model.ntens)
    deps[0] = strain_scale
    state0 = model.initial_state()
    return stress_update(model, deps, np.zeros(model.ntens), state0), state0


class TestCompareJacobiansJ2:
    def test_returns_result_type(self, model):
        r_ad, state0 = _result(model, 3e-3)
        r_an, _ = _result(model, 3e-3)
        out = compare_jacobians(model, r_ad, r_an, state0)
        assert isinstance(out, JacobianComparisonResult)

    def test_identical_results_zero_error(self, model):
        r, state0 = _result(model, 3e-3)
        out = compare_jacobians(model, r, r, state0)
        assert out.passed
        assert out.max_rel_err == pytest.approx(0.0, abs=1e-15)

    def test_numerical_newton_vs_user_defined_plastic(self, model):
        """Jacobian blocks of autodiff and analytical solvers must agree < 1e-10."""
        r_ad, state0 = _result(model, 3e-3)
        r_an, _ = _result(model, 3e-3)
        r_an_ud = stress_update(model, np.array([3e-3] + [0.0] * (model.ntens - 1)),
                                np.zeros(model.ntens), state0, method="user_defined")
        out = compare_jacobians(model, r_ad, r_an_ud, state0, rtol=1e-8)
        assert out.passed, (
            f"max_rel_err={out.max_rel_err:.3e}, blocks={out.blocks}"
        )

    def test_elastic_step_does_not_raise(self, model):
        """Elastic step (dlambda=0) should not raise in compare_jacobians."""
        r, state0 = _result(model, 1e-5)
        assert not r.is_plastic
        out = compare_jacobians(model, r, r, state0)
        assert out.passed

    def test_blocks_dict_populated(self, model):
        """blocks dict should contain at least the four scalar block keys."""
        r, state0 = _result(model, 3e-3)
        out = compare_jacobians(model, r, r, state0)
        for key in ("dstress_dsigma", "dstress_ddlambda", "dyield_dsigma", "dyield_ddlambda"):
            assert key in out.blocks

    def test_max_rel_err_equals_max_of_blocks(self, model):
        r, state0 = _result(model, 3e-3)
        out = compare_jacobians(model, r, r, state0)
        assert out.max_rel_err == pytest.approx(max(out.blocks.values()), abs=1e-15)
