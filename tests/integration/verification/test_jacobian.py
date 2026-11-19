"""Tests for JacobianChecker.compare."""

import autograd.numpy as anp
import numpy as np
import pytest

from manforge.models import J2Isotropic3D
from manforge.simulation import (
    PythonAnalyticalIntegrator, PythonIntegrator, PythonNumericalIntegrator,
)
from manforge.verification import JacobianChecker, JacobianComparisonResult


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
        out = JacobianChecker(model, tol=1e-8).compare(r_ad, r_an_ud)
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


# ---------------------------------------------------------------------------
# Hand-derived blocks against the autodiff reference
# ---------------------------------------------------------------------------

class TestCheckHandBlocks:
    @staticmethod
    def _results(model, n=12):
        """A multiaxial history: direct and shear components together."""
        from manforge.simulation import StrainDriver, FieldHistory, FieldType

        t = np.linspace(0.0, 1.0, n + 1)[1:]
        data = np.zeros((n, 6))
        data[:, 0] = 4e-3 * t
        data[:, 3] = 6e-3 * np.sin(np.pi * t)
        load = FieldHistory(FieldType.STRAIN, "Strain", data)
        return StrainDriver(PythonNumericalIntegrator(model)).run(load).step_results

    def test_exact_blocks_pass_and_the_rest_are_listed(self):
        from tests.fixtures.hand_block_models import OW_PARAMS, OWWithHandBlocks

        m = OWWithHandBlocks(**OW_PARAMS)
        results = self._results(m)
        report = JacobianChecker(m).check_hand_blocks(results)

        assert report.passed, report.summary()
        assert report.n_declared == 8 and report.n_total == 16
        assert report.n_points == sum(bool(r.is_plastic) for r in results) > 0
        # The σ and α rows are not derived in this fixture.
        assert set(report.undeclared) == {
            (row, col) for row in ("stress", "alpha")
            for col in ("stress", "dlambda", "alpha", "ep")
        }
        assert "8/16 derived" in report.summary()
        assert "not derived: [stress][stress]" in report.summary()

    def test_the_checkers_tol_is_the_default_and_can_be_overridden(self):
        from tests.fixtures.hand_block_models import OW_PARAMS, OWWithHandBlocks

        m = OWWithHandBlocks(**OW_PARAMS)
        results = self._results(m)
        strict = JacobianChecker(m, tol=1e-30)
        report = strict.check_hand_blocks(results)
        assert not report.passed                      # round-off exceeds 1e-30
        assert {b.tol for b in report.blocks.values()} == {1e-30}
        assert strict.check_hand_blocks(results, tol=1e-8).passed

    def test_a_wrong_block_is_found_by_name(self):
        """One sign error among eight blocks: named, not just 'the Jacobian is off'."""
        from tests.fixtures.hand_block_models import OW_PARAMS, OWWithWrongBlock

        m = OWWithWrongBlock(**OW_PARAMS)
        report = JacobianChecker(m).check_hand_blocks(self._results(m))

        assert not report.passed
        assert [(b.row, b.col, b.method) for b in report.failed] == [("ep", "dlambda", "dRep_dl")]
        assert report.failed[0].worst_err == pytest.approx(2.0)     # +1 against −1
        assert "FAIL [ep][dlambda] dRep_dl" in report.summary()

    def test_declared_approximation_is_held_to_its_own_tolerance(self):
        from tests.fixtures.hand_block_models import OW_PARAMS, OWWithApproxBlock

        m = OWWithApproxBlock(**OW_PARAMS)
        report = JacobianChecker(m).check_hand_blocks(self._results(m), tol=1e-10)

        block = report.blocks[("ep", "ep")]
        assert block.approx and block.tol == 1e-2
        assert block.worst_err == pytest.approx(1e-3, rel=1e-6)
        assert block.passed and report.passed          # 1e-3 is inside the declared 1e-2
        assert "approx: deliberately off" in report.summary()

    def test_approximation_beyond_its_declared_size_fails(self):
        from manforge.core import hand_block
        from tests.fixtures.hand_block_models import OW_PARAMS, OWWithHandBlocks

        class _TooFar(OWWithHandBlocks):
            @hand_block("ep", "ep", approx=1e-4, note="claims 1e-4, is 1e-3")
            def dRep_dep(self):
                return 1.0 + 1e-3

        m = _TooFar(**OW_PARAMS)
        report = JacobianChecker(m).check_hand_blocks(self._results(m))
        assert [b.method for b in report.failed] == ["dRep_dep"]

    def test_nothing_checked_is_an_error_not_a_pass(self, model):
        from tests.fixtures.hand_block_models import OW_PARAMS, OWWithHandBlocks

        with pytest.raises(ValueError, match="declares no hand-derived block"):
            JacobianChecker(model).check_hand_blocks(self._results(model))

        m = OWWithHandBlocks(**OW_PARAMS)
        elastic_only = [r for r in self._results(m) if not r.is_plastic]
        with pytest.raises(ValueError, match="no plastic step"):
            JacobianChecker(m).check_hand_blocks(elastic_only)

    def test_hand_returns_the_assembled_matrix(self):
        from tests.fixtures.hand_block_models import OW_PARAMS, OWWithWrongBlock

        m = OWWithWrongBlock(**OW_PARAMS)
        result = [r for r in self._results(m) if r.is_plastic][-1]
        hand, ref = JacobianChecker(m).hand(result), JacobianChecker(m).compute(result)

        assert hand.source["ep"]["dlambda"] == "hand"
        assert hand.source["stress"]["stress"] == "ad"
        assert float(hand.part["ep"]["dlambda"]) == +1.0       # the wrong hand value
        assert float(ref.part["ep"]["dlambda"]) == -1.0


class TestCheckSolution:
    """The solutions of a hand-written return mapping, substituted into the
    framework residual."""

    @staticmethod
    def _results(model, n=14):
        from manforge.simulation import (
            FieldHistory, FieldType, PythonAnalyticalIntegrator, StrainDriver,
        )

        t = np.linspace(0.0, 1.0, n + 1)[1:]
        data = np.zeros((n, model.ntens))
        data[:, 0] = 5e-3 * t
        if model.ntens > 1:
            data[:, model.ntens - 1] = 6e-3 * np.sin(np.pi * t)
        load = FieldHistory(FieldType.STRAIN, "Strain", data)
        return StrainDriver(PythonAnalyticalIntegrator(model)).run(load).step_results

    def test_closed_form_solution_satisfies_every_row(self, model):
        report = JacobianChecker(model).check_solution(self._results(model))
        assert report.passed, report.summary()
        assert set(report.rows) == {"stress", "dlambda", "ep"}
        assert report.rows["ep"].kind == "state"
        assert max(r.worst_err for r in report.rows.values()) < 1e-12
        assert report.n_points > 5

    def test_hand_newton_solution_satisfies_every_row(self):
        from manforge.models import YUKinematic3D
        from tests.benchmarks.yu_kinematic.conftest import PARAMS

        m = YUKinematic3D(**PARAMS)
        report = JacobianChecker(m).check_solution(self._results(m))
        assert report.passed, report.summary()
        assert {"stress", "dlambda", "theta", "beta"} <= set(report.rows)
        assert {"R", "q", "r", "eps_eq", "theta_max"} <= set(report.rows)

    def test_a_wrong_hand_row_is_the_only_row_reported(self):
        """A coefficient off by 2 % in the hand residual's θ row."""
        import autograd.numpy as anp
        from manforge.models import YUKinematic3D
        from tests.benchmarks.yu_kinematic.conftest import PARAMS

        class _WrongTheta(YUKinematic3D):
            def calc_residual(self, state_new, state_n, stress_trial, dlambda):
                r = super().calc_residual(state_new, state_n, stress_trial, dlambda)
                d_theta = state_new["theta"] - state_n["theta"]
                extra = anp.concatenate(
                    [anp.zeros(7), 0.02 * d_theta, anp.zeros(6)])
                return r + extra

        m = _WrongTheta(**PARAMS)
        report = JacobianChecker(m).check_solution(self._results(m))
        assert not report.passed
        assert [r.name for r in report.failed] == ["theta"]
        assert report.rows["theta"].worst_err > 1e-4
        assert "FAIL [theta]" in report.summary()
        assert "1 row off" in report.summary()

    def test_a_wrong_closed_form_is_caught(self, model):
        """There is no hand residual to compare here — only the solution."""
        from manforge.core import ReturnMappingResult

        class _Overshoot(type(model)):
            def user_defined_return_mapping(self, inc):
                rm = super().user_defined_return_mapping(inc)
                stress = inc.trial["stress"] + 1.001 * (rm.stress - inc.trial["stress"])
                return ReturnMappingResult(
                    stress=stress, state={**rm.state, "stress": stress},
                    dlambda=rm.dlambda, n_iterations=0,
                    residual_history=[], converged=True,
                )

        m = _Overshoot(**model.params)
        report = JacobianChecker(m).check_solution(self._results(m))
        assert not report.passed
        assert {r.name for r in report.failed} == {"stress", "dlambda"}

    def test_a_wrong_explicit_update_is_reported_as_a_state(self, model):
        from manforge.core import ReturnMappingResult

        class _WrongEp(type(model)):
            def user_defined_return_mapping(self, inc):
                rm = super().user_defined_return_mapping(inc)
                state = {**rm.state, "ep": inc.state_n["ep"] + 1.01 * rm.dlambda}
                return ReturnMappingResult(
                    stress=rm.stress, state=state, dlambda=rm.dlambda,
                    n_iterations=0, residual_history=[], converged=True,
                )

        m = _WrongEp(**model.params)
        report = JacobianChecker(m).check_solution(self._results(m)[:6])
        assert [r.name for r in report.failed] == ["ep"]
        assert "ep (explicit state)" in report.summary()

    def test_autodiff_solutions_pass_at_their_own_solver_tolerance(self):
        """A solution is not faulted for the tolerance its solver stopped at."""
        from manforge.models import OWKinematic3D
        from manforge.simulation import (
            FieldHistory, PythonNumericalIntegrator, StrainDriver,
        )
        from tests.fixtures.hand_block_models import OW_PARAMS

        m = OWKinematic3D(**OW_PARAMS)
        load = FieldHistory.cyclic_strain([4e-3, -4e-3], n_per_segment=10, ntens=6)
        results = StrainDriver(PythonNumericalIntegrator(m)).run(load).step_results
        report = JacobianChecker(m).check_solution(results)
        assert report.passed, report.summary()
        assert max(r.worst_err for r in report.rows.values()) < 1e-9

    def test_a_state_that_never_moves_is_listed_not_passed(self):
        """An explicit state that stays at zero says nothing about its update."""
        from manforge.core import SCALAR, Explicit
        from manforge.models import J2Isotropic
        from manforge.simulation import FieldHistory, StrainDriver

        class _WithIdleState(J2Isotropic):
            idle = Explicit(shape=SCALAR, doc="a mechanism this history never triggers")

            def update_state(self, new, dlambda, inc):
                return [*super().update_state(new, dlambda, inc),
                        self.idle(inc.state_n["idle"])]

        m = _WithIdleState(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
        load = FieldHistory.cyclic_strain([4e-3], n_per_segment=10, ntens=6)
        results = StrainDriver(PythonNumericalIntegrator(m)).run(load).step_results
        report = JacobianChecker(m).check_solution(results)
        assert report.passed, report.summary()
        assert report.not_exercised == ["idle"]
        assert "idle" not in report.rows and "ep" in report.rows
        assert "not exercised: idle" in report.summary()

    def test_nothing_checked_is_an_error_not_a_pass(self, model):
        from manforge.simulation import FieldHistory, FieldType, StrainDriver

        data = np.zeros((3, 6))
        data[:, 0] = np.linspace(1e-5, 3e-5, 3)        # elastic only
        load = FieldHistory(FieldType.STRAIN, "Strain", data)
        results = StrainDriver(PythonNumericalIntegrator(model)).run(load).step_results
        with pytest.raises(ValueError, match="no plastic step"):
            JacobianChecker(model).check_solution(results)

    def test_tol_follows_the_checker_and_can_be_overridden(self, model):
        results = self._results(model)
        strict = JacobianChecker(model, tol=1e-30)
        assert not strict.check_solution(results).passed
        assert strict.check_solution(results, tol=1e-8).passed
