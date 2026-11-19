"""Tests for compare_jacobians, check_hand_blocks and check_solution."""

import autograd.numpy as anp
import numpy as np
import pytest

from manforge.core import hand_block
from manforge.models import J2Isotropic3D
from manforge.simulation import (
    PythonAnalyticalIntegrator, PythonIntegrator, PythonNumericalIntegrator,
    hand_jacobian,
)
from manforge.verification import (
    check_hand_blocks,
    check_solution,
    compare_jacobians,
    JacobianComparisonReport,
)


def _result(model, strain_scale):
    deps = np.zeros(model.ntens)
    deps[0] = strain_scale
    state0 = model.initial_state()
    return PythonIntegrator(model).stress_update(deps, np.zeros(model.ntens), state0), state0


class TestCompareJacobiansJ2:
    def test_returns_result_type(self, model):
        r_ad, state0 = _result(model, 3e-3)
        r_an, _ = _result(model, 3e-3)
        out = compare_jacobians(model, r_ad, r_an)
        assert isinstance(out, JacobianComparisonReport)

    def test_identical_results_zero_error(self, model):
        r, state0 = _result(model, 3e-3)
        out = compare_jacobians(model, r, r)
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
        out = compare_jacobians(model, r_ad, r_an_ud, tol=1e-8)
        assert out.passed, (
            f"max_rel_err={out.max_rel_err:.3e}, blocks={out.blocks}"
        )

    def test_round_off_entries_are_not_a_mismatch(self, model):
        """The two solutions differ by 1e-13 in stress.  Entry by entry, a
        Jacobian component that is zero on one side and 1e-17 on the other
        used to read as a relative error of 2."""
        deps = np.zeros(6)
        deps[0] = 4e-3
        state0 = model.initial_state()
        r_num = PythonNumericalIntegrator(model).stress_update(deps, np.zeros(6), state0)
        r_closed = PythonAnalyticalIntegrator(model).stress_update(deps, np.zeros(6), state0)
        out = compare_jacobians(model, r_num, r_closed)
        assert out.passed
        assert out.max_rel_err < 1e-12

    def test_compare_with_a_result_that_reports_no_dlambda(self, model):
        """A Fortran UMAT result carries Δλ = NaN.  The comparison is the one
        the docstring recommends for such results, and it used to return NaN."""
        from manforge.simulation import FortranIntegrator
        from tests.integration.simulation.integrator.test_fortran import _FakeUmat

        deps = np.zeros(6)
        deps[0] = 4e-3
        state0 = model.initial_state()
        r_py = PythonIntegrator(model).stress_update(deps, np.zeros(6), state0)
        r_f = FortranIntegrator.from_model(_FakeUmat(), "j2", model).stress_update(
            deps, np.zeros(6), state0)
        assert np.isnan(float(r_f.dlambda))

        out = compare_jacobians(model, r_py, r_f)
        assert out.passed
        assert out.max_rel_err == 0.0

    def test_a_real_difference_still_fails(self, model):
        deps = np.zeros(6)
        deps[0] = 4e-3
        state0 = model.initial_state()
        r_a = PythonIntegrator(model).stress_update(deps, np.zeros(6), state0)
        r_b = PythonIntegrator(model).stress_update(1.5 * deps, np.zeros(6), state0)
        assert not compare_jacobians(model, r_a, r_b).passed

    def test_elastic_step_does_not_raise(self, model):
        """Elastic step (dlambda=0) should not raise in compare."""
        r, state0 = _result(model, 1e-5)
        assert not r.is_plastic
        out = compare_jacobians(model, r, r)
        assert out.passed

    def test_blocks_dict_populated(self, model):
        """blocks dict should contain at least the four base block keys."""
        r, state0 = _result(model, 3e-3)
        out = compare_jacobians(model, r, r)
        for key in ("stress::stress", "stress::dlambda", "dlambda::stress", "dlambda::dlambda"):
            assert key in out.blocks

    def test_max_rel_err_equals_max_of_blocks(self, model):
        r, state0 = _result(model, 3e-3)
        out = compare_jacobians(model, r, r)
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
        jac = PythonIntegrator(model).jacobian(result)
        assert jac.full.shape == (7, 7)

    def test_compare_forwards_strain_inc(self):
        model, deps, state0, result = self._plastic_result()
        out = compare_jacobians(model, result, result)
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
        jac = PythonIntegrator(model).jacobian(result)
        assert jac.full.shape == (7, 7)

    def test_result_without_increment_is_rejected(self, model):
        """A Fortran result reports no trial state; its residual is unknown."""
        from dataclasses import replace
        result = replace(self._step(model), increment=None)
        with pytest.raises(ValueError, match="no increment"):
            PythonIntegrator(model).jacobian(result)

    def test_comparison_report_names_the_blocks(self, model):
        from dataclasses import replace
        from manforge.core import ReturnMappingResult

        result = PythonNumericalIntegrator(model).stress_update(
            anp.array([3e-3, 0.0, 0.0, 1e-3, 0.0, 0.0]), anp.zeros(6), model.initial_state())
        same = compare_jacobians(model, result, result)
        assert same.passed and same.failed == []
        assert "jacobian comparison: 4 blocks — all agree" in same.summary()
        assert str(same) == same.summary()

        rm = result.return_mapping
        moved = replace(result, return_mapping=ReturnMappingResult(
            stress=rm.stress * 1.01, state={**rm.state, "stress": rm.stress * 1.01},
            dlambda=rm.dlambda, n_iterations=rm.n_iterations,
            residual_history=rm.residual_history, converged=True,
        ))
        differs = compare_jacobians(model, result, moved, tol=1e-6)
        assert not differs.passed and differs.tol == 1e-6
        assert differs.failed[0] in differs.blocks
        assert differs.blocks[differs.failed[0]] == differs.max_rel_err
        assert "FAIL " + differs.failed[0] in differs.summary()

    def test_compare_evaluates_both_on_the_first_results_increment(self, model):
        from dataclasses import replace
        result = self._step(model)
        no_increment = replace(result, increment=None)
        out = compare_jacobians(model, result, no_increment)
        assert out.passed and out.max_rel_err == 0.0
        with pytest.raises(ValueError, match="result_a must come from a Python integrator"):
            compare_jacobians(model, no_increment, result)


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
        report = check_hand_blocks(m, results)

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

    def test_tol_is_what_exact_blocks_are_held_to(self):
        from tests.fixtures.hand_block_models import OW_PARAMS, OWWithHandBlocks

        m = OWWithHandBlocks(**OW_PARAMS)
        results = self._results(m)
        report = check_hand_blocks(m, results, tol=1e-30)
        assert not report.passed                      # round-off exceeds 1e-30
        assert {b.tol for b in report.blocks.values()} == {1e-30}
        assert check_hand_blocks(m, results).passed   # default 1e-8

    def test_a_block_that_returns_nan_fails(self):
        """NaN loses every comparison, so past the first step it used to
        drop out of the running worst error and the block passed."""
        from tests.fixtures.hand_block_models import OW_PARAMS, OWWithHandBlocks

        class _NanLater(OWWithHandBlocks):
            n_calls = 0

            @hand_block("dlambda", "stress")
            def dRy_ds(self, state):
                type(self).n_calls += 1
                block = super().dRy_ds(state)
                return block * np.nan if type(self).n_calls > 1 else block

        m = _NanLater(**OW_PARAMS)
        report = check_hand_blocks(m, self._results(m))
        assert not report.passed
        assert [(b.row, b.col) for b in report.failed] == [("dlambda", "stress")]
        assert report.blocks[("dlambda", "stress")].worst_err == np.inf

    def test_a_wrong_block_is_found_by_name(self):
        """One sign error among eight blocks: named, not just 'the Jacobian is off'."""
        from tests.fixtures.hand_block_models import OW_PARAMS, OWWithWrongBlock

        m = OWWithWrongBlock(**OW_PARAMS)
        report = check_hand_blocks(m, self._results(m))

        assert not report.passed
        assert [(b.row, b.col, b.method) for b in report.failed] == [("ep", "dlambda", "dRep_dl")]
        assert report.failed[0].worst_err == pytest.approx(2.0)     # +1 against −1
        assert "FAIL [ep][dlambda] dRep_dl" in report.summary()

    def test_declared_approximation_is_held_to_its_own_tolerance(self):
        from tests.fixtures.hand_block_models import OW_PARAMS, OWWithApproxBlock

        m = OWWithApproxBlock(**OW_PARAMS)
        report = check_hand_blocks(m, self._results(m), tol=1e-10)

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
        report = check_hand_blocks(m, self._results(m))
        assert [b.method for b in report.failed] == ["dRep_dep"]

    def test_nothing_checked_is_an_error_not_a_pass(self, model):
        from tests.fixtures.hand_block_models import OW_PARAMS, OWWithHandBlocks

        with pytest.raises(ValueError, match="declares no hand-derived block"):
            check_hand_blocks(model, self._results(model))

        m = OWWithHandBlocks(**OW_PARAMS)
        elastic_only = [r for r in self._results(m) if not r.is_plastic]
        with pytest.raises(ValueError, match="no plastic step"):
            check_hand_blocks(m, elastic_only)

    @pytest.mark.parametrize("unit", [1e-9, 1.0, 1e6])
    def test_error_does_not_depend_on_the_units(self, unit):
        """A block 1e-6 off is 1e-6 off in Pa, MPa or GPa.

        ``∂R_σ/∂Δλ = C n`` carries the unit of stress.  Measured against
        ``max(max|ad|, 1)`` its error shrank with the unit, and a model posed
        in small units passed whatever the block returned.
        """
        from manforge.core import hand_block
        from tests.fixtures.hand_block_models import OW_PARAMS, OWWithHandBlocks

        class _SlightlyOff(OWWithHandBlocks):
            @hand_block("stress", "dlambda")
            def dRs_dl(self, state):
                exact = self.elastic_stiffness(state) @ self.flow(state).strain_like
                return (1.0 + 1e-6) * np.asarray(exact, dtype=float)

        params = {k: v * unit if k in ("E", "sigma_y0", "C_k") else v
                  for k, v in OW_PARAMS.items()}
        m = _SlightlyOff(**params)
        report = check_hand_blocks(m, self._results(m))

        block = report.blocks[("stress", "dlambda")]
        assert block.worst_err == pytest.approx(1e-6, rel=1e-3)
        assert [b.method for b in report.failed] == ["dRs_dl"]

    def test_error_is_relative_to_the_blocks_size_over_the_history(self):
        """The largest difference over the steps, against the largest reference
        over the steps — so a step where the block is small does not turn
        round-off into a large error."""
        from manforge.core import hand_block
        from tests.fixtures.hand_block_models import OW_PARAMS, OWWithHandBlocks

        class _Offset(OWWithHandBlocks):
            @hand_block("stress", "dlambda")
            def dRs_dl(self, state):
                exact = self.elastic_stiffness(state) @ self.flow(state).strain_like
                return np.asarray(exact, dtype=float) + 1.0     # 1 MPa on every entry

        m = _Offset(**OW_PARAMS)
        results = self._results(m)
        report = check_hand_blocks(m, results)

        integrator = PythonNumericalIntegrator(m)
        largest = max(
            np.abs(integrator.jacobian(r).part["stress"]["dlambda"]).max()
            for r in results if r.is_plastic
        )
        assert report.blocks[("stress", "dlambda")].worst_err == pytest.approx(1.0 / largest)

    def test_a_block_that_is_zero_is_held_to_the_size_its_row_implies(self):
        """``∂f/∂Δλ`` is zero for this model, so there is no ``max|ad|`` to
        divide by.  The block is measured against what its row and column
        imply: round-off passes, a value that is not zero does not."""
        from manforge.core import hand_block
        from tests.fixtures.hand_block_models import OW_PARAMS, OWWithHandBlocks

        def model_returning(value):
            class _M(OWWithHandBlocks):
                @hand_block("dlambda", "dlambda")
                def dRy_dl(self):
                    return value
            return _M(**OW_PARAMS)

        m = model_returning(0.0)
        results = self._results(m)
        integrator = PythonNumericalIntegrator(m)
        assert all(float(integrator.jacobian(r).part["dlambda"]["dlambda"]) == 0.0
                   for r in results if r.is_plastic)
        assert check_hand_blocks(m, results).blocks[("dlambda", "dlambda")].worst_err == 0.0

        noise = check_hand_blocks(model_returning(1e-10), results)
        assert noise.blocks[("dlambda", "dlambda")].passed, noise.summary()
        assert 0.0 < noise.blocks[("dlambda", "dlambda")].worst_err < 1e-12

        wrong = check_hand_blocks(model_returning(10.0), results)
        assert [b.method for b in wrong.failed] == ["dRy_dl"]
        assert np.isfinite(wrong.blocks[("dlambda", "dlambda")].worst_err)

    def test_hand_returns_the_assembled_matrix(self):
        from tests.fixtures.hand_block_models import OW_PARAMS, OWWithWrongBlock

        m = OWWithWrongBlock(**OW_PARAMS)
        result = [r for r in self._results(m) if r.is_plastic][-1]
        hand, ref = hand_jacobian(m, result), PythonIntegrator(m).jacobian(result)

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
        report = check_solution(model, self._results(model))
        assert report.passed, report.summary()
        assert set(report.rows) == {"stress", "dlambda", "ep"}
        assert report.rows["ep"].kind == "state"
        assert max(r.worst_err for r in report.rows.values()) < 1e-12
        assert report.n_points > 5

    def test_hand_newton_solution_satisfies_every_row(self):
        from manforge.models import YUKinematic3D
        from tests.benchmarks.yu_kinematic.conftest import PARAMS

        m = YUKinematic3D(**PARAMS)
        report = check_solution(m, self._results(m))
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
        report = check_solution(m, self._results(m))
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
        report = check_solution(m, self._results(m))
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
        report = check_solution(m, self._results(m)[:6])
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
        report = check_solution(m, results)
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
        report = check_solution(m, results)
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
            check_solution(model, results)

    def test_tol_is_what_every_line_is_held_to(self, model):
        results = self._results(model)
        report = check_solution(model, results, tol=1e-30)
        assert not report.passed                      # round-off exceeds 1e-30
        assert {r.tol for r in report.rows.values()} == {1e-30}
        assert check_solution(model, results).passed  # default 1e-8

    # -- what a solution is --------------------------------------------------

    def test_a_non_converged_iterate_is_not_checked_as_a_solution(self):
        """Its own residual norm used to be taken as the allowance, so an
        iterate 190 away from the root came back 'every row satisfied'."""
        from manforge.models import OWKinematic3D
        from manforge.simulation import FieldHistory, FieldType, StrainDriver
        from tests.fixtures.hand_block_models import OW_PARAMS

        m = OWKinematic3D(**OW_PARAMS)
        data = np.zeros((3, 6))
        data[:, 0] = np.linspace(3e-3, 8e-3, 3)
        weak = PythonNumericalIntegrator(m, max_iter=2, raise_on_nonconverged=False)
        steps = list(StrainDriver(weak).iter_run(FieldHistory(FieldType.STRAIN, "eps", data)))
        iterate = [s.result for s in steps if not s.converged]
        assert iterate, "the premise: max_iter=2 leaves a step unsolved"

        with pytest.raises(ValueError, match="did not converge: it is a Newton iterate"):
            check_solution(m, iterate)

    def test_a_single_entry_history_buys_no_allowance(self, model):
        """A closed form that reports only the trial residual has not
        iterated: there is no solver tolerance to allow for, and the trial
        residual must not be mistaken for one."""
        from manforge.core import ReturnMappingResult

        class _Overshoot(type(model)):
            def user_defined_return_mapping(self, inc):
                rm = super().user_defined_return_mapping(inc)
                stress = inc.trial["stress"] + 1.05 * (rm.stress - inc.trial["stress"])
                return ReturnMappingResult(
                    stress=stress, state={**rm.state, "stress": stress},
                    dlambda=rm.dlambda, n_iterations=0,
                    residual_history=[rm.residual_history[0]], converged=True,
                )

        m = _Overshoot(**model.params)
        assert not check_solution(m, self._results(m)).passed

    def test_substepped_steps_are_left_out_and_counted(self):
        """A sub-stepped result ends a chain of smaller steps; it is not a
        root of the undivided increment's residual, and used to be faulted."""
        from manforge.models import OWKinematic3D
        from manforge.simulation import (
            FieldHistory, FieldType, StrainDriver, SubSteppingIntegrator,
        )
        from tests.fixtures.hand_block_models import OW_PARAMS

        m = OWKinematic3D(**OW_PARAMS)
        t = np.linspace(0.0, 1.0, 7)[1:]
        data = np.zeros((6, 6))
        data[:, 0] = 1.2e-2 * t
        data[:, 5] = 1.0e-2 * np.sin(np.pi * t)
        sub = SubSteppingIntegrator(PythonNumericalIntegrator(m, max_iter=5))
        results = StrainDriver(sub).run(FieldHistory(FieldType.STRAIN, "eps", data)).step_results
        n_sub = sum(r.n_substeps > 1 for r in results if r.is_plastic)
        assert 0 < n_sub < len(results), "the premise: some steps divided, some not"

        report = check_solution(m, results)
        assert report.passed, report.summary()
        assert report.n_substepped == n_sub
        assert f"left out: {n_sub} sub-stepped steps" in report.summary()

        only_sub = [r for r in results if r.n_substeps > 1]
        with pytest.raises(ValueError, match="every plastic step in results was sub-stepped"):
            check_solution(m, only_sub)

    def test_fortran_results_are_refused_for_the_right_reason(self, model):
        from manforge.simulation import (
            FieldHistory, FieldType, FortranIntegrator, StrainDriver,
        )
        from tests.integration.simulation.integrator.test_fortran import _FakeUmat

        data = np.zeros((4, 6))
        data[:, 0] = np.linspace(1e-3, 4e-3, 4)
        umat = FortranIntegrator.from_model(_FakeUmat(), "j2", model)
        results = StrainDriver(umat).run(FieldHistory(FieldType.STRAIN, "eps", data)).step_results
        with pytest.raises(ValueError, match="carries no increment"):
            check_solution(model, results)

    # -- explicit states -----------------------------------------------------

    @pytest.mark.parametrize("unit", [1.0, 1.0e6], ids=["MPa", "Pa"])
    def test_a_wrong_explicit_update_fails_in_any_stress_unit(self, unit):
        """Whether a state was exercised is judged from the state itself.
        It used to be compared with the largest unknown — a strain against a
        stress — so in Pa a plastic strain of 5e-4 counted as 'never moved'
        and a wrong update passed."""
        from manforge.core import ReturnMappingResult
        from manforge.simulation import (
            FieldHistory, FieldType, PythonAnalyticalIntegrator, StrainDriver,
        )

        class _DoubleEp(J2Isotropic3D):
            def user_defined_return_mapping(self, inc):
                rm = super().user_defined_return_mapping(inc)
                state = {**rm.state, "ep": inc.state_n["ep"] + 2.0 * rm.dlambda}
                return ReturnMappingResult(
                    stress=rm.stress, state=state, dlambda=rm.dlambda,
                    n_iterations=1, residual_history=[1.0, 0.0], converged=True,
                )

        m = _DoubleEp(E=210e3 * unit, nu=0.3, sigma_y0=250.0 * unit, H=1e3 * unit)
        data = np.zeros((6, 6))
        data[:, 0] = np.linspace(1.5e-3, 1.9e-3, 6)        # just past yield
        results = StrainDriver(PythonAnalyticalIntegrator(m)).run(
            FieldHistory(FieldType.STRAIN, "eps", data)).step_results
        assert float(results[-1].state["ep"]) < 1e-3

        report = check_solution(m, results)
        assert report.not_exercised == []
        assert [r.name for r in report.failed] == ["ep"]

    def test_a_nan_state_fails(self, model):
        from manforge.core import ReturnMappingResult

        class _NanEp(type(model)):
            def user_defined_return_mapping(self, inc):
                rm = super().user_defined_return_mapping(inc)
                if float(inc.state_n["ep"]) == 0.0:
                    return rm                       # first plastic step is fine
                return ReturnMappingResult(
                    stress=rm.stress, state={**rm.state, "ep": np.array(np.nan)},
                    dlambda=rm.dlambda, n_iterations=1,
                    residual_history=[1.0, 0.0], converged=True,
                )

        m = _NanEp(**model.params)
        d = np.zeros(6)
        d[0] = 2e-3
        integ = PythonAnalyticalIntegrator(m, raise_on_nonconverged=False)
        first = integ.stress_update(d, np.zeros(6), m.initial_state())
        second = integ.stress_update(d, np.asarray(first.stress), first.state)
        # The integrator itself now refuses a non-finite state...
        assert first.converged and not second.converged
        # ...and a result that got past it some other way still fails the check.
        import dataclasses
        forced = dataclasses.replace(
            second, return_mapping=dataclasses.replace(second.return_mapping, converged=True))
        report = check_solution(m, [first, forced])
        assert not report.passed
        assert report.rows["ep"].worst_err == np.inf
