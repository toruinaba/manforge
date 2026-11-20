"""check_model — every check that applies to a model, in one call."""

import numpy as np
import pytest

from returnmap.core import ReturnMappingResult
from returnmap.models import (
    AFKinematic3D, J2Isotropic1D, J2Isotropic3D, OWKinematic1D, YUKinematic1D,
)
from returnmap.simulation import FieldHistory, FieldType
from returnmap.verification import (
    ModelCheck, ModelReport, check_model, standard_histories,
)
from tests.fixtures.hand_block_models import (
    OW_PARAMS, OWWithApproxBlock, OWWithHandBlocks, OWWithWrongBlock,
)

HAND_WRITTEN = ("hand-written solution", "hand-written vs autodiff", "hand-written tangent")


class _Overshoot(J2Isotropic3D):
    """Closed form with the plastic correction 0.1 % too large."""

    def user_defined_return_mapping(self, inc):
        rm = super().user_defined_return_mapping(inc)
        stress = inc.trial["stress"] + 1.001 * (rm.stress - inc.trial["stress"])
        return ReturnMappingResult(
            stress=stress, state={**rm.state, "stress": stress},
            dlambda=rm.dlambda, n_iterations=1, residual_history=[1.0, 0.0],
            converged=True,
        )


class _TangentOff(J2Isotropic3D):
    """Right return mapping, hand-written tangent 1 % too stiff."""

    def user_defined_tangent(self, new, dlambda, inc):
        return 1.01 * super().user_defined_tangent(new, dlambda, inc)


class _GoingBack(J2Isotropic3D):
    """Hand-written tangent 1 % too stiff while the first strain component decreases."""

    def going_back(self, inc):
        return inc.strain_inc[0] < 0.0

    def user_defined_tangent(self, new, dlambda, inc):
        scale = 1.01 if self.going_back(inc) else 1.0
        return scale * super().user_defined_tangent(new, dlambda, inc)


def statuses(report):
    return {check.name: check.status for check in report.checks}


# ---------------------------------------------------------------------------
# What is checked follows from what the model declares
# ---------------------------------------------------------------------------

class TestWhatApplies:
    def test_a_python_only_model_passes_on_convergence_and_tangent(self):
        """Stopping at the autodiff model is a legitimate place to be."""
        report = check_model(AFKinematic3D.at_defaults())
        assert isinstance(report, ModelReport)
        assert report.passed, report.summary()
        assert statuses(report) == {
            "converges": "passed", "tangent": "passed",
            "hand blocks": "not declared",
            **{name: "not declared" for name in HAND_WRITTEN},
        }

    def test_a_hand_written_return_mapping_is_checked_three_ways(self):
        report = check_model(J2Isotropic3D.at_defaults())
        assert report.passed, report.summary()
        assert all(report[name].status == "passed" for name in HAND_WRITTEN)
        assert report["hand blocks"].status == "not declared"

    def test_declared_hand_blocks_are_checked(self):
        report = check_model(OWWithHandBlocks(**OW_PARAMS))
        assert report.passed, report.summary()
        assert report["hand blocks"].status == "passed"
        assert "blocks at" in report["hand blocks"].detail

    def test_a_block_declared_approximate_is_held_to_its_own_tolerance(self):
        report = check_model(OWWithApproxBlock(**OW_PARAMS))
        assert report["hand blocks"].status == "passed", report.summary()
        assert "1 declared approximate" in report["hand blocks"].detail


# ---------------------------------------------------------------------------
# Each kind of mistake fails the check that is about it, and only that one
# ---------------------------------------------------------------------------

class TestWhatFails:
    def test_a_wrong_hand_block(self):
        report = check_model(OWWithWrongBlock(**OW_PARAMS))
        assert not report.passed
        assert [c.name for c in report.checks if c.status == "failed"] == ["hand blocks"]
        assert "FAIL" in report["hand blocks"].reports["all"].summary()

    def test_a_wrong_hand_written_return_mapping(self):
        report = check_model(_Overshoot.at_defaults())
        assert not report.passed
        failed = [c.name for c in report.checks if c.status == "failed"]
        assert "hand-written solution" in failed
        assert "hand-written vs autodiff" in failed
        # The autodiff path of the same model is untouched.
        assert report["converges"].status == report["tangent"].status == "passed"

    def test_a_wrong_hand_written_tangent(self):
        report = check_model(_TangentOff.at_defaults())
        failed = [c.name for c in report.checks if c.status == "failed"]
        # The stress is right, so the solution passes; both tangent checks see it.
        assert failed == ["hand-written vs autodiff", "hand-written tangent"]
        assert "tangent 1.0e-02" in report["hand-written vs autodiff"].detail

    def test_parameters_the_model_cannot_be_integrated_at(self):
        model = OWKinematic1D(E=206e3, nu=0.3, sigma_y0=250.0, C_k=1e9, gamma=1e12)
        jump = {"jump": FieldHistory(FieldType.STRAIN, np.array([0.0, 0.05]))}
        report = check_model(model, jump)
        assert report["converges"].status == "failed"
        assert "jump: stopped at step" in report["converges"].detail

    def test_a_history_that_never_yields_checks_nothing(self):
        """Passing because no plastic step was taken would be a vacuous pass."""
        elastic = {"elastic": FieldHistory(FieldType.STRAIN, np.linspace(0.0, 1e-5, 5))}
        report = check_model(J2Isotropic1D.at_defaults(), elastic)
        assert not report.passed
        assert "does not reach yield" in report["converges"].detail
        assert report["tangent"].status == "failed"


# ---------------------------------------------------------------------------
# A hand-written tangent declared approximate (hand_tangent)
# ---------------------------------------------------------------------------

class TestDeclaredTangent:
    def test_undeclared_the_same_tangent_fails(self):
        report = check_model(_GoingBack.at_defaults())
        failed = [c.name for c in report.checks if c.status == "failed"]
        assert failed == ["hand-written vs autodiff", "hand-written tangent"]

    def test_declared_on_those_steps_they_are_left_out_and_counted(self):
        from returnmap.core import hand_tangent

        class Declared(_GoingBack):
            @hand_tangent(approx_when="going_back", note="1 % stiff while going back")
            def user_defined_tangent(self, new, dlambda, inc):
                return super().user_defined_tangent(new, dlambda, inc)

        report = check_model(Declared.at_defaults())
        assert report.passed, report.summary()
        assert report.approximations == ["1 % stiff while going back"]
        assert "declared approximate: 1 % stiff while going back" in report.summary()

        versus = report["hand-written vs autodiff"]
        n_off = sum(r.n_steps - r.n_passed for r in versus.reports.values())
        assert n_off > 0 and f"; {n_off} step(s) declared approximate" in versus.detail
        # What is reported as the tangent error is that of the steps held exact.
        assert "tangent 1.0e-02" not in versus.detail
        assert "declared approximate" in report["hand-written tangent"].detail

    def test_the_declaration_does_not_cover_other_steps(self):
        """Declared approximate while going back, wrong while going forward."""
        from returnmap.core import hand_tangent

        class WrongWay(J2Isotropic3D):
            def going_back(self, inc):
                return inc.strain_inc[0] < 0.0

            @hand_tangent(approx_when="going_back", note="1 % stiff while going back")
            def user_defined_tangent(self, new, dlambda, inc):
                scale = 1.0 if self.going_back(inc) else 1.01
                return scale * super().user_defined_tangent(new, dlambda, inc)

        report = check_model(WrongWay.at_defaults())
        assert report["hand-written vs autodiff"].status == "failed"
        assert report["hand-written tangent"].status == "failed"

    @pytest.mark.parametrize("approx, passes", [(2e-2, True), (1e-3, False)])
    def test_with_approx_the_declared_steps_are_held_to_it(self, approx, passes):
        from returnmap.core import hand_tangent

        class HeldTo(_GoingBack):
            @hand_tangent(approx=approx, approx_when="going_back", note="1 % stiff")
            def user_defined_tangent(self, new, dlambda, inc):
                return super().user_defined_tangent(new, dlambda, inc)

        report = check_model(HeldTo.at_defaults())
        assert report["hand-written vs autodiff"].passed is passes
        assert report["hand-written tangent"].passed is passes

    @pytest.mark.parametrize("approx, passes", [(2e-2, True), (1e-3, False)])
    def test_approx_alone_is_the_tolerance_on_every_step(self, approx, passes):
        from returnmap.core import hand_tangent

        class Everywhere(_TangentOff):
            @hand_tangent(approx=approx, note="1 % stiff throughout")
            def user_defined_tangent(self, new, dlambda, inc):
                return super().user_defined_tangent(new, dlambda, inc)

        report = check_model(Everywhere.at_defaults())
        assert report["hand-written vs autodiff"].passed is passes
        assert report["hand-written tangent"].passed is passes
        if passes:
            assert f"(tol {approx:.0e})" in report["hand-written tangent"].detail

    def test_a_tangent_declared_approximate_everywhere_it_is_sampled_is_held_to_nothing(self):
        from returnmap.core import hand_tangent

        class Always(J2Isotropic3D):
            def always(self):
                return True

            @hand_tangent(approx_when="always", note="never exact")
            def user_defined_tangent(self, new, dlambda, inc):
                return super().user_defined_tangent(new, dlambda, inc)

        report = check_model(Always.at_defaults())
        assert report["hand-written tangent"].status == "failed"
        assert "held to nothing" in report["hand-written tangent"].detail


# ---------------------------------------------------------------------------
# Loading histories
# ---------------------------------------------------------------------------

class TestHistories:
    def test_standard_histories_scale_with_the_model(self):
        """Yield strain is 250/210000 here and ten times that for the stiffer
        yield stress, and the histories follow."""
        soft = standard_histories(J2Isotropic3D.at_defaults())
        hard = standard_histories(J2Isotropic3D.at_defaults(sigma_y0=2500.0))
        assert list(soft) == ["uniaxial", "multiaxial"]
        np.testing.assert_allclose(hard["uniaxial"].data, 10.0 * soft["uniaxial"].data, rtol=1e-6)
        np.testing.assert_allclose(
            hard["multiaxial"].data, 10.0 * soft["multiaxial"].data, rtol=1e-6)

    def test_the_multiaxial_history_drives_every_component_and_turns(self):
        data = standard_histories(J2Isotropic3D.at_defaults())["multiaxial"].data
        assert data.shape == (24, 6)
        assert np.all(np.max(np.abs(data), axis=0) > 0.0)
        np.testing.assert_allclose(data[-1], 0.0, atol=1e-18)
        # Not proportional: the direction after the first corner is another one.
        first, second = data[5], data[11] - data[5]
        cosine = first @ second / np.linalg.norm(first) / np.linalg.norm(second)
        assert abs(cosine) < 0.9

    def test_a_one_component_model_has_only_the_uniaxial_history(self):
        assert list(standard_histories(J2Isotropic1D.at_defaults())) == ["uniaxial"]

    def test_histories_the_family_declares_are_added(self):
        """returnmap.models.yu_kinematic.verification defines histories(model)."""
        report = check_model(YUKinematic1D.at_defaults())
        assert list(report.histories) == ["uniaxial", "reversed 2 %"]
        assert report.passed, report.summary()

    def test_given_histories_replace_the_standard_and_declared_ones(self):
        only = {"mine": FieldHistory.cyclic_strain([0.01, -0.01], n_per_segment=8)}
        report = check_model(YUKinematic1D.at_defaults(), only)
        assert list(report.histories) == ["mine"]
        assert report.histories["mine"] > 0

    def test_a_model_outside_any_package_has_no_declared_histories(self):
        class Local(J2Isotropic1D):
            pass

        assert list(check_model(Local.at_defaults()).histories) == ["uniaxial"]


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

class TestReport:
    @pytest.fixture(scope="class")
    def report(self):
        return check_model(J2Isotropic3D.at_defaults())

    def test_summary_has_one_row_per_check_and_names_the_histories(self, report):
        lines = report.summary().splitlines()
        assert lines[0] == "J2Isotropic3D: every check that applies passes"
        assert [line.split()[0] for line in lines[1:-1]] == ["ok", "ok", "—", "ok", "ok", "ok"]
        assert lines[-1].strip().startswith("histories: uniaxial (")
        assert str(report) == report.summary()

    def test_a_failure_is_named_in_the_first_line(self):
        report = check_model(_TangentOff.at_defaults())
        assert report.summary().splitlines()[0] == (
            "_TangentOff: 2 check(s) failed — hand-written vs autodiff, hand-written tangent"
        )

    def test_a_check_is_reached_by_name_with_the_reports_behind_it(self, report):
        check = report["hand-written vs autodiff"]
        assert isinstance(check, ModelCheck)
        assert set(check.reports) == {"uniaxial", "multiaxial"}
        assert check.reports["multiaxial"].passed
        with pytest.raises(KeyError, match="no check named 'umat'"):
            report["umat"]

    def test_tangents_are_differenced_at_a_few_plastic_steps_per_history(self, report):
        assert len(report["tangent"].reports) == 8
        fewer = check_model(J2Isotropic3D.at_defaults(), n_tangent_points=1)
        assert len(fewer["tangent"].reports) == 2


def test_a_class_is_not_a_model():
    with pytest.raises(TypeError, match="at_defaults"):
        check_model(J2Isotropic3D)
