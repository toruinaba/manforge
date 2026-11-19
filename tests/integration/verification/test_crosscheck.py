"""crosscheck — two integrators over one loading history."""

import numpy as np
import pytest

from manforge.core import ReturnMappingResult
from manforge.models import J2Isotropic3D
from manforge.simulation import (
    FieldHistory, FieldType, MixedDriver, PythonAnalyticalIntegrator,
    PythonNumericalIntegrator, StressDriver,
)
from manforge.verification import (
    CrosscheckResult, CrosscheckStep, crosscheck, generate_strain_history,
)


@pytest.fixture
def model():
    return J2Isotropic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)


@pytest.fixture
def strain_load(model):
    return FieldHistory(FieldType.STRAIN, "eps", generate_strain_history(model))


def _pair(model):
    return PythonNumericalIntegrator(model), PythonAnalyticalIntegrator(model)


class _Overshoot(J2Isotropic3D):
    """Closed form with the plastic correction 0.1 % too large."""

    def user_defined_return_mapping(self, inc):
        rm = super().user_defined_return_mapping(inc)
        stress = inc.trial["stress"] + 1.001 * (rm.stress - inc.trial["stress"])
        return ReturnMappingResult(
            stress=stress, state={**rm.state, "stress": stress},
            dlambda=rm.dlambda, n_iterations=0, residual_history=[], converged=True,
        )


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------

class TestAgreement:
    def test_closed_form_agrees_with_newton(self, model, strain_load):
        result = crosscheck(*_pair(model), strain_load)
        assert isinstance(result, CrosscheckResult)
        assert result.passed, result.summary()
        assert result.n_steps == len(strain_load.data)
        assert result.n_passed == result.n_steps
        assert result.first_failure is None
        assert result.max_stress_rel_err < 1e-9
        assert result.max_tangent_rel_err < 1e-6
        assert set(result.max_state_rel_err) == {"ep"}

    def test_steps_carry_both_raw_results(self, model, strain_load):
        result = crosscheck(*_pair(model), strain_load)
        step = result.steps[-1]
        assert isinstance(step, CrosscheckStep)
        assert step.result_a.n_iterations >= 1
        assert len(step.result_a.residual_history) >= 2   # Newton iterated
        assert step.a_converged and step.b_converged
        assert any(float(s.result_a.dlambda) > 0 for s in result.steps)

    def test_summary_of_a_pass(self, model, strain_load):
        text = crosscheck(*_pair(model), strain_load).summary()
        assert text.startswith(f"crosscheck: {len(strain_load.data)} steps — all agree")
        assert "ok   stress" in text and "ok   state[ep]" in text and "ok   tangent" in text

    def test_no_step_compared_is_not_a_pass(self, model):
        empty = FieldHistory(FieldType.STRAIN, "eps", np.zeros((0, 6)))
        result = crosscheck(*_pair(model), empty)
        assert not result.passed
        assert result.summary() == "crosscheck: no step was compared"


# ---------------------------------------------------------------------------
# The driver follows the load, or is given
# ---------------------------------------------------------------------------

class TestDrivers:
    def test_stress_history_uses_the_stress_driver(self, model):
        sigma = np.zeros((12, 6))
        sigma[:, 0] = np.linspace(0.0, 320.0, 12)
        load = FieldHistory(FieldType.STRESS, "sigma", sigma)
        result = crosscheck(*_pair(model), load)
        assert result.passed, result.summary()
        np.testing.assert_allclose(result.steps[-1].result_a.stress[0], 320.0, rtol=1e-8)

    def test_driver_settings_travel_with_the_factory(self, model):
        sigma = np.zeros((6, 6))
        sigma[:, 0] = np.linspace(0.0, 320.0, 6)
        load = FieldHistory(FieldType.STRESS, "sigma", sigma)
        seen = []

        def factory(integrator):
            driver = StressDriver(integrator, max_iter=30, tol=1e-10)
            seen.append(driver)
            return driver

        assert crosscheck(*_pair(model), load, driver_factory=factory).passed
        assert len(seen) == 2 and seen[0] is not seen[1]
        assert all(d.max_iter == 30 and d.tol == 1e-10 for d in seen)

    def test_mixed_control(self, model):
        """Uniaxial stress on a 3D model — there was no crosscheck for this."""
        axial = FieldHistory.cyclic_strain([4e-3, -4e-3], n_per_segment=8, ntens=1)
        result = crosscheck(
            *_pair(model), axial,
            driver_factory=lambda i: MixedDriver(i, prescribed_strain_idx=[0]),
        )
        assert result.passed, result.summary()
        lateral = np.array([s.result_a.stress[1:] for s in result.steps])
        np.testing.assert_allclose(lateral, 0.0, atol=1e-6)

    def test_single_step_from_a_given_state(self, model):
        state = model.initial_state()
        state["ep"] = np.array(2e-3)
        prestress = np.array([200.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        one = np.zeros((1, 6))
        one[0, 0] = 1e-3
        result = crosscheck(
            *_pair(model), FieldHistory(FieldType.STRAIN, "eps", one),
            initial_stress=prestress, initial_state=state,
        )
        assert result.passed and result.n_steps == 1
        assert float(result.steps[0].result_a.state["ep"]) > 2e-3
        assert float(state["ep"]) == 2e-3                 # the caller's dict is untouched


# ---------------------------------------------------------------------------
# Disagreement
# ---------------------------------------------------------------------------

class TestDisagreement:
    def test_a_wrong_candidate_fails_and_is_located(self, strain_load):
        wrong = _Overshoot(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)
        result = crosscheck(
            PythonNumericalIntegrator(wrong), PythonAnalyticalIntegrator(wrong), strain_load)
        assert not result.passed
        first = result.first_failure
        assert first is not None and first.result_b.is_plastic
        assert all(s.passed for s in result.steps[:first.index])   # elastic steps agree
        assert first.stress_rel_err > 1e-6
        text = result.summary()
        assert f"(first at step {first.index})" in text and "FAIL stress" in text

    def test_stop_on_failure_ends_at_the_first_disagreement(self, strain_load):
        wrong = _Overshoot(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)
        pair = PythonNumericalIntegrator(wrong), PythonAnalyticalIntegrator(wrong)
        full = crosscheck(*pair, strain_load)
        short = crosscheck(*pair, strain_load, stop_on_failure=True)
        assert short.first_failure is short.steps[-1]
        assert short.first_failure.index == full.first_failure.index
        assert short.n_steps < full.n_steps

    def test_tolerances_decide(self, strain_load):
        wrong = _Overshoot(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)
        pair = PythonNumericalIntegrator(wrong), PythonAnalyticalIntegrator(wrong)
        loose = crosscheck(*pair, strain_load, stress_tol=1.0, state_tol=1.0, tangent_tol=1.0)
        assert loose.passed
        assert loose.stress_tol == 1.0

    def test_nonconverged_step_is_not_agreement(self, model):
        """max_iter=1 cannot solve a plastic step; the iterate is not compared as a solution."""
        weak = PythonNumericalIntegrator(model, max_iter=1, raise_on_nonconverged=False)
        ok = PythonAnalyticalIntegrator(model)

        elastic = np.zeros((1, 6))
        elastic[0, 0] = 1e-7
        result = crosscheck(weak, ok, FieldHistory(FieldType.STRAIN, "eps", elastic))
        assert result.passed and result.n_a_nonconverged == 0

        plastic = np.zeros((1, 6))
        plastic[0, 0] = 5e-3
        result = crosscheck(weak, ok, FieldHistory(FieldType.STRAIN, "eps", plastic))
        assert not result.passed
        assert result.steps[0].a_converged is False and result.steps[0].b_converged is True
        assert result.n_a_nonconverged == 1 and result.n_b_nonconverged == 0
        assert "not converged: a at 1 steps" in result.summary()


# ---------------------------------------------------------------------------
# Lock-step: the candidate gets the reference's inputs at every step
# ---------------------------------------------------------------------------

class TestLockstep:
    @staticmethod
    def _cyclic():
        return FieldHistory.cyclic_strain([4e-3, -4e-3, 4e-3], n_per_segment=10, ntens=6)

    def test_agreement(self, model):
        result = crosscheck(*_pair(model), self._cyclic(), lockstep=True)
        assert result.passed, result.summary()
        assert result.n_steps == 30

    def test_candidate_is_given_the_references_state_every_step(self, model):
        calls = []
        ref, cand = _pair(model)
        inner = cand.stress_update

        def spy(strain_inc, stress_n, state_n):
            calls.append((np.array(strain_inc), np.array(stress_n), float(state_n["ep"])))
            return inner(strain_inc, stress_n, state_n)

        cand.stress_update = spy
        result = crosscheck(ref, cand, self._cyclic(), lockstep=True)
        assert len(calls) == result.n_steps
        for k in range(1, result.n_steps):
            prev = result.steps[k - 1].result_a
            np.testing.assert_array_equal(calls[k][1], np.asarray(prev.stress))
            assert calls[k][2] == float(prev.state["ep"])
            np.testing.assert_array_equal(
                calls[k][0], np.asarray(result.steps[k].result_a.increment.strain_inc))

    def test_a_failure_does_not_echo_into_later_steps(self):
        """Independent march: once wrong, every later step is off, the elastic
        unloading steps included.  Lock-step: only the steps where the wrong
        formula is actually used."""
        wrong = _Overshoot(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)
        pair = PythonNumericalIntegrator(wrong), PythonAnalyticalIntegrator(wrong)
        load = self._cyclic()

        independent = crosscheck(*pair, load)
        first = independent.first_failure.index
        assert all(not s.passed for s in independent.steps[first:])

        locked = crosscheck(*pair, load, lockstep=True)
        assert locked.first_failure.index == first
        failed = [s for s in locked.steps if not s.passed]
        assert failed and all(s.result_a.is_plastic for s in failed)
        elastic_later = [s for s in locked.steps[first:] if not s.result_a.is_plastic]
        assert elastic_later and all(s.passed for s in elastic_later)

    def test_works_under_stress_control(self, model):
        """The reference's driver solves for the strain; the candidate takes
        the increment it found."""
        sigma = np.zeros((10, 6))
        sigma[:, 0] = np.linspace(0.0, 320.0, 10)
        load = FieldHistory(FieldType.STRESS, "sigma", sigma)
        result = crosscheck(*_pair(model), load, lockstep=True)
        assert result.passed, result.summary()
        assert any(s.result_b.is_plastic for s in result.steps)

    def test_starts_from_the_given_state(self, model):
        state = model.initial_state()
        state["ep"] = np.array(2e-3)
        one = np.zeros((1, 6))
        one[0, 0] = 1e-3
        result = crosscheck(
            *_pair(model), FieldHistory(FieldType.STRAIN, "eps", one),
            initial_stress=np.array([200.0, 0, 0, 0, 0, 0.0]), initial_state=state,
            lockstep=True,
        )
        assert result.passed and result.n_steps == 1
        assert float(result.steps[0].result_b.state["ep"]) > 2e-3


# ---------------------------------------------------------------------------
# How errors are measured
# ---------------------------------------------------------------------------

class TestMeasure:
    def test_stress_through_zero_does_not_inflate_the_error(self, model):
        """Unloading through σ = 0: round-off there is measured against the
        stress the run has reached, not against the value near zero."""
        load = FieldHistory.cyclic_strain([4e-3, 0.0, -4e-3], n_per_segment=40, ntens=6)
        result = crosscheck(*_pair(model), load)
        assert result.passed, result.summary()
        assert result.max_stress_rel_err < 1e-9

    def test_state_at_noise_level_is_treated_as_zero(self):
        """Two results that differ by 100 % in a state of magnitude 1e-25."""
        from manforge.core import SCALAR, Explicit
        from manforge.models import J2Isotropic

        class _WithResidue(J2Isotropic):
            residue = Explicit(shape=SCALAR, doc="round-off left by a smooth gate")
            level = 1e-25

            def update_state(self, new, dlambda, inc):
                return [*super().update_state(new, dlambda, inc), self.residue(self.level)]

        class _OtherResidue(_WithResidue):
            level = 2e-25

        params = dict(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)
        load = FieldHistory.cyclic_strain([4e-3], n_per_segment=6, ntens=6)
        result = crosscheck(
            PythonNumericalIntegrator(_WithResidue(**params)),
            PythonNumericalIntegrator(_OtherResidue(**params)),
            load,
        )
        assert result.passed, result.summary()
        assert result.max_state_rel_err["residue"] < 1e-6
