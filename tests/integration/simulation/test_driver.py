"""Tests for simulation/driver.py — StrainDriver, StressDriver, MixedDriver."""

import autograd.numpy as anp
import numpy as np
import pytest

from returnmap.core import SOLID_3D, UNIAXIAL_1D, StressUpdateResult
from returnmap.models import J2Isotropic3D, J2Isotropic1D
from returnmap.simulation import (
    StrainDriver, StressDriver, MixedDriver, PythonIntegrator,
    PythonNumericalIntegrator, PythonAnalyticalIntegrator, DriverResult, DriverStep,
    FieldHistory, FieldType,
)


# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model_3d():
    return J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)


@pytest.fixture
def model_1d():
    return J2Isotropic1D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)


@pytest.fixture
def integ_3d(model_3d):
    return PythonNumericalIntegrator(model_3d)


@pytest.fixture
def integ_1d(model_1d):
    return PythonNumericalIntegrator(model_1d)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strain_load(n_steps=20, max_strain=3e-3):
    strains = np.linspace(0, max_strain, n_steps)
    return FieldHistory(FieldType.STRAIN, "Strain", strains)


def _multiaxial_strain_load(integrator, n_steps=20, max_strain=3e-3):
    data = np.zeros((n_steps, integrator.ntens))
    data[:, 0] = np.linspace(0, max_strain, n_steps)
    return FieldHistory(FieldType.STRAIN, "Strain", data)


def _stress_load_1d(integrator, n_steps=20, max_stress=300.0):
    data = np.zeros((n_steps, integrator.ntens))
    data[:, 0] = np.linspace(0, max_stress, n_steps)
    return FieldHistory(FieldType.STRESS, "Stress", data)


def _uniaxial_strain_load(n_steps=20, max_strain=3e-3):
    strains = np.linspace(0, max_strain, n_steps)
    return FieldHistory(FieldType.STRAIN, "Strain", strains)


def _uniaxial_stress_load(model, n_steps=20):
    max_stress = 300.0
    stresses = np.zeros((n_steps, model.ntens))
    stresses[:, 0] = np.linspace(0, max_stress, n_steps)
    return FieldHistory(FieldType.STRESS, "Stress", stresses)


def stress_load(data):
    return FieldHistory(FieldType.STRESS, "Stress", data)


def strain_load(data, name="Strain"):
    return FieldHistory(FieldType.STRAIN, name, data)


# ===========================================================================
# TestIterRun: DriverBase.iter_run and DriverStep
# ===========================================================================

class TestStrainDriverIterRun:
    def test_yields_driver_step_instances(self, model):
        driver = StrainDriver(PythonNumericalIntegrator(model))
        for step in driver.iter_run(_strain_load()):
            assert isinstance(step, DriverStep)

    def test_step_index_sequential(self, model):
        driver = StrainDriver(PythonNumericalIntegrator(model))
        indices = [s.i for s in driver.iter_run(_strain_load(n_steps=10))]
        assert indices == list(range(10))

    def test_strain_shape(self, model):
        driver = StrainDriver(PythonNumericalIntegrator(model))
        for step in driver.iter_run(_strain_load()):
            assert step.strain.shape == (model.ntens,)

    def test_converged_always_true(self, model):
        driver = StrainDriver(PythonNumericalIntegrator(model))
        for step in driver.iter_run(_strain_load()):
            assert step.converged is True
            assert step.n_outer_iter == 1
            assert step.residual_inf == 0.0

    def test_matches_run_uniaxial(self, model):
        load = _strain_load(n_steps=25)
        integrator = PythonNumericalIntegrator(model)
        result_run = StrainDriver(integrator).run(load)
        steps = list(StrainDriver(integrator).iter_run(load))

        assert len(steps) == len(result_run.step_results)
        for step, rm in zip(steps, result_run.step_results):
            np.testing.assert_allclose(np.array(step.result.stress), np.array(rm.stress))

        strain_from_iter = np.stack([s.strain for s in steps])
        np.testing.assert_allclose(strain_from_iter, result_run.strain)

    def test_matches_run_multiaxial(self, model):
        integrator = PythonNumericalIntegrator(model)
        load = _multiaxial_strain_load(integrator, n_steps=20)
        result_run = StrainDriver(integrator).run(load)
        steps = list(StrainDriver(integrator).iter_run(load))

        stress_iter = np.stack([np.array(s.result.stress) for s in steps])
        np.testing.assert_allclose(stress_iter, result_run.stress, rtol=1e-12)

    def test_early_break_on_plastic(self, model):
        load = _strain_load(n_steps=30, max_strain=5e-3)
        driver = StrainDriver(PythonNumericalIntegrator(model))
        plastic_steps = []
        for step in driver.iter_run(load):
            if step.result.is_plastic:
                plastic_steps.append(step.i)
                break
        assert len(plastic_steps) == 1

    def test_cumulative_strain_uniaxial(self, model):
        n = 15
        max_strain = 3e-3
        strains = np.linspace(0, max_strain, n)
        load = FieldHistory(FieldType.STRAIN, "Strain", strains)
        driver = StrainDriver(PythonNumericalIntegrator(model))
        for step in driver.iter_run(load):
            np.testing.assert_allclose(step.strain[0], strains[step.i], rtol=1e-12)

    def test_state_accessible_from_step(self, model):
        load = _strain_load(n_steps=20, max_strain=5e-3)
        driver = StrainDriver(PythonNumericalIntegrator(model))
        for step in driver.iter_run(load):
            assert "ep" in step.result.state

    def test_rejects_stress_typed_load(self, model):
        """A STRESS-typed history must not be silently read as strain."""
        load = FieldHistory(FieldType.STRESS, "Stress", np.linspace(0, 1e-3, 5))
        driver = StrainDriver(PythonNumericalIntegrator(model))
        with pytest.raises(ValueError, match="StrainDriver expects"):
            list(driver.iter_run(load))

    def test_nonconverged_integrator_stops_and_reports(self, model):
        """A non-converged return mapping must not advance the state.

        With raise_on_nonconverged=False the integrator hands back a
        non-converged NR iterate; carrying it forward would silently corrupt
        every later step, so the driver must yield converged=False and stop.
        """
        # max_iter=0 guarantees the first plastic step cannot converge.
        integrator = PythonNumericalIntegrator(
            model, max_iter=0, raise_on_nonconverged=False
        )
        load = _strain_load(n_steps=10, max_strain=5e-3)
        steps = list(StrainDriver(integrator).iter_run(load))

        assert len(steps) < 10, "driver must stop at the non-converged step"
        assert steps[-1].converged is False
        assert all(s.converged for s in steps[:-1])


class TestStressDriverIterRun:
    def test_yields_driver_step_instances(self, model):
        integrator = PythonNumericalIntegrator(model)
        driver = StressDriver(integrator)
        for step in driver.iter_run(_stress_load_1d(integrator)):
            assert isinstance(step, DriverStep)

    def test_step_index_sequential(self, model):
        integrator = PythonNumericalIntegrator(model)
        driver = StressDriver(integrator)
        indices = [s.i for s in driver.iter_run(_stress_load_1d(integrator, n_steps=10))]
        assert indices == list(range(10))

    def test_n_outer_iter_positive(self, model):
        integrator = PythonNumericalIntegrator(model)
        driver = StressDriver(integrator)
        for step in driver.iter_run(_stress_load_1d(integrator)):
            assert step.n_outer_iter >= 1

    def test_residual_inf_below_tol(self, model):
        integrator = PythonNumericalIntegrator(model)
        driver = StressDriver(integrator, tol=1e-8)
        for step in driver.iter_run(_stress_load_1d(integrator)):
            assert step.residual_inf < driver.tol

    def test_matches_run(self, model):
        integrator = PythonNumericalIntegrator(model)
        load = _stress_load_1d(integrator, n_steps=20)
        result_run = StressDriver(integrator).run(load)
        steps = list(StressDriver(integrator).iter_run(load))

        assert len(steps) == len(result_run.step_results)
        for step, rm in zip(steps, result_run.step_results):
            np.testing.assert_allclose(np.array(step.result.stress), np.array(rm.stress))

        strain_from_iter = np.stack([s.strain for s in steps])
        np.testing.assert_allclose(strain_from_iter, result_run.strain)

    # Non-convergence is forced with max_iter=1: a plastic step needs at
    # least two outer iterations (elastic-stiffness predictor, then a
    # consistent-tangent correction).  Driving the stress absurdly high no
    # longer works — that relied on the absolute tolerance being unreachable
    # at large stress scales, which was a unit-dependence bug, now fixed.
    def test_nonconverged_raises_by_default(self, model):
        integrator = PythonNumericalIntegrator(model)
        sigma = np.zeros((5, model.ntens))
        sigma[:, 0] = np.linspace(300.0, 500.0, 5)  # past yield (250)
        load = FieldHistory(FieldType.STRESS, "Stress", sigma)
        driver = StressDriver(integrator, max_iter=1)
        with pytest.raises(RuntimeError, match="NR did not converge"):
            list(driver.iter_run(load))

    def test_nonconverged_opt_in_yields(self, model):
        integrator = PythonNumericalIntegrator(model, raise_on_nonconverged=False)
        sigma = np.zeros((5, model.ntens))
        sigma[:, 0] = np.linspace(300.0, 500.0, 5)
        load = FieldHistory(FieldType.STRESS, "Stress", sigma)
        driver = StressDriver(integrator, max_iter=1)
        steps = list(driver.iter_run(load))
        assert len(steps) == 1
        assert steps[0].converged is False
        assert steps[0].residual_inf > driver.tol

    def test_converges_in_pa_units(self):
        """The criterion must stay reachable regardless of stress units.

        With Pa-scale stresses (~1e8) a purely absolute tol=1e-8 sits below
        round-off of the NR update, so convergence relied on luck; the
        relative term keeps the threshold proportionate.
        """
        from returnmap.models import J2Isotropic3D

        model_pa = J2Isotropic3D(E=210e9, nu=0.3, sigma_y0=250e6, H=1e9)
        sigma = np.zeros((5, 6))
        sigma[:, 0] = np.linspace(100e6, 300e6, 5)  # crosses yield
        load = FieldHistory(FieldType.STRESS, "Stress", sigma)
        driver = StressDriver(PythonNumericalIntegrator(model_pa))
        result = driver.run(load)
        np.testing.assert_allclose(
            result.stress[:, 0], sigma[:, 0], rtol=1e-9
        )

    def test_rejects_strain_typed_load(self, model):
        """A STRAIN-typed history must not be silently read as stress."""
        load = FieldHistory(FieldType.STRAIN, "Strain",
                            np.zeros((5, model.ntens)))
        driver = StressDriver(PythonNumericalIntegrator(model))
        with pytest.raises(ValueError, match="StressDriver expects"):
            list(driver.iter_run(load))

    def test_nonconverged_inner_return_mapping_fails_outer(self, model):
        """A non-converged inner iterate must not satisfy the outer check.

        The outer residual computed from a non-converged rm.stress is
        meaningless and could even pass the tolerance; the driver must treat
        the step as failed instead.
        """
        # max_iter=0: every plastic return mapping comes back converged=False.
        integrator = PythonNumericalIntegrator(
            model, max_iter=0, raise_on_nonconverged=False
        )
        sigma = np.zeros((5, model.ntens))
        sigma[:, 0] = np.linspace(100.0, 2.0 * model.sigma_y0, 5)
        load = FieldHistory(FieldType.STRESS, "Stress", sigma)
        driver = StressDriver(integrator)

        steps = list(driver.iter_run(load))
        assert steps[-1].converged is False
        assert all(s.converged for s in steps[:-1])


# ===========================================================================
# TestStepResults: DriverResult.step_results API
# ===========================================================================

class TestStrainDriverStepResults:
    def test_step_results_length(self, model):
        load = _uniaxial_strain_load(n_steps=20)
        result = StrainDriver(PythonNumericalIntegrator(model)).run(load)
        assert len(result.step_results) == 20

    def test_step_results_are_return_mapping_results(self, model):
        load = _uniaxial_strain_load(n_steps=10)
        result = StrainDriver(PythonNumericalIntegrator(model)).run(load)
        for rm in result.step_results:
            assert isinstance(rm, StressUpdateResult)

    def test_step_results_stress_matches_fields(self, model):
        load = _uniaxial_strain_load(n_steps=15)
        result = StrainDriver(PythonNumericalIntegrator(model)).run(load)
        stress_from_steps = np.array([np.array(rm.stress) for rm in result.step_results])
        np.testing.assert_allclose(result.stress, stress_from_steps, rtol=1e-12)

    def test_plastic_steps_detected(self, model):
        load = _uniaxial_strain_load(n_steps=20, max_strain=5e-3)
        result = StrainDriver(PythonNumericalIntegrator(model)).run(load)
        is_plastic = [rm.is_plastic for rm in result.step_results]
        assert any(is_plastic), "expected at least some plastic steps"
        assert not is_plastic[0], "first step (near zero strain) should be elastic"

    def test_dlambda_positive_in_plastic_steps(self, model):
        load = _uniaxial_strain_load(n_steps=20, max_strain=5e-3)
        result = StrainDriver(PythonNumericalIntegrator(model)).run(load)
        for rm in result.step_results:
            if rm.is_plastic:
                assert float(rm.dlambda) > 0.0

    def test_fields_stress_property(self, model):
        load = _uniaxial_strain_load(n_steps=10)
        result = StrainDriver(PythonNumericalIntegrator(model)).run(load)
        assert result.stress.shape == (10, model.ntens)

    def test_history_needs_no_request(self, model):
        load = _uniaxial_strain_load(n_steps=10, max_strain=5e-3)
        result = StrainDriver(PythonNumericalIntegrator(model)).run(load)
        ep_history = result.history("ep")
        assert ep_history.shape == (10,)
        assert ep_history[-1] > 0.0

    def test_history_matches_step_results(self, model):
        load = _uniaxial_strain_load(n_steps=10, max_strain=5e-3)
        result = StrainDriver(PythonNumericalIntegrator(model)).run(load)
        ep_from_steps = np.array([float(rm.state["ep"]) for rm in result.step_results])
        np.testing.assert_allclose(result.history("ep"), ep_from_steps, rtol=1e-12)

    def test_history_of_stress_matches_stress_property(self, model):
        """history("stress") reads state["stress"]; must not lag on elastic steps."""
        load = _uniaxial_strain_load(n_steps=10, max_strain=5e-3)
        result = StrainDriver(PythonNumericalIntegrator(model)).run(load)
        assert not result.step_results[0].is_plastic, "expected elastic steps in this range"
        np.testing.assert_allclose(
            np.array(result.history("stress")), np.array(result.stress), rtol=1e-14
        )

    def test_step_n_minus_1_state_is_previous_step(self, model):
        load = _uniaxial_strain_load(n_steps=15, max_strain=5e-3)
        result = StrainDriver(PythonNumericalIntegrator(model)).run(load)
        for i in range(1, len(result.step_results)):
            prev_state = result.step_results[i - 1].state
            curr_state_n = result.step_results[i].state
            assert float(curr_state_n["ep"]) >= float(prev_state["ep"])


class TestStressDriverStepResults:
    def test_step_results_length(self, model):
        load = _uniaxial_stress_load(model, n_steps=15)
        result = StressDriver(PythonNumericalIntegrator(model)).run(load)
        assert len(result.step_results) == 15

    def test_step_results_are_return_mapping_results(self, model):
        load = _uniaxial_stress_load(model, n_steps=10)
        result = StressDriver(PythonNumericalIntegrator(model)).run(load)
        for rm in result.step_results:
            assert isinstance(rm, StressUpdateResult)

    def test_fields_stress_property(self, model):
        load = _uniaxial_stress_load(model, n_steps=10)
        result = StressDriver(PythonNumericalIntegrator(model)).run(load)
        assert result.stress.shape == (10, model.ntens)


# ===========================================================================
# Non-convergence: one switch (the integrator's), reported by run() as a value
# ===========================================================================

class TestRunReportsNonConvergence:
    """``run`` must not hand back a non-converged iterate as if it were a step."""

    def test_strain_driver_converged_run(self, model):
        result = StrainDriver(PythonNumericalIntegrator(model)).run(_strain_load())
        assert result.converged is True
        assert result.failed_step is None

    def test_strain_driver_excludes_the_failed_step(self, model):
        # max_iter=0: the first plastic step cannot converge.
        integrator = PythonNumericalIntegrator(
            model, max_iter=0, raise_on_nonconverged=False
        )
        load = _strain_load(n_steps=10, max_strain=5e-3)
        result = StrainDriver(integrator).run(load)

        assert result.converged is False
        assert 0 < result.failed_step < 10
        # Every row is a completed step: the failed one is left out.
        assert len(result.step_results) == result.failed_step
        assert result.stress.shape == (result.failed_step, model.ntens)
        assert result.strain.shape == (result.failed_step, model.ntens)
        assert all(r.converged for r in result.step_results)

    def test_strain_driver_raises_with_default_integrator(self, model):
        integrator = PythonNumericalIntegrator(model, max_iter=0)
        with pytest.raises(RuntimeError):
            StrainDriver(integrator).run(_strain_load(n_steps=10, max_strain=5e-3))

    def test_stress_driver_follows_the_integrator_switch(self, model):
        """The outer NR has no switch of its own — it reads the integrator's."""
        sigma = np.zeros((5, model.ntens))
        sigma[:, 0] = np.linspace(300.0, 500.0, 5)  # past yield (250)
        load = FieldHistory(FieldType.STRESS, "Stress", sigma)

        raising = StressDriver(PythonNumericalIntegrator(model), max_iter=1)
        with pytest.raises(RuntimeError, match="NR did not converge"):
            raising.run(load)

        reporting = StressDriver(
            PythonNumericalIntegrator(model, raise_on_nonconverged=False), max_iter=1
        )
        result = reporting.run(load)
        assert result.converged is False
        assert result.failed_step == 0
        assert result.step_results == []
        assert result.stress.shape == (0, model.ntens)
        assert result.strain.shape == (0, model.ntens)

    def test_mixed_driver_follows_the_integrator_switch(self, model_3d):
        eps = np.linspace(0, 3.0 * model_3d.sigma_y0 / model_3d.E, 3).reshape(-1, 1)
        load = strain_load(eps)
        integrator = PythonNumericalIntegrator(model_3d, raise_on_nonconverged=False)
        result = MixedDriver(integrator, prescribed_strain_idx=[0], max_iter=1).run(load)

        assert result.converged is False
        assert result.failed_step is not None
        assert len(result.step_results) == result.failed_step

    def test_history_with_no_completed_step(self, model):
        sigma = np.zeros((3, model.ntens))
        sigma[:, 0] = np.linspace(300.0, 500.0, 3)
        load = FieldHistory(FieldType.STRESS, "Stress", sigma)
        integrator = PythonNumericalIntegrator(model, raise_on_nonconverged=False)
        result = StressDriver(integrator, max_iter=1).run(load)
        assert result.failed_step == 0
        assert len(result.history("ep")) == 0


class TestIntegratorContract:
    def test_bare_model_is_rejected(self, model):
        with pytest.raises(TypeError, match="PythonIntegrator"):
            StrainDriver(model)

    def test_incomplete_integrator_names_what_is_missing(self, model):
        class _OnlyStressUpdate:
            dimension = model.dimension
            ntens = model.ntens

            def initial_state(self):
                return {}

            def stress_update(self, strain_inc, stress_n, state_n):
                raise AssertionError("not called")

        with pytest.raises(TypeError, match="raise_on_nonconverged, increment, elastic_tangent"):
            StressDriver(_OnlyStressUpdate())

    @pytest.mark.parametrize("cls", [
        PythonIntegrator, PythonNumericalIntegrator, PythonAnalyticalIntegrator,
    ])
    def test_python_integrators_fulfil_it(self, model, cls):
        from returnmap.simulation import StressIntegrator
        assert isinstance(cls(model), StressIntegrator)

    def test_python_elastic_tangent_is_the_model_stiffness(self, model):
        integrator = PythonNumericalIntegrator(model)
        state = model.initial_state()
        np.testing.assert_array_equal(
            np.asarray(integrator.elastic_tangent(np.zeros(model.ntens), state)),
            np.asarray(model.elastic_stiffness(state)),
        )


def test_driver_result_is_dataclass(model):
    load = _uniaxial_strain_load(n_steps=5)
    result = StrainDriver(PythonNumericalIntegrator(model)).run(load)
    assert isinstance(result, DriverResult)
    assert hasattr(result, "step_results")
    assert hasattr(result, "strain")


# ===========================================================================
# TestStressDriver: stress-controlled simulation
# ===========================================================================

class TestStressDriver:
    def test_elastic_strain_matches_compliance(self, model_3d):
        sigma_y0 = model_3d.sigma_y0
        N = 10
        stress_history = np.zeros((N, 6))
        stress_history[:, 0] = np.linspace(0.0, 0.5 * sigma_y0, N)

        result = StressDriver(PythonNumericalIntegrator(model_3d)).run(stress_load(stress_history))

        C = np.array(model_3d.elastic_stiffness(model_3d.initial_state()))
        S = np.linalg.inv(C)

        for i in range(N):
            expected_strain = S @ stress_history[i]
            np.testing.assert_allclose(
                result.strain[i], expected_strain, atol=1e-10,
                err_msg=f"strain mismatch at step {i}"
            )
        np.testing.assert_allclose(result.stress, stress_history, atol=1e-8)

    def test_uniaxial_stress_3d_lateral_strains_negative(self, model_3d):
        sigma_y0 = model_3d.sigma_y0
        N = 20
        stress_history = np.zeros((N, 6))
        stress_history[:, 0] = np.linspace(0.0, 1.5 * sigma_y0, N)

        result = StressDriver(PythonNumericalIntegrator(model_3d)).run(stress_load(stress_history))

        assert np.all(result.strain[1:, 1] <= 0.0), "ε22 should be non-positive"
        assert np.all(result.strain[1:, 2] <= 0.0), "ε33 should be non-positive"

    def test_uniaxial_stress_3d_off_diagonal_stress_near_zero(self, model_3d):
        sigma_y0 = model_3d.sigma_y0
        N = 20
        stress_history = np.zeros((N, 6))
        stress_history[:, 0] = np.linspace(0.0, 1.5 * sigma_y0, N)

        result = StressDriver(PythonNumericalIntegrator(model_3d)).run(stress_load(stress_history))

        np.testing.assert_allclose(
            result.stress[:, 1:], 0.0, atol=1e-6,
            err_msg="Off-axis stresses should be ~0 under uniaxial stress control"
        )

    def test_uniaxial_stress_3d_sigma11_matches_target(self, model_3d):
        sigma_y0 = model_3d.sigma_y0
        N = 20
        stress_history = np.zeros((N, 6))
        stress_history[:, 0] = np.linspace(0.0, 1.5 * sigma_y0, N)

        result = StressDriver(PythonNumericalIntegrator(model_3d)).run(stress_load(stress_history))

        np.testing.assert_allclose(
            result.stress[:, 0], stress_history[:, 0], atol=1e-8,
            err_msg="σ11 output must match prescribed target"
        )

    def test_output_shapes(self, model_3d):
        N = 15
        stress_history = np.zeros((N, 6))
        stress_history[:, 0] = np.linspace(0, 200.0, N)

        result = StressDriver(PythonNumericalIntegrator(model_3d)).run(stress_load(stress_history))

        assert result.stress.shape == (N, 6)
        assert result.strain.shape == (N, 6)

    def test_history_of_an_unknown_state_names_the_available_ones(self, model_3d):
        stress_history = np.zeros((5, 6))
        stress_history[:, 0] = np.linspace(0, 200.0, 5)

        result = StressDriver(PythonNumericalIntegrator(model_3d)).run(stress_load(stress_history))

        with pytest.raises(KeyError, match=r"'alpha' is not a state variable.*'ep'"):
            result.history("alpha")

    def test_stress_driver_state_history(self, model_3d):
        sigma_y0 = model_3d.sigma_y0
        N = 20
        stress_history = np.zeros((N, 6))
        stress_history[:, 0] = np.linspace(0.0, 1.5 * sigma_y0, N)

        result = StressDriver(PythonNumericalIntegrator(model_3d)).run(stress_load(stress_history))

        ep = result.history("ep")
        assert ep.shape == (N,)
        assert np.all(ep >= 0.0)
        assert ep[-1] > 0.0, "ep should be positive after plastic loading"

    def test_strain_driver_state_history(self, model_3d):
        sigma_y0 = model_3d.sigma_y0
        E = model_3d.E
        N = 20
        strain_history = np.zeros((N, 6))
        strain_history[:, 0] = np.linspace(0.0, 3 * sigma_y0 / E, N)

        result = StrainDriver(PythonNumericalIntegrator(model_3d)).run(
            FieldHistory(FieldType.STRAIN, "Strain", strain_history),
        )

        assert result.history("ep").shape == (N,)
        assert result.history("ep")[-1] > 0.0

    def test_consistency_with_strain_driver_1d(self, model_1d):
        sigma_y0 = model_1d.sigma_y0
        E = model_1d.E

        eps_yield = sigma_y0 / E
        N = 25
        eps_history_1d = np.linspace(0.0, 3 * eps_yield, N)

        result_strain = StrainDriver(PythonNumericalIntegrator(model_1d)).run(
            strain_load(eps_history_1d)
        )
        sigma_history_2d = result_strain.stress

        result_stress = StressDriver(PythonNumericalIntegrator(model_1d)).run(
            stress_load(sigma_history_2d)
        )
        recovered_strain = result_stress.strain[:, 0]

        np.testing.assert_allclose(
            recovered_strain, eps_history_1d, atol=1e-8,
            err_msg="StressDriver did not recover original strain from StrainDriver output"
        )

    def test_convergence_failure_raises(self, model_3d):
        sigma_y0 = model_3d.sigma_y0
        stress_history = np.zeros((1, 6))
        stress_history[0, 0] = 1.5 * sigma_y0

        driver = StressDriver(PythonNumericalIntegrator(model_3d), max_iter=1)
        with pytest.raises(RuntimeError, match="NR did not converge"):
            driver.run(stress_load(stress_history))

    def test_strain_driver_general_shape(self, model):
        N = 20
        strain6 = np.zeros((N, 6))
        strain6[:, 0] = np.linspace(0.0, 5e-3, N)
        load = FieldHistory(FieldType.STRAIN, "Strain", strain6)
        result = StrainDriver(PythonNumericalIntegrator(model)).run(load)
        assert result.stress.shape == (N, 6)
        assert result.stress[-1, 0] > model.sigma_y0

    @pytest.mark.parametrize("IntegratorCls", [PythonNumericalIntegrator, PythonAnalyticalIntegrator])
    def test_stress_driver_integrator_types(self, model_3d, IntegratorCls):
        sigma_y0 = model_3d.sigma_y0
        N = 10
        stress_history = np.zeros((N, 6))
        stress_history[:, 0] = np.linspace(0.0, 1.5 * sigma_y0, N)

        result = StressDriver(IntegratorCls(model_3d)).run(stress_load(stress_history))

        assert len(result.step_results) == N
        for sr in result.step_results:
            assert isinstance(sr, StressUpdateResult)


# ===========================================================================
# TestMixedDriver: mixed strain/stress boundary conditions
# ===========================================================================

class TestMixedDriver:
    def test_validation_duplicate_strain_idx(self, integ_3d):
        with pytest.raises(ValueError, match="duplicate"):
            MixedDriver(integ_3d, prescribed_strain_idx=[0, 0])

    def test_validation_out_of_range(self, integ_3d):
        with pytest.raises(ValueError, match="out of range"):
            MixedDriver(integ_3d, prescribed_strain_idx=[7])

    def test_validation_overlap(self, integ_3d):
        with pytest.raises(ValueError, match="disjoint"):
            MixedDriver(integ_3d, prescribed_strain_idx=[0, 1],
                        prescribed_stress_idx=[1, 2, 3, 4, 5])

    def test_validation_incomplete_union(self, integ_3d):
        with pytest.raises(ValueError, match="cover all"):
            MixedDriver(integ_3d, prescribed_strain_idx=[0],
                        prescribed_stress_idx=[1, 2, 3])

    def test_validation_empty_strain_idx(self, integ_3d):
        with pytest.raises(ValueError, match="must not be empty"):
            MixedDriver(integ_3d, prescribed_strain_idx=[])

    def test_validation_load_type_must_be_strain(self, integ_3d):
        driver = MixedDriver(integ_3d, prescribed_strain_idx=[0])
        N = 5
        bad_load = stress_load(np.zeros((N, 6)))
        with pytest.raises(ValueError, match="FieldType.STRAIN"):
            list(driver.iter_run(bad_load))

    def test_validation_load_shape_must_match_nP(self, integ_3d):
        driver = MixedDriver(integ_3d, prescribed_strain_idx=[0])
        N = 5
        bad_load = strain_load(np.zeros((N, 6)))
        with pytest.raises(ValueError, match=r"\(N, 1\)"):
            list(driver.iter_run(bad_load))

    def test_validation_stress_history_shape_mismatch(self, integ_3d):
        driver = MixedDriver(integ_3d, prescribed_strain_idx=[0])
        N = 5
        load = strain_load(np.zeros((N, 1)))
        bad_sig = np.zeros((N, 3))
        with pytest.raises(ValueError, match=r"\(5, 5\)"):
            list(driver.iter_run(load, prescribed_stress_history=bad_sig))

    def test_complement_inferred_from_strain_idx(self, integ_3d):
        driver = MixedDriver(integ_3d, prescribed_strain_idx=[0])
        np.testing.assert_array_equal(driver._P, [0])
        np.testing.assert_array_equal(driver._F, [1, 2, 3, 4, 5])

    def test_explicit_stress_idx_matches_complement(self, integ_3d):
        driver = MixedDriver(integ_3d, prescribed_strain_idx=[0],
                             prescribed_stress_idx=[1, 2, 3, 4, 5])
        np.testing.assert_array_equal(driver._P, [0])
        np.testing.assert_array_equal(driver._F, [1, 2, 3, 4, 5])

    def test_validation_stress_history_provided_when_nF_zero(self, integ_3d):
        driver = MixedDriver(integ_3d, prescribed_strain_idx=[0, 1, 2, 3, 4, 5])
        N = 3
        load = strain_load(np.zeros((N, 6)))
        with pytest.raises(ValueError, match="no stress-prescribed"):
            list(driver.iter_run(load, prescribed_stress_history=np.zeros((N, 1))))

    def test_output_shapes(self, integ_3d):
        N = 10
        driver = MixedDriver(integ_3d, prescribed_strain_idx=[0])
        load = strain_load(np.linspace(0, 0.005, N).reshape(-1, 1))
        result = driver.run(load)
        assert result.stress.shape == (N, 6)
        assert result.strain.shape == (N, 6)

    def test_uniaxial_elastic_stress_11(self, model_3d, integ_3d):
        E, nu = model_3d.E, model_3d.nu
        sigma_y0 = model_3d.sigma_y0
        N = 20
        eps_max = 0.5 * sigma_y0 / E

        load = strain_load(np.linspace(0, eps_max, N).reshape(-1, 1))
        driver = MixedDriver(integ_3d, prescribed_strain_idx=[0])
        result = driver.run(load)

        eps_hist = np.linspace(0, eps_max, N)
        np.testing.assert_allclose(result.stress[:, 0], E * eps_hist, rtol=1e-5,
                                   err_msg="σ11 must equal E * ε11 in elastic regime")

    def test_uniaxial_elastic_lateral_stress_zero(self, model_3d, integ_3d):
        E, sigma_y0 = model_3d.E, model_3d.sigma_y0
        N = 20
        eps_max = 0.5 * sigma_y0 / E

        load = strain_load(np.linspace(0, eps_max, N).reshape(-1, 1))
        result = MixedDriver(integ_3d, prescribed_strain_idx=[0]).run(load)

        np.testing.assert_allclose(result.stress[:, 1:], 0.0, atol=1e-6)

    def test_uniaxial_elastic_lateral_strains(self, model_3d, integ_3d):
        E, nu, sigma_y0 = model_3d.E, model_3d.nu, model_3d.sigma_y0
        N = 20
        eps_max = 0.5 * sigma_y0 / E
        eps_hist = np.linspace(0, eps_max, N)

        load = strain_load(eps_hist.reshape(-1, 1))
        result = MixedDriver(integ_3d, prescribed_strain_idx=[0]).run(load)

        np.testing.assert_allclose(result.strain[:, 1], -nu * eps_hist, rtol=1e-5)
        np.testing.assert_allclose(result.strain[:, 2], -nu * eps_hist, rtol=1e-5)
        np.testing.assert_allclose(result.strain[:, 3:], 0.0, atol=1e-10)

    def test_uniaxial_plastic_matches_stressdriver(self, model_3d, integ_3d):
        sigma_y0 = model_3d.sigma_y0
        E = model_3d.E
        N = 30
        eps_max = 3.0 * sigma_y0 / E

        load_mixed = strain_load(np.linspace(0, eps_max, N).reshape(-1, 1))
        res_mixed = MixedDriver(integ_3d, prescribed_strain_idx=[0]).run(load_mixed)

        sig_history = np.zeros((N, 6))
        sig_history[:, 0] = res_mixed.stress[:, 0]
        res_stress = StressDriver(integ_3d).run(stress_load(sig_history))

        np.testing.assert_allclose(res_mixed.stress, res_stress.stress, atol=1e-5)
        np.testing.assert_allclose(res_mixed.strain, res_stress.strain, atol=1e-5)

    def test_consistency_with_1d_strain_driver(self, model_3d, integ_3d, model_1d, integ_1d):
        sigma_y0 = model_3d.sigma_y0
        E = model_3d.E
        N = 30
        eps_max = 3.0 * sigma_y0 / E

        eps_hist = np.linspace(0, eps_max, N)
        load_mixed = strain_load(eps_hist.reshape(-1, 1))
        res_3d = MixedDriver(integ_3d, prescribed_strain_idx=[0]).run(load_mixed)

        res_1d = StrainDriver(integ_1d).run(FieldHistory(FieldType.STRAIN, "Strain", eps_hist))

        np.testing.assert_allclose(res_3d.stress[:, 0], res_1d.stress[:, 0], rtol=1e-4)
        np.testing.assert_allclose(res_3d.history("ep"), res_1d.history("ep"), rtol=1e-4)

    def test_biaxial_constant_lateral_stress(self, model_3d, integ_3d):
        sigma_y0 = model_3d.sigma_y0
        E = model_3d.E
        N = 30
        eps_max = 2.0 * sigma_y0 / E
        sigma22_target = -100.0

        load = strain_load(np.linspace(0, eps_max, N).reshape(-1, 1))
        sig_F = np.zeros((N, 5))
        sig_F[:, 0] = sigma22_target

        driver = MixedDriver(integ_3d, prescribed_strain_idx=[0])
        result = driver.run(load, prescribed_stress_history=sig_F)

        np.testing.assert_allclose(result.stress[:, 1], sigma22_target, atol=1e-5)
        np.testing.assert_allclose(result.stress[:, 2:], 0.0, atol=1e-5)

    def test_full_strain_control_matches_strain_driver(self, model_3d, integ_3d):
        sigma_y0 = model_3d.sigma_y0
        E = model_3d.E
        N = 20

        strain6 = np.zeros((N, 6))
        strain6[:, 0] = np.linspace(0, 3.0 * sigma_y0 / E, N)

        load_mixed = strain_load(strain6)
        load_strain = FieldHistory(FieldType.STRAIN, "Strain", strain6)

        res_mixed = MixedDriver(integ_3d, prescribed_strain_idx=[0, 1, 2, 3, 4, 5]).run(load_mixed)
        res_strain = StrainDriver(integ_3d).run(load_strain)

        np.testing.assert_allclose(res_mixed.stress, res_strain.stress, atol=1e-10)
        np.testing.assert_allclose(res_mixed.strain, res_strain.strain, atol=1e-10)

    def test_max_iter_one_raises(self, model_3d, integ_3d):
        sigma_y0 = model_3d.sigma_y0
        E = model_3d.E
        load = strain_load(np.array([[3.0 * sigma_y0 / E]]))
        driver = MixedDriver(integ_3d, prescribed_strain_idx=[0], max_iter=1)
        with pytest.raises(RuntimeError, match="NR did not converge"):
            driver.run(load)

    def test_nonraising_integrator_yields_and_stops(self, model_3d):
        sigma_y0 = model_3d.sigma_y0
        E = model_3d.E
        N = 3
        eps_hist = np.linspace(0, 3.0 * sigma_y0 / E, N).reshape(-1, 1)
        load = strain_load(eps_hist)

        integrator = PythonNumericalIntegrator(model_3d, raise_on_nonconverged=False)
        driver = MixedDriver(integrator, prescribed_strain_idx=[0], max_iter=1)
        steps = list(driver.iter_run(load))

        assert len(steps) >= 1
        non_converged = [s for s in steps if not s.converged]
        assert len(non_converged) == 1
        assert non_converged[0].i < N
        assert steps[-1] is non_converged[0], "driver must stop at the failed step"

    def test_initial_stress_zero_increment_preserves_prestress(self, model_3d, integ_3d):
        prestress = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        load = strain_load(np.zeros((1, 1)))
        driver = MixedDriver(integ_3d, prescribed_strain_idx=[0])
        steps = list(driver.iter_run(load, initial_stress=prestress))
        np.testing.assert_allclose(
            np.array(steps[0].result.stress_trial), prestress, atol=1e-12,
        )

    def test_initial_state_shifts_yield_surface(self, model_3d, integ_3d):
        sigma_y0, E, H = model_3d.sigma_y0, model_3d.E, model_3d.H
        eps_y = sigma_y0 / E
        load = strain_load(np.array([[eps_y * 1.5]]))

        driver = MixedDriver(integ_3d, prescribed_strain_idx=[0])
        step_fresh = list(driver.iter_run(load))[0]
        assert step_fresh.result.is_plastic, "Should be plastic without prior hardening"

        large_ep = anp.array(0.2)
        step_hardened = list(driver.iter_run(load, initial_state={"ep": large_ep}))[0]
        assert not step_hardened.result.is_plastic, "Should be elastic with large initial ep"

    def test_initial_stress_state_run_forwards(self, model_3d, integ_3d):
        sigma_y0 = model_3d.sigma_y0
        prestress = np.array([sigma_y0 * 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])

        load = strain_load(np.zeros((1, 1)))
        result = MixedDriver(integ_3d, prescribed_strain_idx=[0]).run(
            load, initial_stress=prestress
        )
        assert result.stress.shape == (1, 6)
        np.testing.assert_allclose(
            np.array(result.step_results[0].stress_trial), prestress, atol=1e-12,
        )

    def test_mixed_driver_state_history(self, model_3d, integ_3d):
        sigma_y0 = model_3d.sigma_y0
        E = model_3d.E
        N = 20
        load = strain_load(np.linspace(0, 3.0 * sigma_y0 / E, N).reshape(-1, 1))
        result = MixedDriver(integ_3d, prescribed_strain_idx=[0]).run(load)
        ep = result.history("ep")
        assert ep.shape == (N,)
        assert ep[-1] > 0.0

    def test_prescribed_strain_equals_input(self, model_3d, integ_3d):
        sigma_y0 = model_3d.sigma_y0
        E = model_3d.E
        N = 30
        eps_hist = np.linspace(0, 3.0 * sigma_y0 / E, N)

        load = strain_load(eps_hist.reshape(-1, 1))
        result = MixedDriver(integ_3d, prescribed_strain_idx=[0]).run(load)

        np.testing.assert_allclose(result.strain[:, 0], eps_hist, atol=1e-12)


# ===========================================================================
# TestInitialState: initial_stress / initial_state kwargs
# ===========================================================================

class TestStrainDriverInitialState:
    def test_default_equals_zero_initial(self, model):
        integrator = PythonIntegrator(model)
        load = _strain_load()
        load_explicit = FieldHistory(FieldType.STRAIN, "Strain",
                                     np.array([[3e-3, 0, 0, 0, 0, 0]]))
        d = StrainDriver(integrator)
        steps_default = list(d.iter_run(load_explicit))
        steps_explicit = list(d.iter_run(load_explicit, initial_stress=np.zeros(6),
                                         initial_state=model.initial_state()))
        np.testing.assert_allclose(steps_default[0].result.stress,
                                   steps_explicit[0].result.stress, atol=1e-14)

    def test_initial_stress_shifts_trial(self, model):
        integrator = PythonIntegrator(model)
        prestress = np.array([100.0, 0, 0, 0, 0, 0])
        load = FieldHistory(FieldType.STRAIN, "Strain", np.array([[0, 0, 0, 0, 0, 0]]))
        d = StrainDriver(integrator)
        steps = list(d.iter_run(load, initial_stress=prestress))
        np.testing.assert_allclose(steps[0].result.stress_trial, prestress, atol=1e-12)

    def test_initial_state_shifts_yield(self, model):
        integrator = PythonIntegrator(model)
        large_ep = anp.array(0.1)
        initial_state = {"ep": large_ep}
        eps_y = model.sigma_y0 / model.E
        load = FieldHistory(FieldType.STRAIN, "Strain", np.array([[eps_y * 1.5, 0, 0, 0, 0, 0]]))
        d = StrainDriver(integrator)
        step_fresh = list(d.iter_run(load))[0]
        step_hardened = list(d.iter_run(load, initial_state=initial_state))[0]
        assert step_fresh.result.is_plastic
        assert not step_hardened.result.is_plastic

    def test_run_forwards_kwargs(self, model):
        integrator = PythonIntegrator(model)
        prestress = np.array([50.0, 0, 0, 0, 0, 0])
        load = FieldHistory(FieldType.STRAIN, "Strain", np.array([[0, 0, 0, 0, 0, 0]]))
        d = StrainDriver(integrator)
        result = d.run(load, initial_stress=prestress)
        np.testing.assert_allclose(result.stress[0], prestress, atol=1e-12)


class TestStressDriverInitialState:
    def test_default_equals_zero_initial(self, model):
        integrator = PythonIntegrator(model)
        load = FieldHistory(FieldType.STRESS, "Stress", np.array([[100.0, 0, 0, 0, 0, 0]]))
        d = StressDriver(integrator)
        steps_default = list(d.iter_run(load))
        steps_explicit = list(d.iter_run(load, initial_stress=np.zeros(6),
                                         initial_state=model.initial_state()))
        np.testing.assert_allclose(steps_default[0].result.stress,
                                   steps_explicit[0].result.stress, atol=1e-12)

    def test_initial_stress_changes_starting_point(self, model):
        integrator = PythonIntegrator(model)
        target = np.array([200.0, 0, 0, 0, 0, 0])
        load = FieldHistory(FieldType.STRESS, "Stress", np.array([target]))
        d = StressDriver(integrator)
        step_zero = list(d.iter_run(load))[0]
        step_pre = list(d.iter_run(load, initial_stress=target * 0.5))[0]
        assert abs(step_pre.strain[0]) < abs(step_zero.strain[0])

    def test_run_forwards_kwargs(self, model):
        integrator = PythonIntegrator(model)
        load = FieldHistory(FieldType.STRESS, "Stress", np.array([[100.0, 0, 0, 0, 0, 0]]))
        d = StressDriver(integrator)
        result = d.run(load, initial_stress=np.zeros(6))
        assert len(result.step_results) == 1


# ---------------------------------------------------------------------------
# Load shape: checked, not broadcast
# ---------------------------------------------------------------------------

class TestLoadShape:
    """numpy would broadcast a history of the wrong width into every
    component — shear included — and the run would proceed."""

    @pytest.fixture
    def integ(self):
        return PythonNumericalIntegrator(
            J2Isotropic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0))

    @pytest.mark.parametrize("data", [
        np.linspace(50.0, 200.0, 4),                 # (N,)
        np.linspace(50.0, 200.0, 4).reshape(-1, 1),  # (N, 1)
        np.zeros((4, 3)),                            # another dimension's width
    ], ids=["1d", "one-column", "three-columns"])
    def test_stress_driver_rejects_a_history_of_the_wrong_width(self, integ, data):
        load = FieldHistory(FieldType.STRESS, "Stress", data)
        with pytest.raises(ValueError, match=r"StressDriver expects load.data of shape \(N, 6\)"):
            StressDriver(integ).run(load)

    def test_default_cyclic_stress_history_is_rejected_for_a_3d_model(self, integ):
        """cyclic_stress defaults to ntens=1; that used to run as σ = [200]*6."""
        with pytest.raises(ValueError, match="ntens=6"):
            StressDriver(integ).run(FieldHistory.cyclic_stress([200.0]))

    def test_stress_driver_takes_a_1d_history_for_a_1d_model(self):
        m = J2Isotropic1D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)
        load = FieldHistory(FieldType.STRESS, "Stress", np.linspace(50.0, 300.0, 5))
        result = StressDriver(PythonNumericalIntegrator(m)).run(load)
        np.testing.assert_allclose(result.stress[:, 0], np.linspace(50.0, 300.0, 5), rtol=1e-8)

    @pytest.mark.parametrize("data", [
        np.linspace(0.0, 5e-3, 4).reshape(-1, 1),
        np.zeros((4, 3)),
    ], ids=["one-column", "three-columns"])
    def test_strain_driver_rejects_a_history_of_the_wrong_width(self, integ, data):
        load = FieldHistory(FieldType.STRAIN, "Strain", data)
        with pytest.raises(ValueError, match=r"StrainDriver expects load.data of shape \(N, 6\)"):
            StrainDriver(integ).run(load)

    def test_strain_driver_keeps_the_1d_shorthand(self, integ):
        """A 1-D strain history is ε11 only, the other strains held at zero."""
        load = FieldHistory(FieldType.STRAIN, "Strain", np.linspace(0.0, 5e-3, 6))
        result = StrainDriver(integ).run(load)
        np.testing.assert_array_equal(result.strain[:, 1:], 0.0)
