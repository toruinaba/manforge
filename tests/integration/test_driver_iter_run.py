"""Tests for DriverBase.iter_run and DriverStep."""

import numpy as np
import pytest

from manforge.simulation.driver import StrainDriver, StressDriver
from manforge.simulation.integrator import PythonNumericalIntegrator
from manforge.simulation.types import DriverStep, FieldHistory, FieldType


def _strain_load(n_steps=20, max_strain=3e-3):
    strains = np.linspace(0, max_strain, n_steps)
    return FieldHistory(FieldType.STRAIN, "Strain", strains)


def _multiaxial_strain_load(integrator, n_steps=20, max_strain=3e-3):
    data = np.zeros((n_steps, integrator.ntens))
    data[:, 0] = np.linspace(0, max_strain, n_steps)
    return FieldHistory(FieldType.STRAIN, "Strain", data)


def _stress_load(integrator, n_steps=20, max_stress=300.0):
    data = np.zeros((n_steps, integrator.ntens))
    data[:, 0] = np.linspace(0, max_stress, n_steps)
    return FieldHistory(FieldType.STRESS, "Stress", data)


# ---------------------------------------------------------------------------
# StrainDriver
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# StressDriver
# ---------------------------------------------------------------------------

class TestStressDriverIterRun:
    def test_yields_driver_step_instances(self, model):
        integrator = PythonNumericalIntegrator(model)
        driver = StressDriver(integrator)
        for step in driver.iter_run(_stress_load(integrator)):
            assert isinstance(step, DriverStep)

    def test_step_index_sequential(self, model):
        integrator = PythonNumericalIntegrator(model)
        driver = StressDriver(integrator)
        indices = [s.i for s in driver.iter_run(_stress_load(integrator, n_steps=10))]
        assert indices == list(range(10))

    def test_n_outer_iter_positive(self, model):
        integrator = PythonNumericalIntegrator(model)
        driver = StressDriver(integrator)
        for step in driver.iter_run(_stress_load(integrator)):
            assert step.n_outer_iter >= 1

    def test_residual_inf_below_tol(self, model):
        integrator = PythonNumericalIntegrator(model)
        driver = StressDriver(integrator, tol=1e-8)
        for step in driver.iter_run(_stress_load(integrator)):
            assert step.residual_inf < driver.tol

    def test_matches_run(self, model):
        integrator = PythonNumericalIntegrator(model)
        load = _stress_load(integrator, n_steps=20)
        result_run = StressDriver(integrator).run(load)
        steps = list(StressDriver(integrator).iter_run(load))

        assert len(steps) == len(result_run.step_results)
        for step, rm in zip(steps, result_run.step_results):
            np.testing.assert_allclose(np.array(step.result.stress), np.array(rm.stress))

        strain_from_iter = np.stack([s.strain for s in steps])
        np.testing.assert_allclose(strain_from_iter, result_run.strain)

    def test_nonconverged_raises_by_default(self, model):
        integrator = PythonNumericalIntegrator(model)
        huge_stress = np.zeros((5, model.ntens))
        huge_stress[:, 0] = np.linspace(1e8, 5e8, 5)
        load = FieldHistory(FieldType.STRESS, "Stress", huge_stress)
        driver = StressDriver(integrator, max_iter=3)
        with pytest.raises(RuntimeError, match="NR did not converge"):
            list(driver.iter_run(load))

    def test_nonconverged_opt_in_yields(self, model):
        # raise_on_nonconverged=False on the integrator lets the constitutive NR
        # return a best-effort result so StressDriver's outer NR can check convergence.
        integrator = PythonNumericalIntegrator(model, raise_on_nonconverged=False)
        huge_stress = np.zeros((5, model.ntens))
        huge_stress[:, 0] = np.linspace(1e8, 5e8, 5)
        load = FieldHistory(FieldType.STRESS, "Stress", huge_stress)
        driver = StressDriver(integrator, max_iter=3)
        steps = []
        for step in driver.iter_run(load, raise_on_nonconverged=False):
            steps.append(step)
        assert len(steps) == 1
        assert steps[0].converged is False
        assert steps[0].residual_inf > driver.tol
