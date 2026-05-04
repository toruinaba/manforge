"""Tests for initial_stress / initial_state kwargs on StrainDriver and StressDriver."""

import numpy as np
import pytest

from manforge.simulation.driver import StrainDriver, StressDriver
from manforge.simulation.integrator import PythonIntegrator
from manforge.simulation.types import FieldHistory, FieldType


def _strain_load(values):
    return FieldHistory(type=FieldType.STRAIN, name="Strain", data=np.array(values))


def _stress_load(values):
    return FieldHistory(type=FieldType.STRESS, name="Stress", data=np.array(values))


class TestStrainDriverInitialState:
    def test_default_equals_zero_initial(self, model):
        """Default call and explicit zeros produce identical results."""
        integrator = PythonIntegrator(model)
        load = _strain_load([[3e-3, 0, 0, 0, 0, 0]])
        d = StrainDriver(integrator)
        steps_default = list(d.iter_run(load))
        steps_explicit = list(d.iter_run(load, initial_stress=np.zeros(6),
                                         initial_state=model.initial_state()))
        np.testing.assert_allclose(steps_default[0].result.stress,
                                   steps_explicit[0].result.stress, atol=1e-14)

    def test_initial_stress_shifts_trial(self, model):
        """Prestressed start shifts the elastic trial stress."""
        integrator = PythonIntegrator(model)
        prestress = np.array([100.0, 0, 0, 0, 0, 0])
        load = _strain_load([[0, 0, 0, 0, 0, 0]])  # zero increment
        d = StrainDriver(integrator)
        steps = list(d.iter_run(load, initial_stress=prestress))
        # Zero increment from prestressed state → trial stress == prestress
        np.testing.assert_allclose(steps[0].result.stress_trial, prestress, atol=1e-12)

    def test_initial_state_shifts_yield(self, model):
        """Non-zero initial ep shifts the yield surface."""
        integrator = PythonIntegrator(model)
        import autograd.numpy as anp
        # Large initial ep so isotropic hardening raises yield stress significantly
        large_ep = anp.array(0.1)
        initial_state = {"ep": large_ep}
        # Moderately plastic strain increment — plastic without hardening, elastic with
        eps_y = model.sigma_y0 / model.E
        load = _strain_load([[eps_y * 1.5, 0, 0, 0, 0, 0]])
        d = StrainDriver(integrator)
        step_fresh = list(d.iter_run(load))[0]
        step_hardened = list(d.iter_run(load, initial_state=initial_state))[0]
        # With extra hardening the step should remain elastic
        assert step_fresh.result.is_plastic
        assert not step_hardened.result.is_plastic

    def test_run_forwards_kwargs(self, model):
        """StrainDriver.run passes initial_stress/initial_state to iter_run."""
        integrator = PythonIntegrator(model)
        prestress = np.array([50.0, 0, 0, 0, 0, 0])
        load = _strain_load([[0, 0, 0, 0, 0, 0]])
        d = StrainDriver(integrator)
        result = d.run(load, initial_stress=prestress)
        np.testing.assert_allclose(result.stress[0], prestress, atol=1e-12)


class TestStressDriverInitialState:
    def test_default_equals_zero_initial(self, model):
        """Default call and explicit zeros produce identical results."""
        integrator = PythonIntegrator(model)
        load = _stress_load([[100.0, 0, 0, 0, 0, 0]])
        d = StressDriver(integrator)
        steps_default = list(d.iter_run(load))
        steps_explicit = list(d.iter_run(load, initial_stress=np.zeros(6),
                                         initial_state=model.initial_state()))
        np.testing.assert_allclose(steps_default[0].result.stress,
                                   steps_explicit[0].result.stress, atol=1e-12)

    def test_initial_stress_changes_starting_point(self, model):
        """Stress driver starting from prestress reaches different final strain."""
        integrator = PythonIntegrator(model)
        target = np.array([200.0, 0, 0, 0, 0, 0])
        load = _stress_load([target])
        d = StressDriver(integrator)
        step_zero = list(d.iter_run(load))[0]
        # Start halfway there — should require much less strain increment
        step_pre = list(d.iter_run(load, initial_stress=target * 0.5))[0]
        assert abs(step_pre.strain[0]) < abs(step_zero.strain[0])

    def test_run_forwards_kwargs(self, model):
        """StressDriver.run passes initial_stress/initial_state to iter_run."""
        integrator = PythonIntegrator(model)
        load = _stress_load([[100.0, 0, 0, 0, 0, 0]])
        d = StressDriver(integrator)
        # Should not raise
        result = d.run(load, initial_stress=np.zeros(6))
        assert len(result.step_results) == 1
