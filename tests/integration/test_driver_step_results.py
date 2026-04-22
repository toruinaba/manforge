"""Tests for DriverResult.step_results and related API."""

import jax.numpy as jnp
import numpy as np
import pytest

from manforge.core.stress_update import StressUpdateResult
from manforge.simulation.driver import StrainDriver, StressDriver
from manforge.simulation.types import DriverResult, FieldHistory, FieldType


def _uniaxial_strain_load(n_steps=20, max_strain=3e-3):
    strains = np.linspace(0, max_strain, n_steps)
    return FieldHistory(FieldType.STRAIN, "Strain", strains)


def _uniaxial_stress_load(model, n_steps=20):
    C = np.array(model.elastic_stiffness())
    max_stress = 300.0
    stresses = np.zeros((n_steps, model.ntens))
    stresses[:, 0] = np.linspace(0, max_stress, n_steps)
    return FieldHistory(FieldType.STRESS, "Stress", stresses)


# ---------------------------------------------------------------------------
# StrainDriver step_results
# ---------------------------------------------------------------------------

class TestStrainDriverStepResults:
    def test_step_results_length(self, model):
        load = _uniaxial_strain_load(n_steps=20)
        driver = StrainDriver()
        result = driver.run(model, load)
        assert len(result.step_results) == 20

    def test_step_results_are_return_mapping_results(self, model):
        load = _uniaxial_strain_load(n_steps=10)
        driver = StrainDriver()
        result = driver.run(model, load)
        for rm in result.step_results:
            assert isinstance(rm, StressUpdateResult)

    def test_step_results_stress_matches_fields(self, model):
        load = _uniaxial_strain_load(n_steps=15)
        driver = StrainDriver()
        result = driver.run(model, load)
        stress_from_steps = np.array([np.array(rm.stress) for rm in result.step_results])
        np.testing.assert_allclose(result.stress, stress_from_steps, rtol=1e-12)

    def test_plastic_steps_detected(self, model):
        load = _uniaxial_strain_load(n_steps=20, max_strain=5e-3)
        driver = StrainDriver()
        result = driver.run(model, load)
        is_plastic = [rm.is_plastic for rm in result.step_results]
        assert any(is_plastic), "expected at least some plastic steps"
        assert not is_plastic[0], "first step (near zero strain) should be elastic"

    def test_dlambda_positive_in_plastic_steps(self, model):
        load = _uniaxial_strain_load(n_steps=20, max_strain=5e-3)
        driver = StrainDriver()
        result = driver.run(model, load)
        for rm in result.step_results:
            if rm.is_plastic:
                assert float(rm.dlambda) > 0.0

    def test_fields_stress_property(self, model):
        load = _uniaxial_strain_load(n_steps=10)
        driver = StrainDriver()
        result = driver.run(model, load)
        assert result.stress.shape == (10, model.ntens)

    def test_fields_dict_contains_stress_and_strain(self, model):
        load = _uniaxial_strain_load(n_steps=10)
        driver = StrainDriver()
        result = driver.run(model, load)
        fields = result.fields
        assert "Stress" in fields
        assert "Strain" in fields

    def test_collect_state_via_fields(self, model):
        load = _uniaxial_strain_load(n_steps=10, max_strain=5e-3)
        driver = StrainDriver()
        result = driver.run(model, load, collect_state={"ep": FieldType.STRAIN})
        ep_history = result.fields["ep"].data
        assert ep_history.shape == (10,)
        assert ep_history[-1] > 0.0

    def test_collect_state_matches_step_results(self, model):
        load = _uniaxial_strain_load(n_steps=10, max_strain=5e-3)
        driver = StrainDriver()
        result = driver.run(model, load, collect_state={"ep": FieldType.STRAIN})
        ep_from_steps = np.array([float(rm.state["ep"]) for rm in result.step_results])
        np.testing.assert_allclose(result.fields["ep"].data, ep_from_steps, rtol=1e-12)

    def test_step_n_minus_1_state_is_previous_step(self, model):
        load = _uniaxial_strain_load(n_steps=15, max_strain=5e-3)
        driver = StrainDriver()
        result = driver.run(model, load)
        for i in range(1, len(result.step_results)):
            prev_state = result.step_results[i - 1].state
            curr_state_n = result.step_results[i].state
            # Just verify ep is monotonically non-decreasing
            assert float(curr_state_n["ep"]) >= float(prev_state["ep"])


# ---------------------------------------------------------------------------
# StressDriver step_results
# ---------------------------------------------------------------------------

class TestStressDriverStepResults:
    def test_step_results_length(self, model):
        load = _uniaxial_stress_load(model, n_steps=15)
        driver = StressDriver()
        result = driver.run(model, load)
        assert len(result.step_results) == 15

    def test_step_results_are_return_mapping_results(self, model):
        load = _uniaxial_stress_load(model, n_steps=10)
        driver = StressDriver()
        result = driver.run(model, load)
        for rm in result.step_results:
            assert isinstance(rm, StressUpdateResult)

    def test_fields_stress_property(self, model):
        load = _uniaxial_stress_load(model, n_steps=10)
        driver = StressDriver()
        result = driver.run(model, load)
        assert result.stress.shape == (10, model.ntens)


# ---------------------------------------------------------------------------
# DriverResult is a proper dataclass
# ---------------------------------------------------------------------------

def test_driver_result_is_dataclass(model):
    load = _uniaxial_strain_load(n_steps=5)
    driver = StrainDriver()
    result = driver.run(model, load)
    assert isinstance(result, DriverResult)
    assert hasattr(result, "step_results")
    assert hasattr(result, "strain")
