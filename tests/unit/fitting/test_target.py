"""Unit tests for FitTarget / Observation — residual assembly, no optimiser."""

import numpy as np
import pytest

from manforge.fitting.target import (
    FitTarget,
    Observation,
    coerce_targets,
    _extract_field,
)
from manforge.simulation.driver import StrainDriver
from manforge.simulation.integrator import PythonIntegrator
from manforge.simulation.types import FieldHistory, FieldType


@pytest.fixture
def strain_load():
    return FieldHistory(
        FieldType.STRAIN, "Strain", np.linspace(0.0, 5e-3, 10)
    )


@pytest.fixture
def driver_result(model, strain_load):
    return StrainDriver(PythonIntegrator(model)).run(strain_load)


@pytest.fixture
def driver_factory():
    return lambda integrator: StrainDriver(integrator)


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

def test_extract_stress_and_strain(driver_result):
    stress = _extract_field(driver_result, "Stress")
    strain = _extract_field(driver_result, "Strain")
    assert stress.shape == strain.shape == (10, 6)


def test_extract_state_key_without_collect_state(driver_result):
    """State keys are read off step_results, so collect_state is not needed."""
    ep = _extract_field(driver_result, "ep")
    assert ep.shape == (10,)


def test_extract_unknown_field_lists_available_keys(driver_result):
    with pytest.raises(KeyError, match="available state keys"):
        _extract_field(driver_result, "nonexistent")


# ---------------------------------------------------------------------------
# Observation residual
# ---------------------------------------------------------------------------

def test_exact_match_gives_zero_residual(driver_result):
    obs = Observation(values=driver_result.stress[:, 0], component=0)
    assert obs.residual(driver_result) == pytest.approx(0.0)


def test_residual_is_normalised_by_point_count(driver_result):
    """Residual is the *mean* squared error, not the sum."""
    offset = driver_result.stress[:, 0] + 2.0
    obs = Observation(values=offset, component=0)
    assert obs.residual(driver_result) == pytest.approx(4.0)


def test_weight_scales_residual(driver_result):
    offset = driver_result.stress[:, 0] + 2.0
    plain = Observation(values=offset, component=0)
    heavy = Observation(values=offset, component=0, weight=3.0)
    assert heavy.residual(driver_result) == pytest.approx(
        3.0 * plain.residual(driver_result)
    )


def test_full_tensor_observation(driver_result):
    obs = Observation(values=driver_result.stress, component=None)
    assert obs.residual(driver_result) == pytest.approx(0.0)


def test_shape_mismatch_hints_at_component(driver_result):
    """A 1-D curve against a tensor history should point at component=."""
    with pytest.raises(ValueError, match="set component= to select"):
        Observation(values=driver_result.stress[:, 0]).residual(driver_result)


def test_state_variable_observation(driver_result):
    ep_computed = np.array([r.state["ep"] for r in driver_result.step_results])
    obs = Observation(values=ep_computed, field="ep")
    assert obs.residual(driver_result) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# FitTarget validation
# ---------------------------------------------------------------------------

def test_target_requires_observations(strain_load):
    with pytest.raises(ValueError, match="no observations"):
        FitTarget(load=strain_load, observations=[])


def test_target_rejects_length_mismatch(strain_load):
    with pytest.raises(ValueError, match="data points but the loading history"):
        FitTarget(
            load=strain_load,
            observations=[Observation(values=np.zeros(3), component=0)],
        )


def test_target_rejects_weights_length_mismatch(strain_load):
    with pytest.raises(ValueError, match="weights for 10 steps"):
        FitTarget(
            load=strain_load,
            observations=[
                Observation(
                    values=np.zeros(10), component=0, weights=np.ones(3)
                )
            ],
        )


def test_target_without_driver_factory_reports_clearly(strain_load):
    target = FitTarget(
        load=strain_load,
        observations=[Observation(values=np.zeros(10), component=0)],
    )
    with pytest.raises(ValueError, match="no driver_factory"):
        target.resolve_driver_factory(None)


def test_target_own_driver_factory_wins(strain_load, driver_factory):
    own = lambda i: StrainDriver(i)
    target = FitTarget(
        load=strain_load,
        observations=[Observation(values=np.zeros(10), component=0)],
        driver_factory=own,
    )
    assert target.resolve_driver_factory(driver_factory) is own


# ---------------------------------------------------------------------------
# Multi-observation / multi-target summation
# ---------------------------------------------------------------------------

def test_observations_on_one_target_share_a_single_run(
    model, strain_load, driver_factory, driver_result
):
    """Two observations sum, and the driver runs only once."""
    runs = []

    def counting_factory(integrator):
        runs.append(integrator)
        return StrainDriver(integrator)

    target = FitTarget(
        load=strain_load,
        observations=[
            Observation(values=driver_result.stress[:, 0] + 2.0, component=0),
            Observation(values=driver_result.stress[:, 1] + 1.0, component=1),
        ],
        driver_factory=counting_factory,
    )
    total = target.residual(PythonIntegrator(model), driver_factory)

    assert len(runs) == 1
    assert total == pytest.approx(4.0 + 1.0)


def test_normalisation_keeps_long_curve_from_dominating(model, driver_factory):
    """Two targets with the same error but different lengths weigh equally."""
    short_load = FieldHistory(FieldType.STRAIN, "Strain", np.linspace(0.0, 5e-3, 10))
    long_load = FieldHistory(FieldType.STRAIN, "Strain", np.linspace(0.0, 5e-3, 80))
    integrator = PythonIntegrator(model)

    residuals = []
    for load in (short_load, long_load):
        computed = StrainDriver(PythonIntegrator(model)).run(load).stress[:, 0]
        target = FitTarget(
            load=load,
            observations=[Observation(values=computed + 2.0, component=0)],
        )
        residuals.append(target.residual(integrator, driver_factory))

    assert residuals[0] == pytest.approx(residuals[1])


# ---------------------------------------------------------------------------
# coerce_targets
# ---------------------------------------------------------------------------

def test_coerce_exp_data_dict():
    exp = {"strain": np.linspace(0.0, 5e-3, 10), "stress": np.zeros(10)}
    targets = coerce_targets(exp)
    assert len(targets) == 1
    obs = targets[0].observations[0]
    assert obs.field == "Stress"
    assert obs.component == 0  # 1-D stress → compare σ11 only


def test_coerce_exp_data_2d_stress_compares_all_components():
    exp = {"strain": np.linspace(0.0, 5e-3, 10), "stress": np.zeros((10, 6))}
    assert coerce_targets(exp)[0].observations[0].component is None


def test_coerce_exp_data_missing_key():
    with pytest.raises(ValueError, match=r"missing required key\(s\) \['stress'\]"):
        coerce_targets({"strain": np.zeros(10)})


def test_coerce_single_target(strain_load):
    target = FitTarget(
        load=strain_load,
        observations=[Observation(values=np.zeros(10), component=0)],
    )
    assert coerce_targets(target) == [target]


def test_coerce_sequence_of_targets(strain_load):
    target = FitTarget(
        load=strain_load,
        observations=[Observation(values=np.zeros(10), component=0)],
    )
    assert coerce_targets([target, target]) == [target, target]


def test_coerce_empty_sequence_rejected():
    with pytest.raises(ValueError, match="no fitting targets"):
        coerce_targets([])


def test_coerce_rejects_non_target_sequence():
    with pytest.raises(TypeError, match="expected FitTarget"):
        coerce_targets(["not a target"])
