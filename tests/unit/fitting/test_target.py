"""Unit tests for FitTarget / Observation — residual assembly, no optimiser."""

import numpy as np
import pytest

from manforge.fitting.target import (
    FitTarget,
    Observation,
    check_targets,
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


def test_extract_state_key_needs_no_request(driver_result):
    """State keys come from DriverResult.history, so nothing is requested up front."""
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
# FitTarget.stress_strain — the one-curve constructor
# ---------------------------------------------------------------------------

def test_stress_strain_builds_a_strain_driven_target():
    target = FitTarget.stress_strain(np.linspace(0.0, 5e-3, 10), np.zeros(10))
    assert target.load.type is FieldType.STRAIN
    assert len(target.observations) == 1
    obs = target.observations[0]
    assert obs.field == "Stress"
    assert obs.component == 0  # 1-D stress → compare σ11 only


def test_stress_strain_2d_stress_compares_all_components():
    target = FitTarget.stress_strain(np.zeros((10, 6)), np.zeros((10, 6)))
    assert target.observations[0].component is None


def test_stress_strain_forwards_weights_name_and_driver():
    weights = np.ones(10)
    target = FitTarget.stress_strain(
        np.linspace(0.0, 5e-3, 10), np.zeros(10),
        weights=weights, name="tension", driver_factory=StrainDriver,
    )
    assert target.observations[0].weights is weights
    assert target.label == "tension"
    assert target.driver_factory is StrainDriver


def test_stress_strain_checks_lengths_like_any_target():
    with pytest.raises(ValueError, match="8 data points but the loading history has 10"):
        FitTarget.stress_strain(np.linspace(0.0, 5e-3, 10), np.zeros(8))


# ---------------------------------------------------------------------------
# check_targets — the targets argument has one form
# ---------------------------------------------------------------------------

def test_check_targets_accepts_a_sequence(strain_load):
    target = FitTarget(
        load=strain_load,
        observations=[Observation(values=np.zeros(10), component=0)],
    )
    assert check_targets([target, target]) == [target, target]
    assert check_targets((target,)) == [target]


def test_check_targets_rejects_a_bare_target(strain_load):
    target = FitTarget(
        load=strain_load,
        observations=[Observation(values=np.zeros(10), component=0)],
    )
    with pytest.raises(TypeError, match="wrap the single target in a list"):
        check_targets(target)


def test_check_targets_points_the_old_dict_form_at_the_constructor():
    with pytest.raises(TypeError, match=r"FitTarget\.stress_strain"):
        check_targets({"strain": np.zeros(10), "stress": np.zeros(10)})


def test_check_targets_rejects_empty_sequence():
    with pytest.raises(ValueError, match="no fitting targets"):
        check_targets([])


def test_check_targets_rejects_non_target_sequence():
    with pytest.raises(TypeError, match="expected FitTarget"):
        check_targets(["not a target"])


# ---------------------------------------------------------------------------
# Curve metrics (metric="pcm")
# ---------------------------------------------------------------------------

def test_pcm_observation_matches_its_own_run(driver_result):
    """A curve read back off the run it came from is a perfect match."""
    obs = Observation(
        values=driver_result.stress[:, 0],
        component=0,
        metric="pcm",
        x_values=driver_result.strain[:, 0],
    )
    assert obs.residual(driver_result) == pytest.approx(0.0, abs=1e-12)


def test_pcm_observation_accepts_finer_experimental_sampling(model, strain_load):
    """The motivating case: the simulation is coarser than the experiment.

    The experiment is resampled onto 40 points from the same underlying
    curve, so a metric that tolerates differing point counts must see it as
    a near-perfect match.
    """
    coarse = StrainDriver(PythonIntegrator(model)).run(strain_load)
    strain = coarse.strain[:, 0]
    stress = coarse.stress[:, 0]

    dense_strain = np.linspace(strain[0], strain[-1], 40)
    dense_stress = np.interp(dense_strain, strain, stress)

    obs = Observation(
        values=dense_stress, component=0, metric="pcm", x_values=dense_strain
    )
    assert obs.n_points != len(coarse.step_results)
    assert obs.residual(coarse) < 1e-3


def test_pcm_observation_penalises_a_wrong_curve(driver_result):
    scaled = 1.5 * driver_result.stress[:, 0]
    obs = Observation(
        values=scaled,
        component=0,
        metric="pcm",
        x_values=driver_result.strain[:, 0],
    )
    assert obs.residual(driver_result) > 1e-3


def test_pcm_weight_scales_residual(driver_result):
    scaled = 1.5 * driver_result.stress[:, 0]
    kwargs = dict(
        component=0, metric="pcm", x_values=driver_result.strain[:, 0]
    )
    plain = Observation(values=scaled, **kwargs)
    heavy = Observation(values=scaled, weight=3.0, **kwargs)
    assert heavy.residual(driver_result) == pytest.approx(
        3.0 * plain.residual(driver_result)
    )


def test_pcm_requires_x_values():
    with pytest.raises(ValueError, match="x_values is required"):
        Observation(values=np.zeros(5), component=0, metric="pcm")


def test_pcm_rejects_mismatched_x_values_length():
    with pytest.raises(ValueError, match="x_values for 5 values"):
        Observation(
            values=np.zeros(5), component=0, metric="pcm", x_values=np.zeros(3)
        )


def test_pcm_rejects_per_point_weights():
    """Arc-length matching has no per-index correspondence to weight."""
    with pytest.raises(ValueError, match="per-point weights are meaningless"):
        Observation(
            values=np.zeros(5),
            component=0,
            metric="pcm",
            x_values=np.linspace(0.0, 1.0, 5),
            weights=np.ones(5),
        )


def test_pcm_rejects_tensor_valued_observation():
    with pytest.raises(ValueError, match="values must be 1-D"):
        Observation(
            values=np.zeros((5, 6)),
            metric="pcm",
            x_values=np.linspace(0.0, 1.0, 5),
        )


def test_mse_rejects_x_values():
    with pytest.raises(ValueError, match="x_values is only used by curve metrics"):
        Observation(
            values=np.zeros(5), component=0, x_values=np.linspace(0.0, 1.0, 5)
        )


def test_unknown_metric_rejected():
    with pytest.raises(ValueError, match="unknown metric"):
        Observation(values=np.zeros(5), component=0, metric="nope")


def test_target_allows_length_mismatch_for_curve_metric(strain_load):
    """The point-count check must not fire for a metric that permits it."""
    target = FitTarget(
        load=strain_load,
        observations=[
            Observation(
                values=np.linspace(0.0, 1.0, 37),
                component=0,
                metric="pcm",
                x_values=np.linspace(0.0, 5e-3, 37),
            )
        ],
    )
    assert target.observations[0].n_points == 37


def test_target_length_mismatch_message_suggests_pcm(strain_load):
    with pytest.raises(ValueError, match='use metric="pcm"'):
        FitTarget(
            load=strain_load,
            observations=[Observation(values=np.zeros(3), component=0)],
        )


# ---------------------------------------------------------------------------
# Non-converged runs
# ---------------------------------------------------------------------------

def test_nonconverged_run_is_raised_not_scored(model, strain_load):
    """A run reported as non-converged must not have its partial history scored.

    A driver_factory may wrap the integrator in one that reports
    non-convergence as a value.  ``driver.run`` then returns only the completed
    steps; comparing that against the full experimental curve would be a
    shape error at best.  It is raised as the same RuntimeError a raising
    integrator produces, which the objective turns into the failure penalty.
    """
    from manforge.simulation.integrator import PythonNumericalIntegrator

    def reporting(_integrator):
        # max_iter=0: the first plastic step cannot converge.
        return StrainDriver(PythonNumericalIntegrator(
            model, max_iter=0, raise_on_nonconverged=False))

    target = FitTarget(
        load=strain_load,
        observations=[Observation(values=np.zeros(10), component=0)],
        driver_factory=reporting,
        name="reporting",
    )
    with pytest.raises(RuntimeError, match="'reporting'.*did not converge at step"):
        target.residual(PythonIntegrator(model), None)
