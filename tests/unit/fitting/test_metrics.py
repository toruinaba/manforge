"""Unit tests for curve-comparison metrics — pure functions, no driver."""

import numpy as np
import pytest

from manforge.fitting.metrics import (
    is_curve_metric,
    mean_squared_error,
    partial_curve_mapping,
    resolve_metric,
)


def hardening_curve(eps, E=210000.0, sy=250.0, H=1000.0):
    """Bilinear isotropic hardening σ(ε) — an analytic stand-in for a run."""
    return np.column_stack([eps, np.minimum(E * eps, (sy * E + H * E * eps) / (E + H))])


@pytest.fixture
def experiment():
    """Finely sampled reference curve, as an experiment would be recorded."""
    return hardening_curve(np.linspace(0.0, 0.05, 200))


# ---------------------------------------------------------------------------
# mean_squared_error
# ---------------------------------------------------------------------------

def test_mse_is_per_point_average():
    computed = np.array([1.0, 2.0, 3.0])
    expected = np.array([1.0, 2.0, 4.0])
    assert mean_squared_error(computed, expected) == pytest.approx(1.0 / 3.0)


def test_mse_sums_components_then_averages_over_points():
    """A 2-D history stays a per-point average, not a per-entry one."""
    computed = np.zeros((4, 2))
    expected = np.ones((4, 2))
    assert mean_squared_error(computed, expected) == pytest.approx(2.0)


def test_mse_applies_weights():
    computed = np.array([0.0, 0.0])
    expected = np.array([1.0, 1.0])
    weights = np.array([3.0, 1.0])
    assert mean_squared_error(computed, expected, weights) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# partial_curve_mapping — the invariants that make it usable as an objective
# ---------------------------------------------------------------------------

def test_pcm_is_zero_for_identical_curves(experiment):
    assert partial_curve_mapping(experiment, experiment) == pytest.approx(0.0)


def test_pcm_is_invariant_to_resampling():
    """The published measure is an area, so it must not scale with point count.

    This is the property ``similaritymeasures.pcm`` loses by weighting each
    trapezoid with the cumulative rather than the segment arc length.
    """
    values = []
    for n in (50, 100, 200, 400, 800):
        x = np.linspace(0.0, 1.0, n)
        reference = np.column_stack([x, x])
        shifted = np.column_stack([x, x + 0.1])
        values.append(partial_curve_mapping(shifted, reference))

    assert values == pytest.approx([0.1] * 5, abs=1e-12)


def test_pcm_minimised_at_true_hardening_modulus(experiment):
    """Sensitivity survives a simulation 5x coarser than the experiment."""
    coarse = np.linspace(0.0, 0.05, 40)
    losses = {
        H: partial_curve_mapping(hardening_curve(coarse, H=H), experiment)
        for H in (400.0, 700.0, 1000.0, 1400.0, 2000.0)
    }

    assert min(losses, key=losses.get) == 1000.0
    assert losses[700.0] < losses[400.0]
    assert losses[1400.0] < losses[2000.0]


def test_pcm_minimised_at_true_yield_stress(experiment):
    """The offset sweep must not absorb a vertical shift in the curve."""
    coarse = np.linspace(0.0, 0.05, 40)
    losses = {
        sy: partial_curve_mapping(hardening_curve(coarse, sy=sy), experiment)
        for sy in (200.0, 225.0, 250.0, 275.0, 300.0)
    }

    assert min(losses, key=losses.get) == 250.0
    assert losses[225.0] < losses[200.0]
    assert losses[275.0] < losses[300.0]


def test_pcm_discretisation_floor_shrinks_with_refinement(experiment):
    """Correct parameters leave a residual set by the simulation's coarseness."""
    coarse = partial_curve_mapping(
        hardening_curve(np.linspace(0.0, 0.05, 20)), experiment
    )
    fine = partial_curve_mapping(
        hardening_curve(np.linspace(0.0, 0.05, 160)), experiment
    )

    assert fine < coarse
    assert coarse < 0.01


def test_pcm_floor_is_parameter_dependent_near_a_corner():
    """A coarse grid cuts the yield corner, letting wrong params beat the truth.

    This is the documented caveat, pinned so the bias is not mistaken for a
    constant offset that optimisation could ignore.  The joint (σy, H) fit is
    the affected case; refining the increments removes it.
    """
    experiment = hardening_curve(np.linspace(0.0, 0.02, 200), H=1200.0)
    coarse = np.linspace(0.0, 0.02, 30)

    at_truth = partial_curve_mapping(
        hardening_curve(coarse, sy=250.0, H=1200.0), experiment
    )
    at_wrong = partial_curve_mapping(
        hardening_curve(coarse, sy=252.0, H=1000.0), experiment
    )

    assert at_wrong < at_truth

    refined = np.linspace(0.0, 0.02, 400)
    assert partial_curve_mapping(
        hardening_curve(refined, sy=250.0, H=1200.0), experiment
    ) < partial_curve_mapping(
        hardening_curve(refined, sy=252.0, H=1000.0), experiment
    )


def test_pcm_single_parameter_sweep_is_unbiased(experiment):
    """With the other parameters held true, the sweep finds the true value."""
    coarse = np.linspace(0.0, 0.05, 30)
    losses = {
        sy: partial_curve_mapping(hardening_curve(coarse, sy=sy), experiment)
        for sy in (246.0, 248.0, 250.0, 252.0, 254.0)
    }
    assert min(losses, key=losses.get) == 250.0


def test_pcm_handles_multi_valued_curves():
    """Arc-length parameterisation is the point: a loop has two y per x."""
    dense = np.linspace(0.0, 2.0 * np.pi, 200)
    sparse = np.linspace(0.0, 2.0 * np.pi, 37)
    circle = np.column_stack([np.cos(dense), np.sin(dense)])

    same_loop = partial_curve_mapping(
        np.column_stack([np.cos(sparse), np.sin(sparse)]), circle
    )
    wrong_loop = partial_curve_mapping(
        np.column_stack([np.cos(sparse), 1.3 * np.sin(sparse)]), circle
    )

    assert same_loop < 1e-2
    assert wrong_loop > 10.0 * same_loop


def test_pcm_is_asymmetric():
    """Normalising against the first argument's range is the published intent."""
    x = np.linspace(0.0, 1.0, 100)
    shallow = np.column_stack([x, x])
    steep = np.column_stack([x, 2.0 * x])

    assert partial_curve_mapping(shallow, steep) != pytest.approx(
        partial_curve_mapping(steep, shallow)
    )


def test_pcm_rejects_degenerate_experiment():
    x = np.linspace(0.0, 1.0, 5)
    flat = np.column_stack([x, np.zeros(5)])
    with pytest.raises(ValueError, match="degenerate"):
        partial_curve_mapping(flat, flat)


def test_pcm_rejects_non_xy_shape(experiment):
    with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
        partial_curve_mapping(np.zeros((10, 3)), experiment)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_resolve_metric_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown metric"):
        resolve_metric("not_a_metric")


def test_only_pcm_compares_curves():
    assert is_curve_metric("pcm")
    assert not is_curve_metric("mse")
