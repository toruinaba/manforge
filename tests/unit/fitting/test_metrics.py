"""Unit tests for curve-comparison metrics — pure functions, no driver."""

import numpy as np
import pytest

from returnmap.fitting.metrics import (
    _arc_length, _measured_arc_length, _noise_amplitude, is_curve_metric,
    resolve_metric,
)
from returnmap.fitting import mean_squared_error, partial_curve_mapping


def hardening_curve(eps, E=210000.0, sy=250.0, H=1000.0):
    """Bilinear isotropic hardening σ(ε) — an analytic stand-in for a run."""
    return np.column_stack([eps, np.minimum(E * eps, (sy * E + H * E * eps) / (E + H))])


def cyclic_curve(n_per_segment, sy=285.0, E=206000.0, H=5000.0,
                 peaks=(0.015, -0.015, 0.015, -0.015, 0.015)):
    """Bilinear kinematic hardening over a cyclic strain path — hysteresis loops."""
    corners = (0.0,) + tuple(peaks)
    eps = np.concatenate([
        np.linspace(a, b, n_per_segment + 1)[1:] for a, b in zip(corners, corners[1:])
    ])
    sig = np.empty_like(eps)
    stress = alpha = eps_prev = 0.0
    for i, e in enumerate(eps):
        trial = stress + E * (e - eps_prev)
        f = abs(trial - alpha) - sy
        if f > 0.0:
            dlambda, sign = f / (E + H), np.sign(trial - alpha)
            stress, alpha = trial - E * dlambda * sign, alpha + H * dlambda * sign
        else:
            stress = trial
        sig[i], eps_prev = stress, e
    return np.column_stack([eps, sig])


def with_noise(curve, stress_sd, strain_sd=2e-6, seed=0):
    """A test record: the curve as a load cell and an extensometer report it."""
    rng = np.random.default_rng(seed)
    return curve + np.column_stack([
        rng.normal(0.0, strain_sd, len(curve)), rng.normal(0.0, stress_sd, len(curve)),
    ])


def normalised(curve, reference):
    lo, span = reference.min(axis=0), reference.max(axis=0) - reference.min(axis=0)
    return (curve - lo) / span


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


# ---------------------------------------------------------------------------
# partial_curve_mapping — measurement noise
# ---------------------------------------------------------------------------

# Noise of 3 MPa on a 600 MPa range: an ordinary load cell.
NOISE = 3.0


@pytest.mark.parametrize("n_per_segment", [400, 4000], ids=["2000 points", "20000 points"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pcm_ranks_models_correctly_against_a_noisy_test_curve(n_per_segment, seed):
    """Noise lengthens the polyline through the test points.  Matched at that
    length, the 2000-point case used to score 0.56 / 0.55 / 0.48 for the three
    models below — the true one worst."""
    test = with_noise(cyclic_curve(n_per_segment), NOISE, seed=seed)

    true = partial_curve_mapping(cyclic_curve(40), test)
    off = partial_curve_mapping(cyclic_curve(40, sy=320.0), test)
    far_off = partial_curve_mapping(cyclic_curve(40, sy=500.0), test)

    assert true < off < far_off
    assert true < 0.05
    assert off > 2.0 * true
    assert far_off > 10.0 * true


@pytest.mark.parametrize("n_per_segment", [400, 4000], ids=["2000 points", "20000 points"])
def test_pcm_of_the_true_model_barely_moves_with_noise(n_per_segment):
    clean = cyclic_curve(n_per_segment)
    computed = cyclic_curve(40)
    without = partial_curve_mapping(computed, clean)
    with_it = partial_curve_mapping(computed, with_noise(clean, NOISE))
    # Noise costs a couple of per cent of the range — the measured length is
    # good to 1–2 %, and the points compared against are themselves noisy —
    # where the plain arc length cost half of it (0.56).
    assert with_it - without < 0.03


def test_noise_amplitude_is_read_off_the_curve():
    clean = cyclic_curve(400)
    span = clean.max(axis=0) - clean.min(axis=0)
    noisy = normalised(with_noise(clean, NOISE, strain_sd=3e-5), clean)
    sd = _noise_amplitude(noisy)
    assert sd[0] == pytest.approx(3e-5 / span[0], rel=0.1)
    assert sd[1] == pytest.approx(NOISE / span[1], rel=0.1)


def test_a_curve_without_noise_has_none():
    """Yield corners and load reversals are not noise."""
    sd = _noise_amplitude(normalised(cyclic_curve(400), cyclic_curve(400)))
    assert np.all(sd < 1e-12)


@pytest.mark.parametrize("n_per_segment", [400, 4000], ids=["2000 points", "20000 points"])
def test_measured_length_is_the_length_of_the_curve_not_of_the_noise(n_per_segment):
    clean = cyclic_curve(n_per_segment)
    noisy = normalised(with_noise(clean, NOISE), clean)
    length = _arc_length(normalised(clean, clean))[1][-1]

    plain = _arc_length(noisy)[1][-1]
    measured = _measured_arc_length(noisy)

    assert plain > 1.3 * length          # what the polyline says
    assert measured[-1] == pytest.approx(length, rel=0.02)
    assert measured[0] == 0.0
    assert np.all(np.diff(measured) >= 0.0)


def test_a_curve_without_noise_keeps_its_plain_arc_length(experiment):
    for curve in (experiment, cyclic_curve(40), cyclic_curve(400)):
        curve = normalised(curve, curve)
        np.testing.assert_array_equal(_measured_arc_length(curve), _arc_length(curve)[1])


def test_too_few_points_to_estimate_noise_from():
    curve = np.column_stack([np.linspace(0.0, 1.0, 8), [0.0, 0.3, 0.1, 0.6, 0.4, 0.9, 0.7, 1.0]])
    assert np.all(_noise_amplitude(curve) == 0.0)
    np.testing.assert_array_equal(_measured_arc_length(curve), _arc_length(curve)[1])


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
