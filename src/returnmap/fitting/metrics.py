"""Curve-comparison metrics for fitting objectives.

Two families, differing in what they require of the data:

* :func:`mean_squared_error` compares values point-by-point.  Requires the
  computed and experimental curves to be sampled identically.
* :func:`partial_curve_mapping` compares (x, y) curves via arc-length
  parameterisation.  Point counts and x ranges may differ, and multi-valued
  curves (hysteresis loops, unloading branches) are handled correctly.

Select via ``Observation(metric=...)``.
"""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np
from typing_extensions import TypeAlias

from returnmap._typing import FloatArray

MetricName: TypeAlias = Literal["mse", "pcm"]

PCM_N_OFFSETS = 200
"""Number of candidate arc-length offsets, following Witowski & Stander."""

PCM_RULER_PER_NOISE = 10.0
"""Length of the ruler the experimental curve is measured with, in units of
its estimated noise amplitude.

Ten is where both errors of the measured length are small.  Shorter, and the
noise still lengthens the curve; longer, and the ruler steps across the tips
of load reversals.  Measured on a 20 000-point cyclic curve with noise of
1.2 % of the stress range, the length came out +15 % at six noise amplitudes,
−0.8 % at ten and −5 % at twenty.
"""


class PointwiseMetric(Protocol):
    """``metric(computed, expected, weights) -> float`` — needs matched sampling."""
    def __call__(
        self,
        computed: FloatArray,
        expected: FloatArray,
        weights: FloatArray | None = None,
    ) -> float: ...


class CurveMetric(Protocol):
    """``metric(computed_xy, expected_xy) -> float`` — sampling may differ."""
    def __call__(self, computed: FloatArray, expected: FloatArray) -> float: ...


def mean_squared_error(
    computed: FloatArray,
    expected: FloatArray,
    weights: FloatArray | None = None,
) -> float:
    """Weighted squared residual divided by the number of data points.

    For a 2-D history the squared error is summed across components at each
    point, so the result stays a per-point average.
    """
    diff = np.asarray(computed, dtype=float) - np.asarray(expected, dtype=float)
    sq = np.sum(diff ** 2, axis=1) if diff.ndim == 2 else diff ** 2

    if weights is not None:
        total = float(np.dot(np.asarray(weights, dtype=float), sq))
    else:
        total = float(np.sum(sq))
    return total / len(sq)


def _arc_length(xy: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Per-segment and cumulative arc lengths, in the curve's own units."""
    seg = np.zeros(len(xy))
    seg[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return seg, np.cumsum(seg)


def _noise_amplitude(xy: FloatArray) -> FloatArray:
    """Standard deviation of the point-to-point noise in x and in y.

    Estimated from fourth differences along the curve.  A fourth difference
    cancels any locally cubic trend, so on a smooth curve it is the noise
    alone, with standard deviation ``sqrt(70) * sigma`` (the squared binomial
    coefficients 1, 4, 6, 4, 1 sum to 70).  The median absolute deviation is
    used rather than the variance so that the few points where the curve is
    not smooth — the yield corner, a load reversal — do not count as noise.
    A curve without noise gives zero, up to round-off.
    """
    if len(xy) < 9:
        return np.zeros(2)
    d4 = np.diff(xy, n=4, axis=0)
    mad = np.median(np.abs(d4 - np.median(d4, axis=0)), axis=0)
    return 1.4826 * mad / np.sqrt(70.0)


def _measured_arc_length(xy: FloatArray) -> FloatArray:
    """Cumulative arc length of a measured curve, with its noise taken out.

    The polyline through noisy points is longer than the curve they were
    measured on, and by an amount that grows with the sampling rate: every
    point steps sideways by the noise amplitude, however short the step
    along the curve.  Summing the segments of a densely sampled test curve
    can give several times its real length.

    The length is therefore taken with a ruler instead (the divider method):
    walk along the points, and place the next mark at the first point at
    least one ruler away from the last mark.  With the ruler
    :data:`PCM_RULER_PER_NOISE` noise amplitudes long, the sideways steps no
    longer matter.  What is left — each chord is still the distance between
    two noisy points, so its square is too large by twice the noise variance
    across it — is subtracted.

    Only the *parameterisation* changes.  The points themselves are kept, so
    the curve is compared at full resolution; each point between two marks is
    placed between them in proportion to its index.  A curve without noise
    has a ruler of zero length and gets its plain arc length.
    """
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    sd = _noise_amplitude(xy)
    ruler = PCM_RULER_PER_NOISE * float(np.hypot(sd[0], sd[1]))
    if ruler <= 0.0 or ruler <= float(np.min(seg, initial=np.inf)):
        # Every segment is already a ruler long: nothing to take out.
        return np.concatenate([[0.0], np.cumsum(seg)])

    marks = [0]
    last = xy[0]
    for i in range(1, len(xy)):
        if np.hypot(xy[i, 0] - last[0], xy[i, 1] - last[1]) >= ruler:
            marks.append(i)
            last = xy[i]
    if marks[-1] != len(xy) - 1:
        marks.append(len(xy) - 1)

    step = np.diff(xy[marks], axis=0)
    chord = np.linalg.norm(step, axis=1)
    direction = step / np.where(chord > 0.0, chord, 1.0)[:, None]
    # Variance of the noise across the chord: x noise acts across a vertical
    # chord, y noise across a horizontal one.
    across = sd[0] ** 2 * direction[:, 1] ** 2 + sd[1] ** 2 * direction[:, 0] ** 2
    chord = np.sqrt(np.maximum(chord ** 2 - 2.0 * across, 0.0))
    at_marks = np.concatenate([[0.0], np.cumsum(chord)])
    return np.interp(np.arange(len(xy)), marks, at_marks)


def partial_curve_mapping(computed: FloatArray, expected: FloatArray) -> float:
    """Normalised area between two curves, minimised over arc-length offset.

    Implements the PCM objective of Witowski & Stander (2012), "Parameter
    Identification of Hysteretic Models Using Partial Curve Mapping"
    (AIAA 2012-5580).  Both arguments are ``(N, 2)`` arrays of (x, y) points;
    ``N`` need not match between them.

    ``expected`` is the measured curve and may be noisy; ``computed`` is taken
    to be exact.  Both curves are min-max normalised against the
    *experimental* ranges, so the measure is asymmetric by construction — that
    is the published intent, not an accident.  The shorter curve (by arc length) is swept along
    the longer one and the smallest enclosed area is returned, which is what
    makes the comparison "partial": a computed curve extending beyond the
    experimental range is not penalised for the excess.

    Notes
    -----
    Not a drop-in replacement for ``similaritymeasures.pcm``, which multiplies
    each trapezoid width by the *cumulative* arc length instead of the segment
    length.  That makes its return value scale linearly with the point count
    rather than converging to an area.  This implementation uses the segment
    length, so the result is invariant to resampling.

    Measurement noise.  The two curves are matched at equal arc length, and
    the polyline through noisy points is longer than the curve itself — the
    more densely it is sampled, the longer.  Taken at face value that
    mismatch decides the result: against a 2000-point cyclic test curve with
    noise of 0.4 % of the stress range, the plain arc length is half as long
    again (ten times as long at 20 000 points), the computed curve is matched
    to the wrong part of the test, and a model with the yield stress at
    nearly twice its true value scores *better* than the true one.  The experimental arc length is therefore measured
    with a ruler ten noise amplitudes long (``_measured_arc_length``); the
    noise amplitude is estimated from the curve itself.  A curve without
    noise is unaffected.  The price is resolution at the scale of the ruler,
    which steps across yield corners and the tips of load reversals: with
    noise of 0.4 % of the range the measured length comes out about 1 % short,
    and a few per cent short once the noise exceeds 1 %.  The value at the
    true parameters is then 0.01–0.05 rather than zero, and differences
    between models smaller than that are not resolved.

    Because the comparison is geometric, a coarsely incremented simulation
    does not merely add a constant offset: where the response has a corner
    (the yield point), a coarse polyline cuts across it, and that chord can be
    traded against the material parameters.  Measured on bilinear isotropic
    hardening against a 200-point reference, the recovered hardening modulus
    was off by 12% at 30 increments, 6% at 120, and exact at 400 — while the
    same fit by ``mean_squared_error`` on matched sampling was exact
    throughout.  A parameter sweep in one variable at a time is unbiased; the
    error appears only in the joint fit, where yield stress absorbs the
    corner-cutting.  Refine the increments near the corner, or check the
    result against a refined run, before trusting a joint fit.
    """
    comp = np.asarray(computed, dtype=float)
    exp = np.asarray(expected, dtype=float)
    if comp.ndim != 2 or comp.shape[1] != 2:
        raise ValueError(f"computed curve must have shape (N, 2), got {comp.shape}")
    if exp.ndim != 2 or exp.shape[1] != 2:
        raise ValueError(f"expected curve must have shape (N, 2), got {exp.shape}")

    span = exp.max(axis=0) - exp.min(axis=0)
    if np.any(span == 0.0):
        raise ValueError(
            "experimental curve is degenerate (zero range in x or y), so it "
            "cannot be normalised; PCM needs a curve with extent in both axes"
        )
    norm_exp = (exp - exp.min(axis=0)) / span
    norm_comp = (comp - exp.min(axis=0)) / span

    # Arc lengths stay in normalised-coordinate units.  Rescaling each curve
    # to unit length -- as similaritymeasures does -- would make both span
    # [0, 1] and collapse the offset range to zero, silently removing the
    # partial match this metric exists to provide.
    long_curve, short_curve = norm_exp, norm_comp
    cum_long = _measured_arc_length(long_curve)
    _, cum_short = _arc_length(short_curve)
    if cum_short[-1] > cum_long[-1]:
        long_curve, short_curve = short_curve, long_curve
        cum_long, cum_short = cum_short, cum_long
    if cum_short[-1] == 0.0:
        raise ValueError("cannot compute arc length of a zero-length curve")

    max_offset = cum_long[-1] - cum_short[-1]
    offsets = [0.0] if max_offset <= 0.0 else np.linspace(0.0, max_offset, PCM_N_OFFSETS)

    # Integrate over the shorter curve, so it is matched in full against a
    # sub-portion of the longer one.  Weights are segment lengths -- the
    # trapezoid widths -- divided by the total so the result is a mean width,
    # invariant to how finely either curve is sampled.
    weights = np.diff(cum_short) / cum_short[-1]

    areas = []
    for offset in offsets:
        probe = cum_short + offset
        mapped = np.column_stack([
            np.interp(probe, cum_long, long_curve[:, 0]),
            np.interp(probe, cum_long, long_curve[:, 1]),
        ])
        d = np.linalg.norm(short_curve - mapped, axis=1)
        areas.append(float(np.sum(0.5 * (d[:-1] + d[1:]) * weights)))
    return min(areas)


POINTWISE_METRICS: dict[str, PointwiseMetric] = {"mse": mean_squared_error}
CURVE_METRICS: dict[str, CurveMetric] = {"pcm": partial_curve_mapping}


def is_curve_metric(metric: MetricName) -> bool:
    """Whether ``metric`` compares (x, y) curves rather than matched values."""
    return metric in CURVE_METRICS


def resolve_metric(metric: MetricName) -> PointwiseMetric | CurveMetric:
    if metric in POINTWISE_METRICS:
        return POINTWISE_METRICS[metric]
    if metric in CURVE_METRICS:
        return CURVE_METRICS[metric]
    known = sorted(POINTWISE_METRICS) + sorted(CURVE_METRICS)
    raise ValueError(f"unknown metric {metric!r}; available: {known}")
