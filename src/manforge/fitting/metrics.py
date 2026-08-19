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

from manforge._typing import FloatArray

MetricName: TypeAlias = Literal["mse", "pcm"]

PCM_N_OFFSETS = 200
"""Number of candidate arc-length offsets, following Witowski & Stander."""


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


def partial_curve_mapping(computed: FloatArray, expected: FloatArray) -> float:
    """Normalised area between two curves, minimised over arc-length offset.

    Implements the PCM objective of Witowski & Stander (2012), "Parameter
    Identification of Hysteretic Models Using Partial Curve Mapping"
    (AIAA 2012-5580).  Both arguments are ``(N, 2)`` arrays of (x, y) points;
    ``N`` need not match between them.

    Both curves are min-max normalised against the *experimental* ranges, so
    the measure is asymmetric by construction — that is the published
    intent, not an accident.  The shorter curve (by arc length) is swept along
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
    seg_long, cum_long = _arc_length(long_curve)
    seg_short, cum_short = _arc_length(short_curve)
    if cum_short[-1] > cum_long[-1]:
        long_curve, short_curve = short_curve, long_curve
        seg_long, cum_long, seg_short, cum_short = (
            seg_short, cum_short, seg_long, cum_long,
        )
    if cum_short[-1] == 0.0:
        raise ValueError("cannot compute arc length of a zero-length curve")

    max_offset = cum_long[-1] - cum_short[-1]
    offsets = [0.0] if max_offset <= 0.0 else np.linspace(0.0, max_offset, PCM_N_OFFSETS)

    # Integrate over the shorter curve, so it is matched in full against a
    # sub-portion of the longer one.  Weights are segment lengths -- the
    # trapezoid widths -- divided by the total so the result is a mean width,
    # invariant to how finely either curve is sampled.
    weights = seg_short[1:] / cum_short[-1]

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
