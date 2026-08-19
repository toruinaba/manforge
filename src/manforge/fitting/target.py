"""Fitting targets — the experimental curves an objective is fitted to.

Two levels, because the two ways of "having several targets" cost different
amounts of compute:

* :class:`FitTarget` is one *simulation run* — one loading history driven by
  one driver.  Separate experiments (monotonic tension, cyclic at several
  strain amplitudes, a stress-controlled ratchetting test) are separate
  targets, each costing a full driver run per objective evaluation.
* :class:`Observation` is one *experimental curve* compared against one
  computed output of that run.  Several observations on the same target
  (σ11 and σ22, or stress and equivalent plastic strain) share a single run.

Each observation carries its own ``metric``.  The default ``"mse"`` divides the
squared residual by the number of data points, so a curve does not dominate
the objective merely by being longer; ``"pcm"`` compares curves by arc length
and tolerates differing point counts.  Use ``Observation.weight`` to rebalance
deliberately — e.g. when curves differ in stress scale, or when mixing metrics
whose values are not commensurate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from manforge.fitting.metrics import (
    MetricName,
    is_curve_metric,
    mean_squared_error,
    partial_curve_mapping,
    resolve_metric,
)
from manforge.simulation.types import FieldHistory, FieldType
from manforge._typing import FloatArray

if TYPE_CHECKING:
    from manforge.fitting.objective import DriverFactoryFn, ExpData
    from manforge.simulation.types import DriverResult

STRESS_FIELD = "Stress"
STRAIN_FIELD = "Strain"


def _extract_field(result: "DriverResult", name: str) -> FloatArray:
    """Pull a computed history out of a driver result by field name.

    Accepts ``"Stress"``, ``"Strain"``, or any model state key.  State keys
    are read straight off ``step_results`` rather than via
    ``DriverResult.fields``, which would require a ``collect_state`` request
    naming a :class:`FieldType` the residual has no use for.
    """
    if name == STRESS_FIELD:
        return np.asarray(result.stress, dtype=float)
    if name == STRAIN_FIELD:
        return np.asarray(result.strain, dtype=float)
    try:
        return np.stack(
            [np.asarray(r.state[name], dtype=float) for r in result.step_results]
        )
    except KeyError:
        available = sorted(result.step_results[0].state) if result.step_results else []
        raise KeyError(
            f"field {name!r} is neither {STRESS_FIELD!r}/{STRAIN_FIELD!r} nor a "
            f"model state key; available state keys: {available}"
        ) from None


@dataclass(frozen=True)
class Observation:
    """One experimental curve and the computed output it is compared against.

    Parameters
    ----------
    values : array-like
        Experimental values.  Shape must match the computed history after
        ``component`` selection — ``(N,)`` for a scalar or a single selected
        component, ``(N, ntens)`` for a full tensor history.
    field : str
        What to compare: ``"Stress"``, ``"Strain"``, or a model state key
        (e.g. ``"ep"``).
    component : int or None
        Column of the computed history to compare.  ``None`` (default)
        compares every column, so ``values`` must then be 2-D.
    weights : array-like or None
        Per-data-point weights, length ``N``.  Only meaningful for
        ``metric="mse"`` — a curve metric has no per-point correspondence to
        weight, so combining the two raises.
    weight : float
        Weight of this curve within the objective, applied on top of the
        metric's own normalisation.
    metric : {"mse", "pcm"}
        How to compare the curves.  ``"mse"`` (default) is a point-by-point
        mean squared error and requires ``values`` to be sampled at the same
        points as the simulation.  ``"pcm"`` is
        :func:`~manforge.fitting.metrics.partial_curve_mapping`, which
        compares (x, y) curves by arc length and so permits different point
        counts — use it when the simulation is forced onto a coarser
        increment than the experiment was sampled at, or when the curve is
        multi-valued.  Requires :attr:`x_values`.  Note that a coarse
        increment biases a joint fit around the yield corner; see
        :func:`~manforge.fitting.metrics.partial_curve_mapping`.
    x_values : array-like or None
        Experimental x coordinates, length ``N`` — required by ``"pcm"`` and
        rejected by ``"mse"``.  The computed x history is taken from
        :attr:`x_field`.
    x_field : str
        Which computed history supplies the x axis for a curve metric.
        Defaults to ``"Strain"``.
    x_component : int or None
        Column of the computed x history.  ``None`` (default) means column 0
        when that history is 2-D.
    name : str
        Label used in error messages.
    """

    values: FloatArray
    field: str = STRESS_FIELD
    component: int | None = None
    weights: FloatArray | None = None
    weight: float = 1.0
    metric: MetricName = "mse"
    x_values: FloatArray | None = None
    x_field: str = STRAIN_FIELD
    x_component: int | None = None
    name: str = ""

    def __post_init__(self) -> None:
        resolve_metric(self.metric)
        if is_curve_metric(self.metric):
            if self.x_values is None:
                raise ValueError(
                    f"observation {self.label!r}: metric {self.metric!r} compares "
                    f"(x, y) curves, so x_values is required"
                )
            if len(np.asarray(self.x_values)) != self.n_points:
                raise ValueError(
                    f"observation {self.label!r}: {len(np.asarray(self.x_values))} "
                    f"x_values for {self.n_points} values"
                )
            if self.weights is not None:
                raise ValueError(
                    f"observation {self.label!r}: per-point weights are meaningless "
                    f"for metric {self.metric!r}, which matches points by arc length "
                    f"rather than by index; use weight= to scale the whole curve"
                )
            if np.asarray(self.values).ndim != 1:
                raise ValueError(
                    f"observation {self.label!r}: metric {self.metric!r} compares a "
                    f"single (x, y) curve, so values must be 1-D; set component= to "
                    f"select one column"
                )
        elif self.x_values is not None:
            raise ValueError(
                f"observation {self.label!r}: x_values is only used by curve metrics, "
                f"not by {self.metric!r}"
            )

    @property
    def compares_curves(self) -> bool:
        """Whether this observation's metric permits differing point counts."""
        return is_curve_metric(self.metric)

    @property
    def label(self) -> str:
        return self.name or f"{self.field}[{self.component}]"

    @property
    def n_points(self) -> int:
        return len(np.asarray(self.values))

    def residual(self, result: "DriverResult") -> float:
        """Residual of this curve under :attr:`metric`, scaled by :attr:`weight`."""
        computed = _extract_field(result, self.field)
        if self.component is not None:
            computed = computed[:, self.component]
        expected = np.asarray(self.values, dtype=float)

        if self.compares_curves:
            return self.weight * partial_curve_mapping(
                np.column_stack([self._computed_x(result), computed]),
                np.column_stack([np.asarray(self.x_values, dtype=float), expected]),
            )

        if computed.shape != expected.shape:
            raise ValueError(
                f"observation {self.label!r}: computed {self.field} has shape "
                f"{computed.shape} but values has shape {expected.shape}"
                + ("; set component= to select a single column"
                   if computed.ndim == 2 and expected.ndim == 1 else "")
            )
        return self.weight * mean_squared_error(computed, expected, self.weights)

    def _computed_x(self, result: "DriverResult") -> FloatArray:
        """The simulated x history a curve metric plots against."""
        x = _extract_field(result, self.x_field)
        if x.ndim == 2:
            x = x[:, 0 if self.x_component is None else self.x_component]
        elif self.x_component is not None:
            raise ValueError(
                f"observation {self.label!r}: x_component={self.x_component} but "
                f"computed {self.x_field} is 1-D"
            )
        return x


@dataclass(frozen=True)
class FitTarget:
    """One simulation run and the experimental curves extracted from it.

    Parameters
    ----------
    load : FieldHistory
        Loading history for this run.  Its :class:`FieldType` must match the
        driver — ``STRAIN`` for :class:`~manforge.simulation.driver.StrainDriver`,
        ``STRESS`` for :class:`~manforge.simulation.driver.StressDriver`.
    observations : sequence of Observation
        Curves compared against this run's outputs.  All share the one run.
    driver_factory : callable or None
        ``driver_factory(integrator) -> DriverBase`` for this target.
        ``None`` (default) uses the one passed to
        :func:`~manforge.fitting.optimizer.fit_params`, which is the right
        choice unless this target is driven differently from the others.
    initial_stress, initial_state : array-like / dict or None
        Forwarded to ``driver.run`` — use for a prestressed specimen or a
        pre-hardened initial state.
    name : str
        Label used in error messages.
    """

    load: FieldHistory
    observations: Sequence[Observation]
    driver_factory: "DriverFactoryFn | None" = None
    initial_stress: FloatArray | None = None
    initial_state: Mapping[str, FloatArray] | None = None
    name: str = ""

    def __post_init__(self) -> None:
        if not self.observations:
            raise ValueError(
                f"target {self.label!r} has no observations to fit against"
            )
        n_steps = len(np.asarray(self.load.data))
        for obs in self.observations:
            # Curve metrics match by arc length, so a coarser simulation than
            # the experiment's sampling is the expected case, not an error.
            if obs.n_points != n_steps and not obs.compares_curves:
                raise ValueError(
                    f"target {self.label!r}, observation {obs.label!r}: "
                    f"{obs.n_points} data points but the loading history has "
                    f"{n_steps} steps; use metric=\"pcm\" to compare curves "
                    f"sampled differently"
                )
            if obs.weights is not None and len(np.asarray(obs.weights)) != n_steps:
                raise ValueError(
                    f"target {self.label!r}, observation {obs.label!r}: "
                    f"{len(np.asarray(obs.weights))} weights for {n_steps} steps"
                )

    @property
    def label(self) -> str:
        return self.name or f"<{self.load.type.name.lower()}-driven>"

    def resolve_driver_factory(
        self, default: "DriverFactoryFn | None"
    ) -> "DriverFactoryFn":
        factory = self.driver_factory or default
        if factory is None:
            raise ValueError(
                f"target {self.label!r} has no driver_factory and none was "
                f"passed to fit_params"
            )
        return factory

    def residual(self, integrator: object, default_factory: "DriverFactoryFn | None") -> float:
        """Run this target once and sum its observations' residuals."""
        driver = self.resolve_driver_factory(default_factory)(integrator)
        result = driver.run(
            self.load,
            initial_stress=self.initial_stress,
            initial_state=dict(self.initial_state) if self.initial_state else None,
        )
        return sum(obs.residual(result) for obs in self.observations)


def coerce_targets(
    exp_data: "ExpData | FitTarget | Sequence[FitTarget]",
) -> list[FitTarget]:
    """Accept a single legacy ``ExpData`` dict, one target, or several."""
    if isinstance(exp_data, FitTarget):
        return [exp_data]
    if isinstance(exp_data, Mapping):
        return [_target_from_exp_data(exp_data)]
    targets = list(exp_data)
    if not targets:
        raise ValueError("no fitting targets given")
    bad = [t for t in targets if not isinstance(t, FitTarget)]
    if bad:
        raise TypeError(
            f"expected FitTarget instances, got {type(bad[0]).__name__}"
        )
    return targets


def _target_from_exp_data(exp_data: "ExpData") -> FitTarget:
    """Wrap the ``{"strain": ..., "stress": ...}`` dict form as one target."""
    missing = {"strain", "stress"} - set(exp_data)
    if missing:
        raise ValueError(
            f"exp_data is missing required key(s) {sorted(missing)}"
        )
    strain = np.asarray(exp_data["strain"], dtype=float)
    stress = np.asarray(exp_data["stress"], dtype=float)
    return FitTarget(
        load=FieldHistory(FieldType.STRAIN, STRAIN_FIELD, strain),
        observations=[
            Observation(
                values=stress,
                field=STRESS_FIELD,
                component=0 if stress.ndim == 1 else None,
                weights=exp_data.get("weights"),
            )
        ],
        name="exp_data",
    )
