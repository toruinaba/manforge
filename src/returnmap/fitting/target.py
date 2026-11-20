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

from returnmap.fitting.metrics import (
    MetricName,
    is_curve_metric,
    mean_squared_error,
    partial_curve_mapping,
    resolve_metric,
)
from returnmap.simulation.driver import default_driver_for
from returnmap.simulation.types import FieldHistory, FieldType
from returnmap._typing import FloatArray

if TYPE_CHECKING:
    from returnmap.fitting.objective import DriverFactoryFn
    from returnmap.simulation.integrator import StressIntegrator
    from returnmap.simulation.types import DriverResult

STRESS_FIELD = "stress"
STRAIN_FIELD = "strain"


def _extract_field(result: "DriverResult", name: str) -> FloatArray:
    """Pull a computed history out of a driver result by field name.

    Accepts ``"stress"``, ``"strain"``, or any model state key — the names
    :meth:`~returnmap.simulation.DriverResult.history` takes.
    """
    try:
        return result.history(name)
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
        What to compare: ``"stress"``, ``"strain"``, or a model state key
        (e.g. ``"ep"``) — the names
        :meth:`~returnmap.simulation.DriverResult.history` takes.
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
        :func:`~returnmap.fitting.partial_curve_mapping`, which
        compares (x, y) curves by arc length and so permits different point
        counts — use it when the simulation is forced onto a coarser
        increment than the experiment was sampled at, or when the curve is
        multi-valued.  Requires ``x_values``.  Note that a coarse
        increment biases a joint fit around the yield corner; see
        :func:`~returnmap.fitting.partial_curve_mapping`.
    x_values : array-like or None
        Experimental x coordinates, length ``N`` — required by ``"pcm"`` and
        rejected by ``"mse"``.  The computed x history is taken from
        ``x_field``.
    x_field : str
        Which computed history supplies the x axis for a curve metric.
        Defaults to ``"strain"``.
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
        """Residual of this curve under ``metric``, scaled by ``weight``."""
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
        Loading history for this run.  Its :class:`~returnmap.simulation.FieldType` must match the
        driver — ``STRAIN`` for :class:`~returnmap.simulation.StrainDriver`,
        ``STRESS`` for :class:`~returnmap.simulation.StressDriver`.
    observations : sequence of Observation
        Curves compared against this run's outputs.  All share the one run.
    driver_factory : callable or None
        ``driver_factory(integrator) -> DriverBase`` for this target.
        ``None`` (default) uses the one passed to
        :func:`~returnmap.fitting.fit_params`, which is the right
        choice unless this target is driven differently from the others.
        If neither is given the driver follows from ``load``:
        :class:`~returnmap.simulation.StrainDriver` for a strain history,
        :class:`~returnmap.simulation.StressDriver` for a stress history.
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

    @classmethod
    def stress_strain(
        cls,
        strain,
        stress,
        *,
        weights=None,
        name: str = "",
        driver_factory: "DriverFactoryFn | None" = None,
    ) -> "FitTarget":
        """One strain-driven run compared against one stress curve.

        The common case — a measured stress-strain curve — without spelling
        out the :class:`~returnmap.simulation.FieldHistory` and the :class:`Observation`.

        Parameters
        ----------
        strain : array-like, shape (N,) or (N, ntens)
            Strain history, used as the load.
        stress : array-like, shape (N,) or (N, ntens)
            Measured stress.  A 1-D array is compared with component 0 of the
            computed stress; a 2-D array with all of it.
        weights : array-like, shape (N,), or None
            Per-point weights of the stress residual.
        name, driver_factory
            As in :class:`FitTarget`.
        """
        strain = np.asarray(strain, dtype=float)
        stress = np.asarray(stress, dtype=float)
        return cls(
            load=FieldHistory(FieldType.STRAIN, strain),
            observations=[
                Observation(
                    values=stress,
                    field=STRESS_FIELD,
                    component=0 if stress.ndim == 1 else None,
                    weights=weights,
                )
            ],
            driver_factory=driver_factory,
            name=name,
        )

    @property
    def label(self) -> str:
        return self.name or f"<{self.load.type.name.lower()}-driven>"

    def resolve_driver_factory(
        self, default: "DriverFactoryFn | None"
    ) -> "DriverFactoryFn":
        """This target's own factory, else ``default``, else the one the load implies."""
        return self.driver_factory or default or default_driver_for(self.load)

    def residual(
        self, integrator: "StressIntegrator", default_factory: "DriverFactoryFn | None",
    ) -> float:
        """Run this target once and sum its observations' residuals."""
        driver = self.resolve_driver_factory(default_factory)(integrator)
        result = driver.run(
            self.load,
            initial_stress=self.initial_stress,
            initial_state=dict(self.initial_state) if self.initial_state else None,
        )
        if not result.converged:
            # Only reachable when the driver_factory wraps the integrator in
            # one that reports non-convergence as a value.  A truncated history
            # cannot be scored; raise the same error a raising integrator
            # would, so the objective penalises it instead of comparing curves
            # of different lengths.
            raise RuntimeError(
                f"target {self.label!r}: the run did not converge at step "
                f"{result.failed_step}"
            )
        return sum(obs.residual(result) for obs in self.observations)


def check_targets(targets) -> list[FitTarget]:
    """Validate the ``targets`` argument: a non-empty sequence of FitTarget."""
    if isinstance(targets, FitTarget):
        raise TypeError(
            "targets must be a sequence of FitTarget; wrap the single target "
            "in a list"
        )
    if isinstance(targets, Mapping):
        raise TypeError(
            "targets must be a sequence of FitTarget, not a dict; for one "
            "stress-strain curve use [FitTarget.stress_strain(strain, stress)]"
        )
    targets = list(targets)
    if not targets:
        raise ValueError("no fitting targets given")
    bad = [t for t in targets if not isinstance(t, FitTarget)]
    if bad:
        raise TypeError(
            f"expected FitTarget instances, got {type(bad[0]).__name__}"
        )
    return targets
