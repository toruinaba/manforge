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

Each observation's squared residual is divided by its number of data points,
so a curve does not dominate the objective merely by being longer.  Use
``Observation.weight`` to rebalance deliberately — e.g. when curves differ in
stress scale.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from manforge.fitting.objective import residual_sum_of_squares
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
        Per-data-point weights, length ``N``.
    weight : float
        Weight of this curve within the objective, applied on top of the
        automatic ``1/N`` normalisation.
    name : str
        Label used in error messages.
    """

    values: FloatArray
    field: str = STRESS_FIELD
    component: int | None = None
    weights: FloatArray | None = None
    weight: float = 1.0
    name: str = ""

    @property
    def label(self) -> str:
        return self.name or f"{self.field}[{self.component}]"

    @property
    def n_points(self) -> int:
        return len(np.asarray(self.values))

    def residual(self, result: "DriverResult") -> float:
        """Mean squared residual of this curve, scaled by :attr:`weight`."""
        computed = _extract_field(result, self.field)
        if self.component is not None:
            computed = computed[:, self.component]
        expected = np.asarray(self.values, dtype=float)
        if computed.shape != expected.shape:
            raise ValueError(
                f"observation {self.label!r}: computed {self.field} has shape "
                f"{computed.shape} but values has shape {expected.shape}"
                + ("; set component= to select a single column"
                   if computed.ndim == 2 and expected.ndim == 1 else "")
            )
        sse = residual_sum_of_squares(computed, expected, self.weights)
        return self.weight * sse / self.n_points


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
            if obs.n_points != n_steps:
                raise ValueError(
                    f"target {self.label!r}, observation {obs.label!r}: "
                    f"{obs.n_points} data points but the loading history has "
                    f"{n_steps} steps"
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
