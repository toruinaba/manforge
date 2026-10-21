"""Objective (loss) function construction for parameter fitting."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np
from typing_extensions import NotRequired, TypedDict

from manforge.fitting.space import ParameterSpace
from manforge.simulation.integrator import PythonIntegrator
from manforge._typing import FloatArray

FAILURE_PENALTY = 1e30
"""Objective value assigned to an evaluation the model could not complete.

Large enough that no real fit ever reaches it, but deliberately *finite*:
returning ``inf`` breaks L-BFGS-B, whose finite-difference gradient at a point
with an ``inf`` neighbour evaluates to ``nan`` and silently poisons the line
search.  A huge finite value keeps the gradient finite (and enormous), so the
optimiser backs away from the failing region instead.
"""

if TYPE_CHECKING:
    from manforge.core.material import MaterialModel
    from manforge.fitting.target import FitTarget
    from manforge.simulation.driver import DriverBase


@dataclass
class FailureLog:
    """Records evaluations the model could not complete.

    A constitutive model need not be solvable everywhere in a box-bounded
    parameter space — return mapping can legitimately have no real root — so a
    failed evaluation is information ("not the optimum") rather than an error.
    The objective reports it as ``inf`` and records it here, which keeps a long
    fit alive instead of discarding every prior evaluation.
    """

    n_failed: int = 0
    first_message: str = ""
    first_params: dict[str, float] = field(default_factory=dict)

    def record(self, exc: Exception, params: dict[str, float]) -> None:
        self.n_failed += 1
        if self.n_failed == 1:
            self.first_message = f"{type(exc).__name__}: {exc}"
            self.first_params = dict(params)


class ExpData(TypedDict):
    """Single stress-strain curve — the shorthand for a one-target fit.

    Equivalent to one :class:`~manforge.fitting.target.FitTarget` with a
    strain-driven load and one stress :class:`~manforge.fitting.target.Observation`.
    """
    strain: FloatArray
    stress: FloatArray
    weights: NotRequired[FloatArray | None]


class DriverFactoryFn(Protocol):
    """``driver_factory(integrator) -> DriverBase`` — constructs a driver for one objective evaluation."""
    def __call__(self, integrator: object) -> "DriverBase": ...


def build_objective(
    model: "MaterialModel",
    driver_factory: DriverFactoryFn | None,
    targets: Sequence["FitTarget"],
    space: ParameterSpace,
    failures: FailureLog | None = None,
) -> Callable[[Sequence[float] | FloatArray], float]:
    """Build a scalar objective function for use with scipy.optimize.

    The objective is the sum over every target's observations of each
    observation's residual under its own metric.  Each target costs one driver
    run per evaluation; observations on the same target share that run.

    An evaluation the model cannot complete — return mapping or the driver's
    outer NR failing to converge, or a singular NR Jacobian — yields the large
    finite :data:`FAILURE_PENALTY` rather than propagating, so the optimiser
    steers away from that region instead of the whole fit dying on one bad
    trial point.  Errors that indicate a broken *setup* rather than an
    unsolvable parameter value (shape mismatches, unknown field names) are
    left to propagate.

    Parameters
    ----------
    model : MaterialModel
        Template model instance — used to determine the class and dimension.
        A new instance is constructed each evaluation with the current params.
    driver_factory : callable or None
        ``driver_factory(integrator) -> DriverBase`` — the default driver for
        targets that do not carry their own.  Example: ``lambda i: StrainDriver(i)``.
    targets : sequence of FitTarget
        Already-coerced targets — see
        :func:`~manforge.fitting.target.coerce_targets`.
    space : ParameterSpace
        Maps the optimisation vector onto model parameters.
    failures : FailureLog or None
        Collects the non-convergent evaluations.  ``None`` (default) still
        penalises them, but does not report which ones.

    Returns
    -------
    callable
        ``objective(x: Sequence[float]) -> float``
    """
    model_cls = type(model)
    dimension = model.dimension
    _accepts_dimension = "dimension" in inspect.signature(model_cls.__init__).parameters
    param_names = model.param_names
    targets = list(targets)

    def objective(x: Sequence[float] | FloatArray) -> float:
        all_params = space.model_params(x, param_names)
        extra = {"dimension": dimension} if _accepts_dimension else {}
        m = model_cls(**extra, **all_params)  # type: ignore[call-arg]
        integrator = PythonIntegrator(m)
        # LinAlgError is listed even though it subclasses ValueError (which is
        # otherwise a setup error left to propagate): a singular NR Jacobian
        # is precisely "this parameter value is unsolvable", not a bug.
        try:
            total = sum(t.residual(integrator, driver_factory) for t in targets)
        except (
            RuntimeError, FloatingPointError, ZeroDivisionError,
            np.linalg.LinAlgError,
        ) as exc:
            if failures is not None:
                failures.record(exc, all_params)
            return FAILURE_PENALTY
        # A model can converge to nan without raising -- an inf/nan objective
        # would make the optimiser's line search meaningless, so treat it the
        # same way as an outright failure.
        if not math.isfinite(total):
            if failures is not None:
                failures.record(
                    ArithmeticError(f"objective evaluated to {total}"), all_params
                )
            return FAILURE_PENALTY
        return total

    return objective
