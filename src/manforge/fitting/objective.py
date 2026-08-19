"""Objective (loss) function construction for parameter fitting."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol

from typing_extensions import NotRequired, TypedDict

from manforge.fitting.space import ParameterSpace
from manforge.simulation.integrator import PythonIntegrator
from manforge._typing import FloatArray

if TYPE_CHECKING:
    from manforge.core.material import MaterialModel
    from manforge.fitting.target import FitTarget
    from manforge.simulation.driver import DriverBase


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
) -> Callable[[Sequence[float] | FloatArray], float]:
    """Build a scalar objective function for use with scipy.optimize.

    The objective is the sum over every target's observations of each
    observation's residual under its own metric.  Each target costs one driver
    run per evaluation; observations on the same target share that run.

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
        return sum(t.residual(integrator, driver_factory) for t in targets)

    return objective
