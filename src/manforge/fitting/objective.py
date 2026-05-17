"""Objective (loss) function construction for parameter fitting."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import numpy as np
from typing_extensions import NotRequired, TypedDict

from manforge.simulation.types import FieldHistory, FieldType
from manforge.simulation.integrator import PythonIntegrator
from manforge._typing import FloatArray

if TYPE_CHECKING:
    from manforge.core.material import MaterialModel
    from manforge.simulation.driver import DriverBase


class ExpData(TypedDict):
    """Experimental stress-strain data for parameter fitting."""
    strain: FloatArray
    stress: FloatArray
    weights: NotRequired[FloatArray | None]


class DriverFactoryFn(Protocol):
    """``driver_factory(integrator) -> DriverBase`` — constructs a driver for one objective evaluation."""
    def __call__(self, integrator: object) -> "DriverBase": ...


def residual_sum_of_squares(
    stress_computed: FloatArray,
    stress_experiment: FloatArray,
    weights: FloatArray | None = None,
) -> float:
    """Weighted sum of squared residuals."""
    sc = np.asarray(stress_computed, dtype=float)
    se = np.asarray(stress_experiment, dtype=float)
    diff = sc - se

    if diff.ndim == 2:
        sq = np.sum(diff ** 2, axis=1)
    else:
        sq = diff ** 2

    if weights is not None:
        w = np.asarray(weights, dtype=float)
        return float(np.dot(w, sq))
    return float(np.sum(sq))


def build_objective(
    model: "MaterialModel",
    driver_factory: DriverFactoryFn,
    exp_data: ExpData,
    fixed_params: dict[str, float] | None = None,
) -> Callable[[dict[str, float]], float]:
    """Build a scalar objective function for use with scipy.optimize.

    Parameters
    ----------
    model : MaterialModel
        Template model instance — used to determine the class and dimension.
        A new instance is constructed each evaluation with the current params.
    driver_factory : callable
        ``driver_factory(integrator) -> DriverBase`` — called once per
        objective evaluation to construct a driver bound to the current
        integrator.  Example: ``lambda i: StrainDriver(i)``.
    exp_data : dict
        Must contain ``"strain"`` and ``"stress"``.  Optionally ``"weights"``.
    fixed_params : dict or None
        Parameters held constant during optimisation.

    Returns
    -------
    callable
        ``objective(free_params: dict) -> float``
    """
    model_cls = type(model)
    dimension = model.dimension
    _accepts_dimension = "dimension" in inspect.signature(model_cls.__init__).parameters
    fixed = dict(fixed_params) if fixed_params else {}
    load = FieldHistory(
        FieldType.STRAIN, "Strain", np.asarray(exp_data["strain"], dtype=float)
    )
    stress_exp = np.asarray(exp_data["stress"], dtype=float)
    weights = exp_data.get("weights", None)

    def objective(free_params: dict[str, float]) -> float:
        all_params = {**fixed, **free_params}
        extra = {"dimension": dimension} if _accepts_dimension else {}
        m = model_cls(**extra, **all_params)  # type: ignore[call-arg]
        integrator = PythonIntegrator(m)
        result = driver_factory(integrator).run(load)
        stress_comp = result.stress
        if stress_exp.ndim == 1:
            stress_comp = stress_comp[:, 0]
        return residual_sum_of_squares(stress_comp, stress_exp, weights)

    return objective
