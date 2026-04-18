"""Objective (loss) function construction for parameter fitting."""

import numpy as np

from manforge.simulation.types import FieldHistory, FieldType


def residual_sum_of_squares(stress_computed, stress_experiment, weights=None):
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


def build_objective(model, driver, exp_data, fixed_params=None):
    """Build a scalar objective function for use with scipy.optimize.

    Parameters
    ----------
    model : MaterialModel
        Template model instance — used to determine the class and stress_state.
        A new instance is constructed each evaluation with the current params.
    driver : DriverBase
        Pre-configured driver instance.
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
    stress_state = model.stress_state
    fixed = dict(fixed_params) if fixed_params else {}
    load = FieldHistory(
        FieldType.STRAIN, "Strain", np.asarray(exp_data["strain"], dtype=float)
    )
    stress_exp = np.asarray(exp_data["stress"], dtype=float)
    weights = exp_data.get("weights", None)

    def objective(free_params: dict) -> float:
        all_params = {**fixed, **free_params}
        m = model_cls(stress_state=stress_state, **all_params)
        result = driver.run(m, load)
        stress_comp = result.stress
        if stress_exp.ndim == 1:
            stress_comp = stress_comp[:, 0]
        return residual_sum_of_squares(stress_comp, stress_exp, weights)

    return objective
