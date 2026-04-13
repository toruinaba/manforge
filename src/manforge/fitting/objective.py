"""Objective (loss) function construction for parameter fitting."""

import numpy as np

from manforge.simulation.types import FieldHistory, FieldType


def residual_sum_of_squares(stress_computed, stress_experiment, weights=None):
    """Weighted sum of squared residuals.

    Parameters
    ----------
    stress_computed : array-like, shape (N,) or (N, 6)
        Model-predicted stress.
    stress_experiment : array-like, shape (N,) or (N, 6)
        Experimental / reference stress.
    weights : array-like, shape (N,) or None
        Per-sample weights.  If None, uniform weights are used.

    Returns
    -------
    float
        Σ w_i ‖σ_comp_i − σ_exp_i‖²
    """
    sc = np.asarray(stress_computed, dtype=float)
    se = np.asarray(stress_experiment, dtype=float)
    diff = sc - se

    if diff.ndim == 2:
        sq = np.sum(diff ** 2, axis=1)   # (N,)
    else:
        sq = diff ** 2                    # (N,)

    if weights is not None:
        w = np.asarray(weights, dtype=float)
        return float(np.dot(w, sq))
    return float(np.sum(sq))


def build_objective(model, driver, exp_data, fixed_params=None):
    """Build a scalar objective function for use with scipy.optimize.

    Parameters
    ----------
    model : MaterialModel
    driver : DriverBase
        Pre-configured driver instance (StrainDriver, StressDriver, …).
    exp_data : dict
        Must contain:

        - ``"strain"`` : array, shape ``(N,)`` for uniaxial or ``(N, ntens)``
          for general loading.
        - ``"stress"`` : array, matching shape of experimental stress.
          Use ``(N,)`` for uniaxial (compared against σ11 only) or
          ``(N, ntens)`` for full-tensor comparison.

        Optionally ``"weights"`` : array, shape ``(N,)``.
    fixed_params : dict or None
        Parameters held constant during optimisation.  These are merged
        with the optimised parameters before each model evaluation.

    Returns
    -------
    callable
        ``objective(param_vector: np.ndarray) -> float``

        ``param_vector`` contains only the *free* parameters in the order
        established by :func:`fit_params` (not this function directly).
        Use :func:`build_objective` via :func:`fit_params` rather than
        calling it standalone.
    """
    fixed = dict(fixed_params) if fixed_params else {}
    load = FieldHistory(
        FieldType.STRAIN, "Strain", np.asarray(exp_data["strain"], dtype=float)
    )
    stress_exp = np.asarray(exp_data["stress"], dtype=float)
    weights = exp_data.get("weights", None)

    def objective(free_params: dict) -> float:
        params = {**fixed, **free_params}
        result = driver.run(model, load, params)
        stress_comp = result.stress
        # Uniaxial experiment: compare σ11 only
        if stress_exp.ndim == 1:
            stress_comp = stress_comp[:, 0]
        return residual_sum_of_squares(stress_comp, stress_exp, weights)

    return objective
