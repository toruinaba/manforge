"""Parameter fitting API.

Main entry point: :func:`fit_params`.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize, differential_evolution

from manforge.fitting.objective import build_objective


@dataclass
class FitResult:
    """Result of a parameter fitting run.

    Attributes
    ----------
    params : dict
        Optimised parameter values (free + fixed).
    residual : float
        Objective function value at the optimum.
    success : bool
        ``True`` if the optimiser reported convergence.
    n_iter : int
        Number of objective function evaluations.
    message : str
        Optimiser status message.
    history : list[dict]
        Parameter values at each optimiser callback (populated only when
        ``method="L-BFGS-B"`` or ``"Nelder-Mead"``; empty otherwise).
    """

    params: dict
    residual: float
    success: bool
    n_iter: int
    message: str = ""
    history: list = field(default_factory=list)


def fit_params(
    model,
    driver_factory,
    exp_data: dict,
    fit_config: dict,
    fixed_params: dict | None = None,
    method: str = "L-BFGS-B",
) -> FitResult:
    """Fit material parameters to experimental stress-strain data.

    Parameters
    ----------
    model : MaterialModel
        Constitutive model instance.
    driver_factory : callable
        ``driver_factory(integrator) -> DriverBase`` — constructs a driver
        bound to the given integrator for each objective evaluation.
        Example: ``lambda i: StrainDriver(i)``.
    exp_data : dict
        ``{"strain": ..., "stress": ..., "weights": ...}``  (weights optional).
    fit_config : dict
        Free parameters to optimise, format::

            {
                "sigma_y0": (initial_value, (lower_bound, upper_bound)),
                "H":        (500.0,         (0.0, 10000.0)),
            }

        Use ``(None, None)`` as bounds for unbounded parameters.
    fixed_params : dict or None
        Parameters held constant during optimisation.
    method : str
        Optimisation algorithm:
        - ``"L-BFGS-B"``              — gradient-free quasi-Newton (default)
        - ``"Nelder-Mead"``           — simplex, no bounds
        - ``"differential_evolution"``— global stochastic (slow but robust)

    Returns
    -------
    FitResult
    """
    fixed = dict(fixed_params) if fixed_params else {}

    # Extract ordered free-parameter names, initial values, and bounds
    free_names = list(fit_config.keys())
    x0 = np.array([fit_config[n][0] for n in free_names], dtype=float)
    bounds = [fit_config[n][1] for n in free_names]  # list of (lb, ub) or None

    # Normalise bounds: replace None with (-inf, inf)
    bounds_clean = [
        ((-np.inf if b is None else b[0]),
         ( np.inf if b is None else b[1]))
        if b is not None else (-np.inf, np.inf)
        for b in bounds
    ]

    # Build objective — accepts a dict of free params
    obj_fn = build_objective(model, driver_factory, exp_data, fixed_params=fixed)

    history: list[dict] = []

    def _scalar_obj(x: np.ndarray) -> float:
        free = dict(zip(free_names, x))
        return obj_fn(free)

    def _callback(x: np.ndarray) -> None:
        history.append(dict(zip(free_names, x.tolist())))

    # ------------------------------------------------------------------ #
    if method == "differential_evolution":
        # differential_evolution needs finite bounds
        de_bounds = [
            (max(b[0], -1e9), min(b[1], 1e9))
            for b in bounds_clean
        ]
        result = differential_evolution(
            _scalar_obj,
            de_bounds,
            seed=0,
            tol=1e-8,
            maxiter=1000,
            callback=lambda xk, convergence=None: _callback(xk),
        )
    else:
        if method == "Nelder-Mead":
            opts = {"maxiter": 1000, "xatol": 1e-8, "fatol": 1e-10}
        else:
            opts = {"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8}
        result = minimize(
            _scalar_obj,
            x0,
            method=method,
            bounds=bounds_clean if method != "Nelder-Mead" else None,
            callback=_callback,
            options=opts,
        )

    # Reconstruct full parameter dict
    best_free = dict(zip(free_names, result.x.tolist()))
    best_params = {**fixed, **best_free}

    return FitResult(
        params=best_params,
        residual=float(result.fun),
        success=bool(result.success),
        n_iter=int(result.nfev),
        message=result.message if hasattr(result, "message") else "",
        history=history,
    )
