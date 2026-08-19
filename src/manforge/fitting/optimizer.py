"""Parameter fitting API.

Main entry point: :func:`fit_params`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing_extensions import TypeAlias

from manforge.fitting.objective import ExpData, DriverFactoryFn, build_objective
from manforge.fitting.space import FitConfig, ParameterSpace
from manforge._typing import FloatArray

if TYPE_CHECKING:
    from manforge.core.material import MaterialModel

Method: TypeAlias = Literal["L-BFGS-B", "Nelder-Mead", "differential_evolution"]


@dataclass
class FitResult:
    """Result of a parameter fitting run.

    Attributes
    ----------
    params : dict[str, float]
        Optimised parameter values — free variables, fixed values, and
        derived parameters alike.
    residual : float
        Objective function value at the optimum.
    success : bool
        ``True`` if the optimiser reported convergence.
    n_iter : int
        Number of objective function evaluations.
    message : str
        Optimiser status message.
    history : list[dict[str, float]]
        Free-variable values at each optimiser callback (populated only when
        ``method="L-BFGS-B"`` or ``"Nelder-Mead"``; empty otherwise).
    """

    params: dict[str, float]
    residual: float
    success: bool
    n_iter: int
    message: str = ""
    history: list[dict[str, float]] = field(default_factory=list)


def fit_params(
    model: "MaterialModel",
    driver_factory: DriverFactoryFn,
    exp_data: ExpData,
    fit_config: ParameterSpace | FitConfig,
    fixed_params: dict[str, float] | None = None,
    method: Method = "L-BFGS-B",
    *,
    max_iter: int = 1000,
    tol: float | None = None,
    options: Mapping[str, Any] | None = None,
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
    fit_config : ParameterSpace or dict
        A :class:`~manforge.fitting.space.ParameterSpace`, or the plain dict
        form for the case where every free variable is a model parameter::

            {
                "sigma_y0": (initial_value, (lower_bound, upper_bound)),
                "H":        (500.0,         (0.0, 10000.0)),
            }

        Use ``None`` (or ``(None, None)``) as bounds for unbounded parameters.
        Use ``ParameterSpace`` when parameters are correlated — see its
        docstring for the ``Y < B`` ratio example.
    fixed_params : dict or None
        Parameters held constant during optimisation.  Dict form only;
        with a ``ParameterSpace`` declare them in ``ParameterSpace(fixed=...)``.
    method : str
        Optimisation algorithm:
        - ``"L-BFGS-B"``              — gradient-free quasi-Newton (default)
        - ``"Nelder-Mead"``           — simplex, no bounds
        - ``"differential_evolution"``— global stochastic (slow but robust)
    max_iter : int
        Maximum optimiser iterations.  Set small (e.g. 2) to keep tests fast.
    tol : float or None
        Convergence tolerance passed to the optimiser's primary tolerance
        (``ftol`` for L-BFGS-B, ``fatol`` for Nelder-Mead, ``tol`` for
        differential_evolution).  ``None`` keeps the method default.
    options : Mapping or None
        Extra options forwarded verbatim to scipy, overriding the above.

    Returns
    -------
    FitResult
    """
    space = ParameterSpace.coerce(fit_config, fixed_params)
    free_names = space.free_names
    bounds = space.bounds

    obj_fn = build_objective(model, driver_factory, exp_data, space)

    history: list[dict[str, float]] = []

    def _callback(x: FloatArray) -> None:
        history.append(dict(zip(free_names, np.asarray(x, dtype=float).tolist())))

    extra = dict(options) if options else {}

    # ------------------------------------------------------------------ #
    if method == "differential_evolution":
        # differential_evolution needs finite bounds
        de_bounds = [(max(b[0], -1e9), min(b[1], 1e9)) for b in bounds]
        de_kwargs: dict[str, Any] = {
            "seed": 0,
            "tol": tol if tol is not None else 1e-8,
            "maxiter": max_iter,
        }
        de_kwargs.update(extra)
        result = differential_evolution(
            obj_fn,
            de_bounds,
            callback=lambda xk, convergence=None: _callback(xk),
            **de_kwargs,
        )
    else:
        opts: dict[str, Any]
        if method == "Nelder-Mead":
            opts = {"maxiter": max_iter, "xatol": 1e-8, "fatol": 1e-10}
            if tol is not None:
                opts["fatol"] = tol
        else:
            opts = {"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8}
            if tol is not None:
                opts["ftol"] = tol
        opts.update(extra)
        result = minimize(
            obj_fn,
            space.x0,
            method=method,
            bounds=bounds if method != "Nelder-Mead" else None,
            callback=_callback,
            options=opts,
        )

    return FitResult(
        params=space.resolve(result.x),
        residual=float(result.fun),
        success=bool(result.success),
        n_iter=int(result.nfev),
        message=result.message if hasattr(result, "message") else "",
        history=history,
    )
