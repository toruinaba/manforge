"""Parameter fitting API.

Main entry point: :func:`fit_params`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing_extensions import TypeAlias

from manforge.fitting.objective import (
    ExpData,
    DriverFactoryFn,
    FAILURE_PENALTY,
    FailureLog,
    build_objective,
)
from manforge.fitting.space import FitConfig, ParameterSpace
from manforge.fitting.target import FitTarget, coerce_targets
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
    n_failed : int
        Evaluations the model could not complete, scored as ``inf`` so the
        optimiser avoids that region.  A nonzero count is not itself a problem
        — the parameter box can legitimately include unsolvable values — but a
        large fraction of ``n_iter`` means the search spent its budget there,
        so tighten the bounds or revisit the initial values.
    failure_message : str
        The first failure's exception text, empty when none occurred.
    failure_params : dict[str, float]
        Parameters that produced the first failure — the starting point for
        reproducing it outside the fit.
    """

    params: dict[str, float]
    residual: float
    success: bool
    n_iter: int
    message: str = ""
    history: list[dict[str, float]] = field(default_factory=list)
    n_failed: int = 0
    failure_message: str = ""
    failure_params: dict[str, float] = field(default_factory=dict)


def _out_of_bounds_initials(space: ParameterSpace) -> list[str]:
    """Free variables whose initial value sits outside their own bounds.

    scipy silently clips these into the box, so the fit does not start where
    the caller wrote — worth saying out loud when something else then fails.
    """
    offenders = []
    for name, (lo, hi) in zip(space.free_names, space.bounds):
        init = space.free[name][0]
        if not (lo <= init <= hi):
            offenders.append(f"{name}={init:g} outside ({lo:g}, {hi:g})")
    return offenders


def _clipped_x0(space: ParameterSpace) -> FloatArray:
    """``x0`` clipped into the bounds box — the point scipy actually starts at.

    Probing the raw ``x0`` would check a *different* point than the search
    uses whenever an initial value sits outside its bounds, and the diagnosis
    ("cannot start" vs "search left the solvable region") would be wrong.
    """
    lo, hi = np.array(space.bounds, dtype=float).T
    return np.clip(space.x0, lo, hi)


def _check_initial_point(
    obj_fn: Callable[[Sequence[float] | FloatArray], float],
    space: ParameterSpace,
    failures: FailureLog,
) -> None:
    """Raise with a setup-focused message if the objective fails at ``x0``."""
    if obj_fn(_clipped_x0(space)) < FAILURE_PENALTY:
        return

    detail = failures.first_message or "objective was not finite"
    message = (
        "fitting cannot start: the objective is not finite at the initial "
        f"parameters, before any search happens. Cause: {detail}. The model "
        "cannot be integrated at these parameters, so check the initial "
        "values; if the loading history was decimated, note that the wider "
        "strain increment can make an otherwise workable parameter set "
        "unsolvable."
    )
    if offenders := _out_of_bounds_initials(space):
        message += (
            " Also, these initial values sit outside their own bounds and were "
            f"clipped, so the fit does not start where you wrote it: "
            f"{', '.join(offenders)}."
        )
    raise RuntimeError(message)


def fit_params(
    model: "MaterialModel",
    driver_factory: DriverFactoryFn | None,
    exp_data: "ExpData | FitTarget | Sequence[FitTarget]",
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
    driver_factory : callable or None
        ``driver_factory(integrator) -> DriverBase`` — constructs a driver
        bound to the given integrator for each objective evaluation.
        Example: ``lambda i: StrainDriver(i)``.  May be ``None`` only if every
        target carries its own ``driver_factory``.
    exp_data : dict, FitTarget, or sequence of FitTarget
        The curves to fit.  ``{"strain": ..., "stress": ..., "weights": ...}``
        is shorthand for a single strain-driven target with one stress curve.
        Pass :class:`~manforge.fitting.target.FitTarget` instances to fit
        several experiments at once, to drive a target by stress, or to
        compare state variables as well as stress.  Each observation's squared
        residual is divided by its point count before summing, so a longer
        curve does not dominate; use ``Observation.weight`` to rebalance.
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
    targets = coerce_targets(exp_data)
    free_names = space.free_names
    bounds = space.bounds

    failures = FailureLog()
    obj_fn = build_objective(model, driver_factory, targets, space, failures)

    # Probe the starting point before handing over to scipy.  If x0 itself is
    # unsolvable the optimiser cannot compute a first gradient and every
    # evaluation fails -- a setup problem that reads very differently from
    # "the search wandered somewhere bad", so it gets its own diagnosis.
    _check_initial_point(obj_fn, space, failures)

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

    # Every evaluation failing means no parameter set in the box was solvable,
    # so there is no optimum to report -- that is a setup problem, and silently
    # returning the initial guess would hide it.
    if float(result.fun) >= FAILURE_PENALTY:
        # x0 was solvable (checked above), so the search itself left the
        # solvable region -- that points at the bounds, not the initial values.
        message = (
            f"fitting failed: the initial parameters were solvable but no point "
            f"the optimiser tried was ({failures.n_failed} of "
            f"{int(result.nfev)} evaluations could not be completed), so the "
            f"search left the region where the model can be integrated. "
            f"Consider tightening the bounds. First failure at "
            f"{failures.first_params}: {failures.first_message}"
        )
        if offenders := _out_of_bounds_initials(space):
            message += (
                f" Note also these clipped initial values: {', '.join(offenders)}."
            )
        raise RuntimeError(message)

    return FitResult(
        params=space.resolve(result.x),
        residual=float(result.fun),
        success=bool(result.success),
        n_iter=int(result.nfev),
        message=result.message if hasattr(result, "message") else "",
        history=history,
        n_failed=failures.n_failed,
        failure_message=failures.first_message,
        failure_params=failures.first_params,
    )
