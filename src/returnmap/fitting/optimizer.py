"""Parameter fitting API.

Main entry point: :func:`fit_params`.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing_extensions import TypeAlias

from returnmap.core.material import MaterialModel
from returnmap.fitting.objective import (
    DriverFactoryFn,
    FAILURE_PENALTY,
    FailureLog,
    build_objective,
)
from returnmap.fitting.space import ParameterSpace
from returnmap.fitting.target import FitTarget, check_targets
from returnmap._typing import FloatArray, NRJacobian

Method: TypeAlias = Literal["L-BFGS-B", "Nelder-Mead", "differential_evolution"]

SLOW_EVALUATION_SECONDS = 0.5
"""Time for one objective evaluation above which :func:`fit_params` warns.

A fit evaluates the objective a few hundred times (45 to 412 over the fits
this was measured on), so half a second each is where the wait turns from
seconds into minutes.
"""


class SlowFitWarning(UserWarning):
    """One objective evaluation is slow enough that the fit will take minutes.

    Issued once by :func:`fit_params`, before the search starts, with the
    measured time, the reason when the model is integrated by automatic
    differentiation, and what would shorten it.  Silence it with
    ``warnings.simplefilter("ignore", SlowFitWarning)``.
    """


def _slow_fit_message(
    model: MaterialModel, targets: Sequence[FitTarget], seconds: float,
    nr_jacobian: NRJacobian,
) -> str:
    n_steps = sum(len(np.asarray(t.load.data)) for t in targets)
    autodiff = (
        type(model).user_defined_return_mapping
        is MaterialModel.user_defined_return_mapping
    )
    parts = [
        f"one evaluation of the objective takes {seconds:.1f} s ({n_steps} load "
        "steps) and a fit needs a few hundred of them, so expect minutes."
    ]
    remedies = []
    if autodiff:
        parts.append(
            f"{type(model).__name__} has no hand-written return mapping: every "
            "plastic step is solved by automatic differentiation, which costs "
            "milliseconds where a closed form costs microseconds."
        )
        if nr_jacobian == "exact":
            remedies.append('nr_jacobian="broyden"')
    remedies.append(
        'fewer load steps (metric="pcm" compares a coarse run against an '
        "experiment sampled more finely)"
    )
    if autodiff:
        remedies.append("a model that has a hand-written return mapping")
    parts.append("To shorten it: " + "; ".join(remedies) + ".")
    parts.append("progress=print shows each evaluation as it completes.")
    return " ".join(parts)


@dataclass
class FitResult:
    """Result of a parameter fitting run.

    Attributes
    ----------
    params : dict[str, float]
        Every name of the parameter space at the optimum — free variables,
        fixed values and derived parameters alike.  This includes helper
        variables that are not model parameters (a ratio the search runs
        on, say); use ``model_params`` for the model's own.
    model_params : dict[str, float]
        The model parameters at the optimum: exactly the model's
        ``param_names``.
    model : MaterialModel
        The fitted model — the template passed to
        :func:`~returnmap.fitting.fit_params` with ``model_params`` in place
        (same class, dimension and non-parameter settings).
    residual : float
        Objective function value at the optimum.
    success : bool
        ``True`` if the optimiser reported convergence.
    n_evaluations : int
        Number of objective function evaluations.
    message : str
        Optimiser status message.
    history : list[dict[str, float]]
        Free-variable values at each optimiser callback (populated only when
        ``method="L-BFGS-B"`` or ``"Nelder-Mead"``; empty otherwise).
    n_failed : int
        Evaluations the model could not complete, scored with the finite
        penalty ``FAILURE_PENALTY`` so the optimiser avoids that region.  A nonzero count is not itself a problem
        — the parameter box can legitimately include unsolvable values — but a
        large fraction of ``n_evaluations`` means the search spent its budget there,
        so tighten the bounds or revisit the initial values.
    failure_message : str
        The first failure's exception text, empty when none occurred.
    failure_params : dict[str, float]
        Parameters that produced the first failure — the starting point for
        reproducing it outside the fit.
    """

    params: dict[str, float]
    model_params: dict[str, float]
    model: "MaterialModel"
    residual: float
    success: bool
    n_evaluations: int
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


@dataclass(frozen=True)
class _SearchScale:
    """The free variables as the optimiser sees them: each about one unit wide.

    Material parameters differ by orders of magnitude — a hardening modulus of
    2e4 next to a recovery coefficient of 1e2.  Handed over as they are, a
    gradient method takes steps sized for the small ones and leaves the large
    ones where they started: measured on an Armstrong-Frederick fit, L-BFGS-B
    reported convergence with ``C_k`` still at its initial value.  The search
    therefore runs on ``u = (x - offset) / scale``, where ``scale`` is the
    width of the variable's bounds, so that every variable crosses its range
    as ``u`` goes from 0 to 1.  A variable without two finite bounds is scaled
    by the magnitude of its initial value instead (by 1 if that is zero).

    Only the optimiser's view changes; the objective, the bounds the caller
    wrote and everything reported back are in the caller's units.
    """

    offset: FloatArray
    scale: FloatArray
    lower: FloatArray
    upper: FloatArray

    @classmethod
    def of(cls, space: ParameterSpace) -> "_SearchScale":
        lower, upper = np.array(space.bounds, dtype=float).T
        x0 = _clipped_x0(space)
        boxed = np.isfinite(lower) & np.isfinite(upper) & (upper > lower)
        magnitude = np.where(x0 != 0.0, np.abs(x0), 1.0)
        return cls(
            offset=np.where(boxed, lower, 0.0),
            scale=np.where(boxed, upper - lower, magnitude),
            lower=lower,
            upper=upper,
        )

    def to_search(self, x: Sequence[float] | FloatArray) -> FloatArray:
        return (np.asarray(x, dtype=float) - self.offset) / self.scale

    def to_params(self, u: Sequence[float] | FloatArray) -> FloatArray:
        # Clipped: offset + scale * 1.0 can land one rounding step past the bound.
        x = self.offset + self.scale * np.asarray(u, dtype=float)
        return np.clip(x, self.lower, self.upper)

    @property
    def bounds(self) -> list[tuple[float, float]]:
        lo, hi = self.to_search(self.lower), self.to_search(self.upper)
        return list(zip(lo.tolist(), hi.tolist()))


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


def _reporting(
    obj_fn: Callable[[Sequence[float] | FloatArray], float],
    progress: Callable[[int, float, float], None],
) -> Callable[[Sequence[float] | FloatArray], float]:
    """``obj_fn`` that calls ``progress(n, residual, best)`` after each evaluation."""
    n_done, best = 0, float("inf")

    def evaluate(x: Sequence[float] | FloatArray) -> float:
        nonlocal n_done, best
        value = obj_fn(x)
        n_done += 1
        best = min(best, value)
        progress(n_done, value, best)
        return value

    return evaluate


def fit_params(
    model: "MaterialModel",
    targets: Sequence[FitTarget],
    space: ParameterSpace,
    *,
    driver_factory: DriverFactoryFn | None = None,
    method: Method = "L-BFGS-B",
    max_iter: int = 1000,
    tol: float | None = None,
    options: Mapping[str, Any] | None = None,
    nr_jacobian: NRJacobian = "exact",
    progress: Callable[[int, float, float], None] | None = None,
) -> FitResult:
    """Fit material parameters to experimental curves.

    One form: a model, the targets, the parameter space::

        result = fit_params(
            model,
            [FitTarget.stress_strain(strain, stress)],
            ParameterSpace(
                free={"sigma_y0": (200.0, (100.0, 400.0)), "H": (500.0, (0.0, 1e4))},
                fixed={"E": 210_000.0, "nu": 0.3},
            ),
        )
        fitted = result.model          # the template with the fitted parameters

    Parameters
    ----------
    model : MaterialModel
        Template model: it supplies the class, the dimension and any
        non-parameter setting.  Its parameter values are not used — every
        evaluation takes them from ``space``
        (:meth:`~returnmap.core.MaterialModel.with_params`).
    targets : sequence of FitTarget
        The runs to fit against, one
        :class:`~returnmap.fitting.FitTarget` per experiment.  For a
        single measured stress-strain curve,
        ``[FitTarget.stress_strain(strain, stress)]``.  Each observation's
        squared residual is divided by its point count before summing, so a
        longer curve does not dominate; use ``Observation.weight`` to
        rebalance.
    space : ParameterSpace
        Free variables with initial values and bounds, fixed values, and
        derived parameters.  It must supply every model parameter.  Use
        ``derived`` when parameters are correlated — see
        :class:`~returnmap.fitting.ParameterSpace` for the ``Y < B``
        ratio example.
    driver_factory : callable or None
        ``driver_factory(integrator) -> DriverBase`` — constructs the driver
        for each objective evaluation.  ``None`` (default) lets each
        target's load decide, as :func:`~returnmap.verification.crosscheck`
        does: :class:`~returnmap.simulation.StrainDriver` for a strain
        history, :class:`~returnmap.simulation.StressDriver` for a stress
        history.  Pass a factory for anything else — uniaxial stress on a 3D
        model is ``lambda i: MixedDriver(i, prescribed_strain_idx=[0])``.  A
        target's own ``driver_factory`` takes precedence over this one.
    method : str
        Optimisation algorithm:
        - ``"L-BFGS-B"``              — bounded quasi-Newton, gradient by finite differences (default)
        - ``"Nelder-Mead"``           — simplex, no bounds
        - ``"differential_evolution"``— global stochastic (slow but robust)

        The first two search on the free variables rescaled by the width of
        their bounds, so parameters of very different magnitude are treated
        alike; give every free variable a finite range that reflects where it
        can plausibly lie.  Results are reported in the units you wrote.
    max_iter : int
        Maximum optimiser iterations.
    tol : float or None
        The optimiser's convergence tolerance (not the return-mapping NR's,
        which belongs to the integrator): passed to its primary tolerance
        (``ftol`` for L-BFGS-B, ``fatol`` for Nelder-Mead, ``tol`` for
        differential_evolution).  ``None`` keeps the method default.
    options : Mapping or None
        Extra options forwarded verbatim to scipy, overriding the above.
    nr_jacobian : {"exact", "broyden"}
        NR Jacobian strategy for the integrator each evaluation constructs —
        see :class:`~returnmap.simulation.PythonIntegrator`.
        ``"broyden"`` is several times faster per evaluation on models
        without hand-derived solvers (measured ~4× on YU) and a fit pays
        that cost at every evaluation; the converged stresses agree with
        ``"exact"`` to within the NR tolerance.  Opt-in because the extra
        1–2 NR iterations per step can matter near the edge of the solvable
        parameter region.
    progress : callable or None
        Called as ``progress(n_evaluations, residual, best_residual)`` after
        every objective evaluation of the search — the residual of that
        evaluation and the smallest one so far.  ``progress=print`` is enough
        to see that a long fit is alive and whether it is still improving.

    Returns
    -------
    FitResult

    Warns
    -----
    SlowFitWarning
        When the first evaluation of the objective takes longer than
        :data:`~returnmap.fitting.SLOW_EVALUATION_SECONDS` — typically a model
        without a hand-written return mapping run over many load steps.  The
        fit goes ahead; the message says how long one evaluation takes and
        what would shorten it.
    """
    if not isinstance(space, ParameterSpace):
        raise TypeError(
            "space must be a ParameterSpace; for the plain case write "
            "ParameterSpace(free={name: (initial, (lower, upper))}, fixed={...})"
        )
    targets = check_targets(targets)
    free_names = space.free_names
    bounds = space.bounds

    failures = FailureLog()
    obj_fn = build_objective(
        model, targets, space,
        driver_factory=driver_factory, failures=failures,
        nr_jacobian=nr_jacobian,
    )

    # Probe the starting point before handing over to scipy.  If x0 itself is
    # unsolvable the optimiser cannot compute a first gradient and every
    # evaluation fails -- a setup problem that reads very differently from
    # "the search wandered somewhere bad", so it gets its own diagnosis.
    started = time.perf_counter()
    _check_initial_point(obj_fn, space, failures)
    seconds = time.perf_counter() - started
    if seconds > SLOW_EVALUATION_SECONDS:
        warnings.warn(
            _slow_fit_message(model, targets, seconds, nr_jacobian),
            SlowFitWarning, stacklevel=2,
        )

    if progress is not None:
        # After the probe above, so the count matches FitResult.n_evaluations.
        obj_fn = _reporting(obj_fn, progress)

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
        search = _SearchScale.of(space)
        if method == "Nelder-Mead":
            # Run without bounds, as documented: scaled by them, not held to them.
            search = replace(search, lower=-np.inf * np.ones_like(search.lower),
                             upper=np.inf * np.ones_like(search.upper))
        # The clipped point, for every method: it is the one probed above.
        # L-BFGS-B clips by itself; Nelder-Mead is run without bounds and
        # would otherwise start from an out-of-range value nobody checked.
        result = minimize(
            lambda u: obj_fn(search.to_params(u)),
            search.to_search(_clipped_x0(space)),
            method=method,
            bounds=search.bounds if method != "Nelder-Mead" else None,
            callback=lambda u: _callback(search.to_params(u)),
            options=opts,
        )
        result.x = search.to_params(result.x)

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

    model_params = space.model_params(result.x, model.param_names)
    return FitResult(
        params=space.resolve(result.x),
        model_params=model_params,
        model=model.with_params(**model_params),
        residual=float(result.fun),
        success=bool(result.success),
        n_evaluations=int(result.nfev),
        message=result.message if hasattr(result, "message") else "",
        history=history,
        n_failed=failures.n_failed,
        failure_message=failures.first_message,
        failure_params=failures.first_params,
    )
