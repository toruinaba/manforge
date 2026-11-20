"""Everything a model can be checked for, in one call.

:func:`check_model` runs the checks of this package that apply to a model and
reports which ones it ran.  What applies is read off the model itself:

==========================  ===================================================
always                      the autodiff return mapping converges over the
                            loading histories (with Δλ ≥ 0 and finite values),
                            and its consistent tangent agrees with central
                            differences
``hand_block`` declared     every declared Jacobian block agrees with autodiff
                            (:func:`check_hand_blocks`)
hand-written return         its solutions are roots of the framework residual
mapping                     (:func:`check_solution`), its trajectory agrees
                            with the autodiff one (:func:`crosscheck`), and
                            its tangent agrees with central differences
==========================  ===================================================

A model that is only a Python model — a yield function and evolution laws,
solved by autodiff — is checked for the first row and passes on it: stopping
there is a legitimate place for a model to be.  The rest is checked as soon
as the model declares it.

::

    from returnmap.verification import check_model

    report = check_model(model)
    assert report.passed, report.summary()

Loading histories
-----------------
Two standard histories are generated for any model, scaled by its own yield
strain: a uniaxial tension-unload-compression history, and a multiaxial one
that drives direct and shear components together and changes direction (a
transposed outer product or a ×2 shear-convention error is invisible under
uniaxial loading).

A model family can add histories that exercise its own mechanisms — the
second hardening branch of a two-surface model, say — by defining
``histories(model)`` in a module named ``verification`` next to the model
(``returnmap.models.yu_kinematic.verification`` for a model defined in
``returnmap.models.yu_kinematic.model``).  :func:`check_model` finds it by
that name; nothing has to be registered.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from returnmap.core.material import MaterialModel
from returnmap.simulation.driver import StrainDriver
from returnmap.simulation.integrator import (
    PythonAnalyticalIntegrator, PythonNumericalIntegrator,
)
from returnmap.simulation.types import FieldHistory, FieldType
from returnmap.verification.crosscheck import crosscheck
from returnmap.verification.jacobian import check_hand_blocks
from returnmap.verification.solution import check_solution
from returnmap.verification.tangent import check_tangent
from returnmap.verification.test_cases import estimate_yield_strain, generate_strain_history

if TYPE_CHECKING:
    from returnmap.core.result import StressUpdateResult
    from returnmap.simulation.integrator import StressIntegrator

Status = Literal["passed", "failed", "not declared"]

# Direction of the multiaxial history, as (direct, shear) components: every
# component is driven, none of them in proportion to another after the turn.
_FIRST_LEG = ((1.0, -0.4, 0.2), (0.7, 0.3, -0.5))
_SECOND_LEG = ((-0.3, 1.0, 0.1), (-0.6, 0.8, 0.4))


@dataclass
class ModelCheck:
    """One line of a :class:`ModelReport`.

    Attributes
    ----------
    name : str
        What was checked.
    status : {"passed", "failed", "not declared"}
        ``"not declared"`` means the model does not have the thing the check
        is about (no hand-written return mapping, no hand-derived block); it
        does not count against the model.
    detail : str
        The largest error and its tolerance, or the reason for the status.
    reports : dict
        The reports of the underlying checks, by loading history where there
        is one per history — ``reports["multiaxial"].summary()`` shows the
        table behind a failure.
    """

    name: str
    status: Status
    detail: str
    reports: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status != "failed"


@dataclass
class ModelReport:
    """Result of :func:`check_model`.

    Attributes
    ----------
    model : str
        Class name of the model that was checked.
    checks : list[ModelCheck]
        One entry per check, in the order they were run.
    histories : dict[str, int]
        The loading histories used, with the number of plastic steps the
        autodiff path took on each.
    """

    model: str
    checks: list[ModelCheck]
    histories: dict[str, int]

    @property
    def passed(self) -> bool:
        """No check failed.  Checks that do not apply do not count."""
        return all(check.passed for check in self.checks)

    def __getitem__(self, name: str) -> ModelCheck:
        for check in self.checks:
            if check.name == name:
                return check
        raise KeyError(f"no check named {name!r}; have {[c.name for c in self.checks]}")

    def summary(self) -> str:
        """A table: one row per check, then the histories it was run on."""
        failed = [c.name for c in self.checks if c.status == "failed"]
        head = f"{self.model}: " + (
            "every check that applies passes" if not failed
            else f"{len(failed)} check(s) failed — {', '.join(failed)}"
        )
        marks = {"passed": "ok  ", "failed": "FAIL", "not declared": "—   "}
        width = max(len(c.name) for c in self.checks)
        lines = [head] + [
            f"  {marks[c.status]} {c.name.ljust(width)}  {c.detail}" for c in self.checks
        ]
        lines.append(
            "  histories: " + ", ".join(
                f"{name} ({n} plastic steps)" for name, n in self.histories.items()
            )
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def standard_histories(model: MaterialModel) -> dict[str, FieldHistory]:
    """The loading histories every model is checked on.

    ``"uniaxial"``
        Tension, partial unloading, compression in the first component
        (:func:`generate_strain_history`).
    ``"multiaxial"``
        All components at once, direct and shear together: out along one
        direction to five yield strains, a turn to a second direction, and
        back through the origin.  Absent for a one-component model.

    Both are scaled by the model's own yield strain, so they reach
    plasticity whatever units and parameters the model has.
    """
    histories = {
        "uniaxial": FieldHistory(FieldType.STRAIN, generate_strain_history(model)),
    }
    dim = model.dimension
    if dim.ntens > 1:
        eps_y = estimate_yield_strain(model)

        def direction(leg):
            direct, shear = leg
            return np.array(direct[:dim.ndi] + shear[:dim.nshr])

        first = 5.0 * eps_y * direction(_FIRST_LEG)
        second = 5.0 * eps_y * direction(_SECOND_LEG)
        corners = [np.zeros(dim.ntens), first, second, -0.6 * first, np.zeros(dim.ntens)]
        rows = [
            a + (b - a) * t
            for a, b in zip(corners, corners[1:])
            for t in np.linspace(0.0, 1.0, 7)[1:]
        ]
        histories["multiaxial"] = FieldHistory(FieldType.STRAIN, np.array(rows))
    return histories


def _declared_histories(model: MaterialModel) -> dict[str, FieldHistory]:
    """Histories the model's family declares in its ``verification`` module."""
    package = type(model).__module__.rpartition(".")[0]
    if not package:
        return {}
    try:
        module = importlib.import_module(f"{package}.verification")
    except ModuleNotFoundError as exc:
        if exc.name == f"{package}.verification":
            return {}
        raise                       # the module exists and its own import failed
    declare: Callable | None = getattr(module, "histories", None)
    return dict(declare(model)) if declare is not None else {}


def _worst(errors: list[float]) -> float:
    """Largest error, NaN counting as the worst."""
    return float("nan") if any(np.isnan(e) for e in errors) else max(errors)


def _tangent_points(results: list["StressUpdateResult"], n: int) -> list["StressUpdateResult"]:
    """Up to ``n`` plastic steps, spread over the history, the first included."""
    plastic = [r for r in results if r.is_plastic]
    if len(plastic) <= n:
        return plastic
    picks = np.unique(np.linspace(0, len(plastic) - 1, n).round().astype(int))
    return [plastic[i] for i in picks]


def _check_tangents(
    integrator: "StressIntegrator", runs: Mapping[str, list], n_points: int, tol: float,
) -> tuple[Status, str, dict]:
    reports = {}
    for name, results in runs.items():
        for k, r in enumerate(_tangent_points(results, n_points)):
            inc = r.increment
            assert inc is not None, "a Python integrator's result carries its increment"
            state_n = dict(inc.state_n.as_dict())
            reports[f"{name}[{k}]"] = check_tangent(
                integrator, inc.strain_inc, state_n["stress"], state_n, tol=tol,
            )
    if not reports:
        return "failed", "no plastic step to check a tangent at", reports
    worst = _worst([r.max_rel_err for r in reports.values()])
    status: Status = "passed" if all(r.passed for r in reports.values()) else "failed"
    return status, f"{worst:.1e} (tol {tol:.0e}) at {len(reports)} plastic steps", reports


def check_model(
    model: MaterialModel,
    histories: Mapping[str, FieldHistory] | None = None,
    *,
    tangent_tol: float = 1e-5,
    block_tol: float = 1e-8,
    solution_tol: float = 1e-8,
    stress_tol: float = 1e-6,
    n_tangent_points: int = 4,
) -> ModelReport:
    """Run every check that applies to ``model`` and report which ones ran.

    Parameters
    ----------
    model : MaterialModel
        The model, with the parameters to check it at
        (:meth:`~returnmap.core.MaterialModel.at_defaults` gives the
        documented typical ones).
    histories : mapping of str to FieldHistory, or None
        Strain histories to check on.  ``None`` (default) uses
        :func:`standard_histories` plus those the model's family declares
        (see the module docstring).  Given, exactly these are used.
    tangent_tol : float, default 1e-5
        Tolerance of the tangent against central differences.
    block_tol : float, default 1e-8
        Tolerance of hand-derived blocks meant to be exact; a block declared
        with ``approx=`` is held to that value.
    solution_tol : float, default 1e-8
        Tolerance of a hand-written solution in the framework residual.
    stress_tol : float, default 1e-6
        Tolerance of the hand-written trajectory against the autodiff one,
        for stress and state variables; the tangents are compared at ten
        times this.
    n_tangent_points : int, default 4
        Plastic steps per history at which a tangent is differenced.  Each
        costs ``2 * ntens`` stress updates.

    Returns
    -------
    ModelReport
        ``report.passed``, ``report.summary()``, and ``report[name]`` for one
        check with the reports behind it.

    Notes
    -----
    What this does not check: that the formulation is the one intended.
    Every check here compares two derivations of the same model — autodiff
    against finite differences, a hand derivation against autodiff — so an
    assumption both share (a unit, a sign convention, the formulation
    itself) passes unnoticed.  The comparison with a compiled UMAT
    (:func:`crosscheck`, :func:`check_bindings`) is a separate step, because
    it needs the module built.
    """
    if not isinstance(model, MaterialModel):
        raise TypeError(
            f"check_model takes a model instance, got {model!r}; "
            "for a class, check cls.at_defaults()"
        )
    cls = type(model)
    if histories is None:
        histories = {**standard_histories(model), **_declared_histories(model)}
    if not histories:
        raise ValueError("no loading history to check the model on")
    checks: list[ModelCheck] = []

    # ---- the autodiff path: convergence over every history ----------------
    numerical = PythonNumericalIntegrator(model, raise_on_nonconverged=False)
    runs: dict[str, list] = {}
    stopped: list[str] = []
    for name, load in histories.items():
        try:
            result = StrainDriver(numerical).run(load)
        except (np.linalg.LinAlgError, FloatingPointError, ZeroDivisionError) as exc:
            runs[name] = []
            stopped.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        runs[name] = list(result.step_results)
        if not result.converged:
            stopped.append(f"{name}: stopped at step {result.failed_step}")
        elif not all(
            np.all(np.isfinite(np.asarray(r.stress, dtype=float))) for r in runs[name]
        ):
            stopped.append(f"{name}: non-finite stress")
    n_plastic = {name: sum(bool(r.is_plastic) for r in rs) for name, rs in runs.items()}
    idle = [name for name, n in n_plastic.items() if n == 0 and name not in
            {s.split(":")[0] for s in stopped}]
    if stopped:
        checks.append(ModelCheck("converges", "failed", "; ".join(stopped)))
    elif idle:
        checks.append(ModelCheck(
            "converges", "failed",
            f"no plastic step on {', '.join(idle)} — the history does not reach yield",
        ))
    else:
        n_steps = sum(len(rs) for rs in runs.values())
        checks.append(ModelCheck(
            "converges", "passed",
            f"{n_steps} steps over {len(runs)} histories, "
            f"{sum(n_plastic.values())} plastic",
        ))

    # ---- its tangent -------------------------------------------------------
    status, detail, reports = _check_tangents(numerical, runs, n_tangent_points, tangent_tol)
    checks.append(ModelCheck("tangent", status, detail, reports))

    # ---- hand-derived Jacobian blocks -------------------------------------
    if getattr(cls, "_hand_blocks", None):
        plastic = [r for rs in runs.values() for r in rs if r.is_plastic]
        if plastic:
            report = check_hand_blocks(model, plastic, tol=block_tol)
            exact = [b.worst_err for b in report.blocks.values() if not b.approx]
            worst = _worst(exact) if exact else 0.0
            n_approx = sum(bool(b.approx) for b in report.blocks.values())
            detail = (
                f"{worst:.1e} (tol {block_tol:.0e}) over {len(report.blocks)} blocks "
                f"at {report.n_points} plastic steps"
                + (f"; {n_approx} declared approximate" if n_approx else "")
            )
            checks.append(ModelCheck(
                "hand blocks", "passed" if report.passed else "failed", detail,
                {"all": report},
            ))
        else:
            checks.append(ModelCheck("hand blocks", "failed", "no plastic step to check at"))
    else:
        checks.append(ModelCheck("hand blocks", "not declared", "no @hand_block on the model"))

    # ---- a hand-written return mapping ------------------------------------
    hand_rm = cls.user_defined_return_mapping is not MaterialModel.user_defined_return_mapping
    if not hand_rm:
        reason = "no user_defined_return_mapping on the model"
        checks += [
            ModelCheck("hand-written solution", "not declared", reason),
            ModelCheck("hand-written vs autodiff", "not declared", reason),
            ModelCheck("hand-written tangent", "not declared", reason),
        ]
        return ModelReport(cls.__name__, checks, n_plastic)

    analytical = PythonAnalyticalIntegrator(model, raise_on_nonconverged=False)
    hand_runs: dict[str, list] = {}
    for name, load in histories.items():
        hand_runs[name] = list(StrainDriver(analytical).run(load).step_results)

    hand_plastic = [r for rs in hand_runs.values() for r in rs if r.is_plastic]
    if hand_plastic:
        report = check_solution(model, hand_plastic, tol=solution_tol)
        worst = _worst([row.worst_err for row in report.rows.values()])
        checks.append(ModelCheck(
            "hand-written solution", "passed" if report.passed else "failed",
            f"{worst:.1e} (tol {solution_tol:.0e}) over {len(report.rows)} rows "
            f"at {report.n_points} plastic steps",
            {"all": report},
        ))
    else:
        checks.append(ModelCheck(
            "hand-written solution", "failed", "the hand-written path took no plastic step"))

    reports = {
        name: crosscheck(
            numerical, analytical, load,
            stress_tol=stress_tol, state_tol=stress_tol, tangent_tol=10.0 * stress_tol,
        )
        for name, load in histories.items()
    }
    stress = _worst([r.max_stress_rel_err for r in reports.values()])
    state = _worst([e for r in reports.values() for e in r.max_state_rel_err.values()] or [0.0])
    tangents = [r.max_tangent_rel_err for r in reports.values()
                if r.max_tangent_rel_err is not None]
    detail = f"stress {stress:.1e}, state {state:.1e} (tol {stress_tol:.0e})"
    if tangents:
        detail += f", tangent {_worst(tangents):.1e} (tol {10.0 * stress_tol:.0e})"
    stopped_at = [
        f"{name} step {r.first_failure.index}" for name, r in reports.items()
        if r.first_failure is not None
        and not (r.first_failure.a_converged and r.first_failure.b_converged)
    ]
    if stopped_at:
        detail += f"; not converged at {', '.join(stopped_at)}"
    checks.append(ModelCheck(
        "hand-written vs autodiff",
        "passed" if all(r.passed for r in reports.values()) else "failed",
        detail + f" over {len(reports)} histories",
        reports,
    ))

    if cls.user_defined_tangent is not MaterialModel.user_defined_tangent:
        status, detail, reports = _check_tangents(
            analytical, hand_runs, n_tangent_points, tangent_tol)
        checks.append(ModelCheck("hand-written tangent", status, detail, reports))
    else:
        checks.append(ModelCheck(
            "hand-written tangent", "not declared",
            "no user_defined_tangent on the model (the tangent is the autodiff one)",
        ))
    return ModelReport(cls.__name__, checks, n_plastic)
