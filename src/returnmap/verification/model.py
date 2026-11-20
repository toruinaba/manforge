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
a UMAT declared and built   the compiled UMAT agrees with the Python model
                            step by step over the same histories
                            (:func:`crosscheck`)
==========================  ===================================================

A model that is only a Python model — a yield function and evolution laws,
solved by autodiff — is checked for the first row and passes on it: stopping
there is a legitimate place for a model to be.  The rest is checked as soon
as the model declares it.  A UMAT is declared in the family's
``verification.py`` (:class:`Umat`) and compared once its module has been
built (``returnmap build --family <package>``); until then its row reads
"not built", which does not count against the model either.

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
that name; nothing has to be registered.  A history declared under one of the
standard names replaces that one — for a model the standard scaling does not
suit (:func:`standard_histories` takes the strain to scale by).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from returnmap.core.material import MaterialModel
from returnmap.core.material.hand_block import (
    HandTangent, condition_holds, hand_block_record, hand_tangent_record,
)
from returnmap.simulation.driver import StrainDriver
from returnmap.simulation.integrator import (
    PythonAnalyticalIntegrator, PythonIntegrator, PythonNumericalIntegrator,
)
from returnmap.simulation.types import FieldHistory, FieldType
from returnmap.verification.crosscheck import crosscheck
from returnmap.verification.jacobian import check_hand_blocks
from returnmap.verification.solution import check_solution
from returnmap.verification.tangent import check_tangent
from returnmap.verification.test_cases import estimate_yield_strain, generate_strain_history
from returnmap.verification.umat import umat_of

if TYPE_CHECKING:
    from returnmap.core.result import StressUpdateResult
    from returnmap.simulation.integrator import StressIntegrator

Status = Literal["passed", "failed", "not declared", "not built"]

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
    status : {"passed", "failed", "not declared", "not built"}
        ``"not declared"`` means the model does not have the thing the check
        is about (no hand-written return mapping, no hand-derived block, no
        UMAT); ``"not built"`` that its UMAT is declared but the compiled
        module is not there to compare.  Neither counts against the model.
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
    approximations : list[str]
        The reasons the model gives for the approximations it declares
        (``hand_block(..., note=)``, ``hand_tangent(..., note=)``) — listed
        whether or not a step of these histories fell under them.
    """

    model: str
    checks: list[ModelCheck]
    histories: dict[str, int]
    approximations: list[str] = field(default_factory=list)

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
        marks = {"passed": "ok  ", "failed": "FAIL", "not declared": "—   ", "not built": "—   "}
        width = max(len(c.name) for c in self.checks)
        lines = [head] + [
            f"  {marks[c.status]} {c.name.ljust(width)}  {c.detail}" for c in self.checks
        ]
        lines.append(
            "  histories: " + ", ".join(
                f"{name} ({n} plastic steps)" for name, n in self.histories.items()
            )
        )
        lines += [f"  declared approximate: {note}" for note in self.approximations]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def standard_histories(
    model: MaterialModel, eps_y: float | None = None,
) -> dict[str, FieldHistory]:
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

    Parameters
    ----------
    model : MaterialModel
    eps_y : float or None
        The strain to scale the histories by.  ``None`` (default): the
        model's yield strain (:func:`estimate_yield_strain`).  Give it for a
        model that has no yield strain to find — one without an elastic
        domain is plastic from the first increment — and declare the result
        under the same names in the family's ``verification.py``, which
        replaces the standard ones::

            def histories(model):
                return standard_histories(model, eps_y=model.F0 / model.E)
    """
    if eps_y is None:
        eps_y = estimate_yield_strain(model)
    histories = {
        "uniaxial": FieldHistory(FieldType.STRAIN, generate_strain_history(model, eps_y=eps_y)),
    }
    dim = model.dimension
    if dim.ntens > 1:

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


def _declared_approximate(
    model: MaterialModel, declared: HandTangent | None, result: "StressUpdateResult",
) -> bool:
    """Whether the model declares its hand-written tangent approximate at this step."""
    if declared is None or declared.approx_when is None:
        return False
    return condition_holds(
        model, declared.approx_when,
        state=result.state, dlambda=result.dlambda, inc=result.increment,
    )


def _approx_note(n: int, declared: HandTangent | None) -> str:
    return f"; {n} step(s) declared approximate" if n and declared else ""


def _check_tangents(
    integrator: "StressIntegrator", runs: Mapping[str, list], n_points: int, tol: float,
    model: MaterialModel | None = None, declared: HandTangent | None = None,
) -> tuple[Status, str, dict]:
    """Difference the tangent at a few plastic steps of each history.

    ``declared`` is the model's ``hand_tangent`` declaration when the tangent
    being checked is the hand-written one: an ``approx`` without a condition
    is the tolerance at every step; with ``approx_when``, the steps the
    condition holds on are held to ``approx`` — or left out, if none is given
    — and the rest to ``tol``.
    """
    exact_tol = tol
    if declared is not None and declared.approx_when is None and declared.approx is not None:
        exact_tol = declared.approx             # approximate at every step
    reports = {}
    n_approx = 0
    ok = True
    for name, results in runs.items():
        for k, r in enumerate(_tangent_points(results, n_points)):
            inc = r.increment
            assert inc is not None, "a Python integrator's result carries its increment"
            state_n = dict(inc.state_n.as_dict())
            approximate = model is not None and _declared_approximate(model, declared, r)
            step_tol = exact_tol
            if approximate:
                n_approx += 1
                assert declared is not None
                if declared.approx is None:
                    continue                    # left out of the judgement
                step_tol = declared.approx
            report = check_tangent(
                integrator, inc.strain_inc, state_n["stress"], state_n, tol=step_tol,
            )
            reports[f"{name}[{k}]"] = report
            ok = ok and report.passed
    if not reports:
        if n_approx:
            return "failed", (
                f"all {n_approx} sampled steps are declared approximate — "
                "the tangent was held to nothing"
            ), reports
        return "failed", "no plastic step to check a tangent at", reports
    exact = [r.max_rel_err for r in reports.values() if r.tol == exact_tol]
    worst = _worst(exact) if exact else 0.0
    detail = (
        f"{worst:.1e} (tol {exact_tol:.0e}) at {len(exact)} plastic steps"
        + _approx_note(n_approx, declared)
    )
    return ("passed" if ok else "failed"), detail, reports


def _check_umat(
    model: MaterialModel, histories: Mapping[str, FieldHistory], stress_tol: float,
) -> ModelCheck:
    """The model's UMAT against the Python model, over the same histories."""
    umat = umat_of(model)
    package = type(model).__module__.rpartition(".")[0]
    if umat is None:
        return ModelCheck(
            "UMAT", "not declared",
            f"no UMAT declared for {type(model).__name__}"
            + (f" in {package}.verification" if package else ""),
        )
    if not umat.is_built():
        return ModelCheck(
            "UMAT", "not built",
            f"module {umat.module} is not built — returnmap build --family {package}",
        )
    python = PythonIntegrator(model, raise_on_nonconverged=False)
    fortran = umat.integrator(model, raise_on_nonconverged=False)
    reports = {
        name: crosscheck(
            python, fortran, load,
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
    return ModelCheck(
        "UMAT", "passed" if all(r.passed for r in reports.values()) else "failed",
        detail + f" over {len(reports)} histories ({umat.module})", reports,
    )


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
    against finite differences, a hand derivation against autodiff, the UMAT
    against the Python model — so an assumption all of them share (a unit, a
    sign convention, the formulation itself) passes unnoticed.

    The UMAT is compared with ``PythonIntegrator(model)``: the hand-written
    return mapping when the model has one — that is what the UMAT is a port
    of — and the autodiff one otherwise.  The individual subroutines a model
    binds with ``verified_against_fortran`` need hand-written inputs and are
    checked separately (:func:`check_bindings`).
    """
    if not isinstance(model, MaterialModel):
        raise TypeError(
            f"check_model takes a model instance, got {model!r}; "
            "for a class, check cls.at_defaults()"
        )
    cls = type(model)
    notes = [
        hand_block_record(cls, name).note for name in getattr(cls, "_hand_blocks", {}).values()
    ]
    tangent_record = hand_tangent_record(cls)
    if tangent_record is not None:
        notes.append(tangent_record.note)
    approximations = list(dict.fromkeys(note for note in notes if note))
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
            exact = [b.worst_err for b in report.blocks.values() if b.tol == block_tol]
            worst = _worst(exact) if exact else 0.0
            everywhere = sum(b.approx and b.tol != block_tol for b in report.blocks.values())
            on_steps = [b for b in report.blocks.values() if b.n_approx_points]
            detail = (
                f"{worst:.1e} (tol {block_tol:.0e}) over {len(report.blocks)} blocks "
                f"at {report.n_points} plastic steps"
                + (f"; {everywhere} declared approximate" if everywhere else "")
                + (f"; {len(on_steps)} declared approximate on "
                   f"{max(b.n_approx_points for b in on_steps)} step(s)" if on_steps else "")
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
        checks.append(_check_umat(model, histories, stress_tol))
        return ModelReport(cls.__name__, checks, n_plastic, approximations)

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

    # The hand-written tangent may be declared an approximation (hand_tangent):
    # everywhere, to a tolerance; or on the steps a condition names.
    declared = hand_tangent_record(cls)
    cc_tangent_tol = 10.0 * stress_tol
    if declared is not None and declared.approx_when is None and declared.approx is not None:
        cc_tangent_tol = declared.approx
    reports = {
        name: crosscheck(
            numerical, analytical, load,
            stress_tol=stress_tol, state_tol=stress_tol, tangent_tol=cc_tangent_tol,
        )
        for name, load in histories.items()
    }

    def only_the_declared_tangent_is_off(step) -> bool:
        """A step that disagrees in nothing but a tangent declared approximate there."""
        if not (step.a_converged and step.b_converged):
            return False
        if step.stress_rel_err > stress_tol or any(
            e > stress_tol for e in step.state_rel_err.values()
        ):
            return False
        if not _declared_approximate(model, declared, step.result_b):
            return False
        assert declared is not None
        return declared.approx is None or step.tangent_rel_err <= declared.approx

    excused = {
        name: [s for s in r.steps if not s.passed and only_the_declared_tangent_is_off(s)]
        for name, r in reports.items()
    }
    n_excused = sum(len(steps) for steps in excused.values())
    all_agree = all(
        r.n_steps > 0 and r.n_steps - r.n_passed == len(excused[name])
        for name, r in reports.items()
    )
    stress = _worst([r.max_stress_rel_err for r in reports.values()])
    state = _worst([e for r in reports.values() for e in r.max_state_rel_err.values()] or [0.0])
    tangents = [
        s.tangent_rel_err for name, r in reports.items() for s in r.steps
        if s.tangent_rel_err is not None and s not in excused[name]
    ]
    detail = f"stress {stress:.1e}, state {state:.1e} (tol {stress_tol:.0e})"
    if tangents:
        detail += f", tangent {_worst(tangents):.1e} (tol {cc_tangent_tol:.0e})"
    stopped_at = [
        f"{name} step {r.first_failure.index}" for name, r in reports.items()
        if r.first_failure is not None
        and not (r.first_failure.a_converged and r.first_failure.b_converged)
    ]
    if stopped_at:
        detail += f"; not converged at {', '.join(stopped_at)}"
    checks.append(ModelCheck(
        "hand-written vs autodiff",
        "passed" if all_agree else "failed",
        detail + f" over {len(reports)} histories" + _approx_note(n_excused, declared),
        reports,
    ))

    if cls.user_defined_tangent is not MaterialModel.user_defined_tangent:
        status, detail, reports = _check_tangents(
            analytical, hand_runs, n_tangent_points, tangent_tol, model, declared)
        checks.append(ModelCheck("hand-written tangent", status, detail, reports))
    else:
        checks.append(ModelCheck(
            "hand-written tangent", "not declared",
            "no user_defined_tangent on the model (the tangent is the autodiff one)",
        ))
    checks.append(_check_umat(model, histories, stress_tol))
    return ModelReport(cls.__name__, checks, n_plastic, approximations)
