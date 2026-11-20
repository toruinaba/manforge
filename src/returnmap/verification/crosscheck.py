"""Run two integrators through the same loading history and compare every step.

:func:`crosscheck` is the trajectory-level check between two implementations
of one model: the autodiff return mapping against a closed form, or the Python
model against its Fortran UMAT.  Each integrator is driven independently
through the same history — each carries its own state forward — and stress,
state variables and consistent tangent are compared at every step.

::

    from returnmap.simulation import (
        FortranIntegrator, FortranModule, PythonIntegrator,
    )
    from returnmap.verification import crosscheck

    py = PythonIntegrator(model)
    umat = FortranIntegrator.from_model(FortranModule("my_model"), "my_model_core", model)

    report = crosscheck(py, umat, load)
    assert report.passed, report.summary()

The driver is chosen from the load (strain history → ``StrainDriver``, stress
history → ``StressDriver``); pass ``driver_factory`` for anything else, such
as uniaxial stress on a 3D model::

    report = crosscheck(
        py, umat, axial_strain,
        driver_factory=lambda i: MixedDriver(i, prescribed_strain_idx=[0]),
    )

When a run disagrees, ``report.first_failure`` is the first step that does,
with both raw results (``.result_a`` / ``.result_b``) for inspection — e.g.
``compare_jacobians(model, step.result_a, step.result_b)``.

Two ways of marching
--------------------
By default the two trajectories are **independent**: after the first
disagreement every later step differs as well, and over a long history even
two faithful implementations drift apart by accumulated round-off.

``lockstep=True`` removes both effects.  Only the reference is driven through
the history; at every step the candidate is given the reference's own inputs
(stress and state at the start of the step, and the strain increment the
reference took) and its output is compared.  Each step is then a separate
question — *from the same state, with the same increment, do the two agree?*
— which is exactly what a UMAT port has to satisfy.  A step that disagrees is
wrong in itself, not an echo of an earlier one.

To judge each step of a hand-written Python return mapping against the
framework residual instead of against another implementation, use
:func:`~returnmap.verification.check_solution`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from returnmap.simulation.driver import default_driver_for
from returnmap.simulation.types import FieldHistory
from returnmap.verification._measure import NOISE_FLOOR, max_abs, rel_err

if TYPE_CHECKING:
    from returnmap.core.result import StressUpdateResult
    from returnmap.simulation.driver import DriverBase
    from returnmap.simulation.integrator import StressIntegrator



@dataclass
class CrosscheckStep:
    """One step of a :func:`crosscheck` run.

    Attributes
    ----------
    index : int
        0-based step index.
    result_a, result_b : StressUpdateResult
        The raw results of the two integrators.
    stress_rel_err : float
        ``max|σ_b − σ_a|`` relative to the largest stress component the
        reference has reached so far in the run.
    state_rel_err : dict[str, float]
        The same measure per state variable both results carry.
    tangent_rel_err : float or None
        ``max|D_b − D_a| / max|D_a|``; ``None`` if either result has no
        tangent.
    a_converged, b_converged : bool
        Whether each integrator (and its driver's outer NR) converged.
    passed : bool
        Both converged and every error is within its tolerance.
    """

    index: int
    result_a: "StressUpdateResult"
    result_b: "StressUpdateResult"
    stress_rel_err: float
    state_rel_err: dict[str, float]
    tangent_rel_err: float | None
    a_converged: bool
    b_converged: bool
    passed: bool


@dataclass
class CrosscheckReport:
    """Result of :func:`crosscheck`.

    Attributes
    ----------
    steps : list[CrosscheckStep]
        Every compared step, in order.
    stress_tol, state_tol, tangent_tol : float
        The tolerances the steps were judged against.
    """

    steps: list[CrosscheckStep]
    stress_tol: float
    state_tol: float
    tangent_tol: float

    @property
    def passed(self) -> bool:
        """At least one step was compared and every step agrees."""
        return bool(self.steps) and all(s.passed for s in self.steps)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def n_passed(self) -> int:
        return sum(s.passed for s in self.steps)

    @property
    def first_failure(self) -> CrosscheckStep | None:
        """The first step that disagrees, or ``None``."""
        return next((s for s in self.steps if not s.passed), None)

    @property
    def max_stress_rel_err(self) -> float:
        return max((s.stress_rel_err for s in self.steps), default=0.0)

    @property
    def max_state_rel_err(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for s in self.steps:
            for k, v in s.state_rel_err.items():
                out[k] = max(out.get(k, 0.0), v)
        return out

    @property
    def max_tangent_rel_err(self) -> float | None:
        errs = [s.tangent_rel_err for s in self.steps if s.tangent_rel_err is not None]
        return max(errs) if errs else None

    @property
    def n_a_nonconverged(self) -> int:
        return sum(not s.a_converged for s in self.steps)

    @property
    def n_b_nonconverged(self) -> int:
        return sum(not s.b_converged for s in self.steps)

    def summary(self) -> str:
        """Multi-line text: the verdict, then the largest error of each quantity."""
        if not self.steps:
            return "crosscheck: no step was compared"
        if self.passed:
            head = f"crosscheck: {self.n_steps} steps — all agree"
        else:
            first = self.first_failure
            assert first is not None
            head = (
                f"crosscheck: {self.n_steps - self.n_passed} of {self.n_steps} "
                f"steps disagree (first at step {first.index})"
            )
        lines = [head]

        def line(label: str, err: float, tol: float) -> str:
            mark = "ok  " if err <= tol else "FAIL"
            return f"  {mark} {label}: {err:.3e} (tol {tol:.0e})"

        lines.append(line("stress", self.max_stress_rel_err, self.stress_tol))
        for k, v in self.max_state_rel_err.items():
            lines.append(line(f"state[{k}]", v, self.state_tol))
        t = self.max_tangent_rel_err
        if t is not None:
            lines.append(line("tangent", t, self.tangent_tol))
        if self.n_a_nonconverged or self.n_b_nonconverged:
            lines.append(
                f"  not converged: a at {self.n_a_nonconverged} steps, "
                f"b at {self.n_b_nonconverged} steps"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def crosscheck(
    integrator_a: "StressIntegrator",
    integrator_b: "StressIntegrator",
    load: FieldHistory,
    *,
    driver_factory: "Callable[[StressIntegrator], DriverBase] | None" = None,
    stress_tol: float = 1e-6,
    state_tol: float = 1e-6,
    tangent_tol: float = 1e-5,
    stop_on_failure: bool = False,
    lockstep: bool = False,
    **run_kwargs,
) -> CrosscheckReport:
    """Drive two integrators through ``load`` and compare them step by step.

    Parameters
    ----------
    integrator_a : StressIntegrator
        The reference — typically a Python integrator.  Errors are measured
        relative to its values.
    integrator_b : StressIntegrator
        The candidate — another Python integrator, or a
        :class:`~returnmap.simulation.FortranIntegrator`.
    load : FieldHistory
        The loading history, as the driver expects it.
    driver_factory : callable or None
        ``driver_factory(integrator) -> driver``, called once per integrator.
        ``None`` (default) picks :class:`~returnmap.simulation.StrainDriver`
        for a strain history and :class:`~returnmap.simulation.StressDriver`
        for a stress history.  Pass a factory for
        :class:`~returnmap.simulation.MixedDriver` or to set a driver's own
        outer-NR tolerances.
    stress_tol, state_tol, tangent_tol : float
        Pass thresholds on the relative errors (defaults 1e-6, 1e-6, 1e-5).
    stop_on_failure : bool
        Stop at the first step that disagrees instead of running the whole
        history; ``report.first_failure`` is then the last step.
    lockstep : bool
        ``False`` (default): both integrators are driven through the history
        independently, each carrying its own state.  ``True``: only
        ``integrator_a`` is driven; at each step ``integrator_b`` receives the
        same stress, state and strain increment and its single update is
        compared.  See the module docstring.
    **run_kwargs
        Forwarded to each driver's ``iter_run`` — ``initial_stress``,
        ``initial_state``, and for ``MixedDriver`` ``prescribed_stress_history``.
        A one-row load with ``initial_stress`` / ``initial_state`` compares a
        single step from a given state.

    Returns
    -------
    CrosscheckReport
        ``report.passed``, ``report.summary()``, and the per-step records.

    Notes
    -----
    How the errors are measured.  Stress and each state variable are compared
    as whole arrays, ``max|b − a|`` relative to the largest magnitude the
    reference has reached so far in the run — not component by component and
    not against the current value, so a stress passing through zero on
    unloading, or a backstress still near zero, does not turn round-off into a
    large relative error.  A state variable that stays twelve orders below the
    largest quantity of the run is treated as zero.  The tangent is compared
    against the largest entry of the reference tangent of the same step.

    Only the state variables both results carry are compared (a Fortran UMAT
    result has no ``"stress"`` entry in its state; stress is compared on its
    own).
    """
    factory = driver_factory if driver_factory is not None else default_driver_for(load)
    driver_a = factory(integrator_a)

    def independent(kwargs: dict) -> dict:
        """Each driver gets its own arrays: neither may see the other's updates."""
        out = dict(kwargs)
        if out.get("initial_state") is not None:
            out["initial_state"] = {k: np.array(v) for k, v in out["initial_state"].items()}
        if out.get("initial_stress") is not None:
            out["initial_stress"] = np.array(out["initial_stress"], dtype=float)
        return out

    def paired():
        """Yield (index, result_a, a_converged, result_b, b_converged) per step."""
        if not lockstep:
            for sa, sb in zip(
                driver_a.iter_run(load, **independent(run_kwargs)),
                factory(integrator_b).iter_run(load, **independent(run_kwargs)),
            ):
                yield sa.index, sa.result, bool(sa.converged), sb.result, bool(sb.converged)
            return
        # The inputs of each step are the reference's: the state it started
        # the step from and the strain increment its driver took.
        start = independent(run_kwargs)
        stress_n = start.get("initial_stress")
        if stress_n is None:
            stress_n = np.zeros(integrator_a.ntens)
        state_n = start.get("initial_state")
        if state_n is None:
            state_n = integrator_a.initial_state()
        strain_n = np.zeros(integrator_a.ntens)
        for sa in driver_a.iter_run(load, **independent(run_kwargs)):
            strain = np.asarray(sa.strain, dtype=float)
            # A Python result carries the increment it was solved for; a
            # Fortran one does not, and the cumulative strain is differenced.
            inc = sa.result.increment
            strain_inc = inc.strain_inc if inc is not None else strain - strain_n
            rb = integrator_b.stress_update(strain_inc, stress_n, state_n)
            yield sa.index, sa.result, bool(sa.converged), rb, bool(rb.converged)
            strain_n, stress_n, state_n = strain, sa.result.stress, sa.result.state

    steps: list[CrosscheckStep] = []
    stress_scale = 0.0
    state_scale: dict[str, float] = {}
    run_scale = 0.0

    for index, ra, a_ok, rb, b_ok in paired():
        stress_a = np.asarray(ra.stress, dtype=float)
        stress_b = np.asarray(rb.stress, dtype=float)
        stress_scale = float(np.fmax(stress_scale, np.max(np.abs(stress_a))))
        run_scale = max(run_scale, stress_scale)

        keys = [k for k in ra.state if k != "stress" and k in rb.state]
        values = {k: (np.asarray(ra.state[k], dtype=float),
                      np.asarray(rb.state[k], dtype=float)) for k in keys}
        for k, (va, _) in values.items():
            state_scale[k] = float(np.fmax(state_scale.get(k, 0.0), np.max(np.abs(va))))
            run_scale = max(run_scale, state_scale[k])

        s_err = rel_err(max_abs(stress_b - stress_a), stress_scale)
        st_err = {
            k: rel_err(max_abs(vb - va), max(state_scale[k], NOISE_FLOOR * run_scale))
            for k, (va, vb) in values.items()
        }
        t_err: float | None = None
        if ra.ddsdde is not None and rb.ddsdde is not None:
            da = np.asarray(ra.ddsdde, dtype=float)
            db = np.asarray(rb.ddsdde, dtype=float)
            t_err = rel_err(max_abs(db - da), max_abs(da))

        passed = (
            a_ok and b_ok
            and s_err <= stress_tol
            and all(v <= state_tol for v in st_err.values())
            and (t_err is None or t_err <= tangent_tol)
        )
        steps.append(CrosscheckStep(
            index=index, result_a=ra, result_b=rb,
            stress_rel_err=s_err, state_rel_err=st_err, tangent_rel_err=t_err,
            a_converged=a_ok, b_converged=b_ok, passed=passed,
        ))
        if stop_on_failure and not passed:
            break

    return CrosscheckReport(
        steps=steps, stress_tol=stress_tol, state_tol=state_tol, tangent_tol=tangent_tol,
    )


__all__ = ["crosscheck", "CrosscheckReport", "CrosscheckStep"]
