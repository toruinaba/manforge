"""Run several models through one loading history and set them side by side.

:func:`compare_models` answers the questions that come up when choosing
between constitutive models, each only as far as the caller asks:

* **How do the responses differ?**  Every run keeps its
  :class:`~returnmap.simulation.DriverResult`, so stress, strain and any state
  history can be plotted on shared axes.
* **What does each cost?**  Newton iterations of the return mapping and
  wall-clock time, per step.
* **How well does each fit?**  When ``targets`` are given, the value the
  fitting objective takes for each model.

::

    from returnmap.simulation import FieldHistory, compare_models

    load = FieldHistory.cyclic_strain([0.02, -0.02, 0.02], n_per_segment=50)
    comparison = compare_models({"j2": j2, "af": af, "ow": ow}, load)

    print(comparison.summary())
    for name, run in comparison.runs.items():
        plt.plot(run.strain[:, 0], run.stress[:, 0], label=name)

A model that stops converging part-way is not an error here: its row reports
the step it stopped at and its histories hold the steps it completed, so the
other models are still compared.

Nothing is fitted.  To compare fitted models, fit each with
:func:`~returnmap.fitting.fit_params` and pass the ``FitResult.model`` of each.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from returnmap.core.material import MaterialModel
from returnmap.simulation.driver import default_driver_for
from returnmap.simulation.integrator import PythonIntegrator
from returnmap.simulation.types import DriverResult, FieldHistory
from returnmap._typing import NRJacobian

if TYPE_CHECKING:
    from returnmap.fitting.target import FitTarget
    from returnmap.simulation.driver import DriverBase
    from returnmap.simulation.integrator import StressIntegrator

# What "these parameters cannot be integrated here" looks like when it is
# raised rather than reported: non-convergence from an integrator left at its
# default raise_on_nonconverged=True, a singular NR Jacobian, a numerical
# blow-up.  The same set the fitting objective and run_simulation treat as a
# failed evaluation; anything else (a load of the wrong shape, an unknown
# field name) is a mistake in the call and propagates.
_UNSOLVABLE = (
    RuntimeError, FloatingPointError, ZeroDivisionError, np.linalg.LinAlgError,
)


@dataclass
class ModelRun:
    """One model's run in a :func:`compare_models` comparison.

    Attributes
    ----------
    name : str
        The key the model was passed under.
    result : DriverResult
        The completed steps.  ``stress``, ``strain`` and ``history(name)`` are
        also available directly on the run.
    elapsed : float
        Wall-clock seconds the run took, measured once.  It is what this
        driver needed: a strain-controlled run never evaluates the consistent
        tangent, a stress-controlled or mixed one does at every outer
        iteration.  Good for telling models apart by a factor, not by a few
        per cent.
    n_outer_iterations : int
        Stress updates over the completed steps — one per step under
        :class:`~returnmap.simulation.StrainDriver`, the outer Newton
        iterations under :class:`~returnmap.simulation.StressDriver` and
        :class:`~returnmap.simulation.MixedDriver`.
    error : str
        Why the run stopped, when it stopped on an exception rather than on a
        step reported as not converged.  Empty otherwise.
    fit_residual : float or None
        Sum of the residuals of ``targets`` for this model — the value the
        fitting objective takes, so it equals ``FitResult.residual`` for a
        model that was fitted to the same targets.  ``None`` when no targets
        were given or a target's run could not be completed.
    fit_error : str
        Why ``fit_residual`` could not be evaluated.  Empty otherwise.
    """

    name: str
    result: DriverResult
    elapsed: float
    n_outer_iterations: int
    error: str = ""
    fit_residual: float | None = None
    fit_error: str = ""

    @property
    def converged(self) -> bool:
        """Whether every step of the history was completed."""
        return self.result.converged

    @property
    def failed_step(self) -> int | None:
        """0-based index of the step the run stopped at, or ``None``."""
        return self.result.failed_step

    @property
    def n_steps(self) -> int:
        """Number of completed steps."""
        return len(self.result.step_results)

    @property
    def stress(self) -> np.ndarray:
        """Stress history of the completed steps, shape ``(n_steps, ntens)``."""
        return self.result.stress

    @property
    def strain(self) -> np.ndarray:
        """Strain history of the completed steps, shape ``(n_steps, ntens)``."""
        return self.result.strain

    def history(self, name: str) -> np.ndarray:
        """History of one state variable — see :meth:`DriverResult.history`."""
        return self.result.history(name)

    @property
    def n_plastic_steps(self) -> int | None:
        """Completed steps that were plastic.

        ``None`` when the integrator does not say which steps were (a Fortran
        UMAT reports no plasticity flag).
        """
        flags = [r.is_plastic for r in self.result.step_results]
        if any(flag is None for flag in flags):
            return None
        return sum(flags)

    @property
    def n_iterations(self) -> int | None:
        """Newton updates of the return mapping, summed over the completed steps.

        Counts the accepted stress update of each step, so under a stress or
        mixed driver the updates of earlier outer iterations are not in it.
        ``None`` when the integrator does not report them (a Fortran UMAT).
        """
        if self.n_plastic_steps is None:
            return None
        return sum(r.n_iterations for r in self.result.step_results)

    @property
    def iterations_per_plastic_step(self) -> float | None:
        """Mean Newton updates per plastic step.

        Elastic steps take none, so averaging over them would make a model
        look cheaper merely for yielding later.  ``None`` when there is no
        plastic step or the integrator does not report iterations.
        """
        n_plastic, n_iter = self.n_plastic_steps, self.n_iterations
        if not n_plastic or n_iter is None:
            return None
        return n_iter / n_plastic

    @property
    def time_per_step(self) -> float | None:
        """``elapsed`` per attempted step, in seconds (``None`` if none was)."""
        attempted = self.n_steps + (0 if self.converged else 1)
        return self.elapsed / attempted if attempted else None


@dataclass
class ModelComparisonResult:
    """Result of :func:`compare_models`.

    Attributes
    ----------
    runs : dict[str, ModelRun]
        One run per model, under the names and in the order they were passed.
    n_load_steps : int
        Number of steps in the loading history every model was given.
    """

    runs: dict[str, ModelRun]
    n_load_steps: int

    @property
    def converged(self) -> bool:
        """Whether every model completed the history."""
        return all(run.converged for run in self.runs.values())

    def summary(self) -> str:
        """A table with one row per model: steps, cost, and fit if evaluated."""
        runs = list(self.runs.values())
        show_outer = any(r.n_outer_iterations != r.n_steps for r in runs)
        show_fit = any(r.fit_residual is not None or r.fit_error for r in runs)

        def num(value: float | None, fmt: str) -> str:
            return "—" if value is None else format(value, fmt)

        header = ["model", "steps", "plastic", "NR it/plastic"]
        if show_outer:
            header.append("outer it/step")
        header.append("ms/step")
        if show_fit:
            header.append("fit residual")
        header.append("converged")

        rows = [header]
        for r in runs:
            row = [
                r.name,
                f"{r.n_steps}/{self.n_load_steps}",
                num(r.n_plastic_steps, "d"),
                num(r.iterations_per_plastic_step, ".2f"),
            ]
            if show_outer:
                row.append(
                    num(r.n_outer_iterations / r.n_steps if r.n_steps else None, ".2f")
                )
            per_step = r.time_per_step
            row.append(num(None if per_step is None else 1e3 * per_step, ".3f"))
            if show_fit:
                row.append(num(r.fit_residual, ".4g"))
            row.append("yes" if r.converged else f"no — stopped at step {r.failed_step}")
            rows.append(row)

        widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
        last = len(header) - 1
        lines = [
            "  ".join(
                # Names and the verdict read left to right; numbers line up right.
                cell.ljust(widths[i]) if i in (0, last) else cell.rjust(widths[i])
                for i, cell in enumerate(row)
            ).rstrip()
            for row in rows
        ]
        notes = [f"  {r.name}: {r.error}" for r in runs if r.error]
        notes += [f"  {r.name}: fit not evaluated — {r.fit_error}" for r in runs if r.fit_error]
        return "\n".join(lines + notes)

    def __str__(self) -> str:
        return self.summary()


def compare_models(
    models: "Mapping[str, MaterialModel | StressIntegrator]",
    load: FieldHistory,
    *,
    targets: "Sequence[FitTarget] | None" = None,
    driver_factory: "Callable[[StressIntegrator], DriverBase] | None" = None,
    nr_jacobian: NRJacobian = "exact",
) -> ModelComparisonResult:
    """Run every model through ``load`` and collect response, cost and fit.

    Parameters
    ----------
    models : mapping of str to MaterialModel or StressIntegrator
        The models to compare, under the names to report them by.  A model is
        integrated with :class:`~returnmap.simulation.PythonIntegrator`.  Pass
        an integrator instead to choose how a model is solved — the autodiff
        path against a hand-written one, ``"exact"`` against ``"broyden"``, or
        a Fortran UMAT against its Python model.
    load : FieldHistory
        The loading history, given to every model.
    targets : sequence of FitTarget or None
        Experimental curves to score each model against.  Each target is run
        with its own load, as in :func:`~returnmap.fitting.fit_params`, and the
        residuals are summed into ``ModelRun.fit_residual``.  ``None``
        (default) skips this.
    driver_factory : callable or None
        ``driver_factory(integrator) -> driver``, called once per model.
        ``None`` (default) picks :class:`~returnmap.simulation.StrainDriver`
        for a strain history and :class:`~returnmap.simulation.StressDriver`
        for a stress history — the rule ``fit_params`` and ``crosscheck``
        use.  The factory sees the integrator, so models of different
        dimensions can share one axial history::

            def uniaxial(integrator):
                if integrator.ntens == 1:
                    return StrainDriver(integrator)
                return MixedDriver(integrator, prescribed_strain_idx=[0])

        Targets that carry no ``driver_factory`` of their own use this one
        too.
    nr_jacobian : {"exact", "broyden"}
        Forwarded to the ``PythonIntegrator`` built for each bare model.  An
        integrator passed in ``models`` keeps its own setting.

    Returns
    -------
    ModelComparisonResult
        ``comparison.runs[name]`` for each model, and ``comparison.summary()``.

    Raises
    ------
    TypeError
        ``models`` is not a mapping, or a value is neither a model nor an
        integrator.
    ValueError
        ``models`` is empty.

    Notes
    -----
    A run that cannot be completed is reported, not raised: the run's
    ``converged`` is ``False``, ``failed_step`` names the step, and its
    histories hold the steps before it.  This covers a step that does not
    converge, and — for an integrator passed with its default
    ``raise_on_nonconverged=True``, or a singular Jacobian — the exception
    that would otherwise end the comparison (kept in ``ModelRun.error``).
    Mistakes in the call still raise: a load whose shape does not match a
    model, an unknown field name in a target.
    """
    if not isinstance(models, Mapping):
        raise TypeError(
            "models must be a mapping of name to model, e.g. "
            f"{{'j2': j2_model, 'af': af_model}}; got {type(models).__name__}"
        )
    if not models:
        raise ValueError("no models to compare")
    if targets is not None:
        # Imported here: fitting is built on simulation, not the reverse.
        from returnmap.fitting.target import check_targets

        targets = check_targets(targets)

    factory = driver_factory if driver_factory is not None else default_driver_for(load)
    runs: dict[str, ModelRun] = {}
    for name, entry in models.items():
        if isinstance(entry, type):
            raise TypeError(
                f"models[{name!r}] is the class {entry.__name__}, not a model; "
                "construct it with its parameters"
            )
        integrator = (
            PythonIntegrator(entry, nr_jacobian=nr_jacobian, raise_on_nonconverged=False)
            if isinstance(entry, MaterialModel)
            else entry
        )
        driver = factory(integrator)  # TypeError if it is not an integrator

        step_results: list = []
        strain_rows: list[np.ndarray] = []
        n_outer = 0
        failed_step: int | None = None
        error = ""
        start = time.perf_counter()
        try:
            for step in driver.iter_run(load):
                if not step.converged:
                    # Its result is a non-converged iterate, not a solution.
                    failed_step = step.index
                    break
                step_results.append(step.result)
                strain_rows.append(step.strain)
                n_outer += step.n_outer_iter
        except _UNSOLVABLE as exc:
            failed_step = len(step_results)
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - start

        run = ModelRun(
            name=name,
            result=DriverResult(
                step_results=step_results,
                strain=(
                    np.stack(strain_rows) if strain_rows
                    else np.zeros((0, integrator.ntens))
                ),
                converged=failed_step is None,
                failed_step=failed_step,
            ),
            elapsed=elapsed,
            n_outer_iterations=n_outer,
            error=error,
        )
        if targets is not None:
            try:
                total = float(sum(t.residual(integrator, driver_factory) for t in targets))
            except _UNSOLVABLE as exc:
                run.fit_error = f"{type(exc).__name__}: {exc}"
            else:
                if math.isfinite(total):
                    run.fit_residual = total
                else:
                    run.fit_error = f"the residual evaluated to {total}"
        runs[name] = run

    return ModelComparisonResult(runs=runs, n_load_steps=len(np.asarray(load.data)))
