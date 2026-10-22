"""Serializable-boundary simulation entry point.

:func:`run_simulation` wraps model construction, integrator setup, and a
driver run behind plain inputs (dict / ndarray) and a plain dataclass output,
so service consumers (web APIs, job workers) never touch framework objects.

Non-convergence is a value, not an exception: a user exploring material
parameters will routinely submit values where return mapping has no solution,
so the partial history up to the failing step is returned with
``converged=False`` / ``failed_step`` instead of raising.  Setup errors
(unknown parameter or field names, malformed load shapes) do raise — those are
caller bugs, not material physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import numpy as np

from manforge.core.material import MaterialModel
from manforge.simulation.driver import StrainDriver, StressDriver
from manforge.simulation.integrator import PythonIntegrator
from manforge.simulation.types import FieldHistory, FieldType


@dataclass
class SimulationOutput:
    """Result of one :func:`run_simulation` call — plain arrays only.

    Parameters
    ----------
    stress : np.ndarray, shape (n_steps, ntens)
        Stress history over completed steps.
    strain : np.ndarray, shape (n_steps, ntens)
        Cumulative strain history (computed by the driver, so it is the
        authoritative x-axis for both strain- and stress-controlled runs).
    fields : dict[str, np.ndarray]
        State-variable histories requested via ``include_fields``.
    converged : bool
        ``False`` when the run stopped early on a non-converged step.
    failed_step : int or None
        0-based index of the step that failed to converge (``None`` when
        ``converged``).  The failed step itself is excluded from the arrays.
    n_steps : int
        Number of completed steps (array length).
    """

    stress: np.ndarray
    strain: np.ndarray
    fields: dict[str, np.ndarray] = field(default_factory=dict)
    converged: bool = True
    failed_step: int | None = None
    n_steps: int = 0


def run_simulation(
    model_cls: type[MaterialModel],
    params: dict[str, float],
    load: np.ndarray,
    *,
    control: Literal["strain", "stress"] = "strain",
    include_fields: Sequence[str] = (),
    nr_jacobian: str = "broyden",
    progress: Callable[[int, int], None] | None = None,
) -> SimulationOutput:
    """Run a strain- or stress-controlled history and return plain arrays.

    Parameters
    ----------
    model_cls : type[MaterialModel]
        Concrete model class (typically from a
        :class:`~manforge.service.registry.ModelRegistry`).
    params : dict[str, float]
        Material parameters.  Keys must match ``model_cls.param_names``
        exactly (missing or extra keys raise ``ValueError``).
    load : array-like
        Cumulative load history.  Shape ``(N,)`` (uniaxial shorthand) or
        ``(N, ntens)``.  Interpreted per ``control``.
    control : {"strain", "stress"}
        Driving mode.  ``"strain"`` → :class:`StrainDriver`, ``"stress"`` →
        :class:`StressDriver`.
    include_fields : sequence of str
        Model state keys to include in ``SimulationOutput.fields``.
        Unknown names raise ``ValueError`` listing the available keys.
    nr_jacobian : {"broyden", "exact"}
        Return-mapping Jacobian strategy.  Default ``"broyden"`` — measured
        ~2.4× faster on autodiff-NR models; the consistent tangent stays
        exact either way.
    progress : callable or None
        Called as ``progress(done, total)`` after each completed step.

    Raises
    ------
    ValueError
        Parameter-name mismatch, unknown ``include_fields`` entry, unknown
        ``control``, or malformed ``load`` shape.
    """
    expected = set(model_cls.param_names)
    got = set(params)
    if got != expected:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        raise ValueError(
            f"{model_cls.__name__}: parameter mismatch"
            + (f" — missing {missing}" if missing else "")
            + (f" — unexpected {extra}" if extra else "")
        )

    model = model_cls(**params)
    unknown = sorted(set(include_fields) - set(model.state_names))
    if unknown:
        raise ValueError(
            f"unknown state fields {unknown}; available: {model.state_names}"
        )

    data = np.asarray(load, dtype=float)
    ntens = model.ntens
    if data.ndim == 1:
        if control == "stress" and ntens > 1:
            raise ValueError(
                f"stress control with ntens={ntens} requires load shape (N, {ntens}), "
                f"got {data.shape}"
            )
    elif data.ndim == 2:
        if data.shape[1] != ntens:
            raise ValueError(
                f"load shape {data.shape} does not match ntens={ntens}"
            )
    else:
        raise ValueError(f"load must be 1-D or 2-D, got shape {data.shape}")

    integrator = PythonIntegrator(
        model, nr_jacobian=nr_jacobian, raise_on_nonconverged=False
    )

    n_total = data.shape[0]
    if control == "strain":
        driver = StrainDriver(integrator)
        history = FieldHistory(FieldType.STRAIN, "Strain", data)
        step_iter = driver.iter_run(history)
    elif control == "stress":
        driver = StressDriver(integrator)
        stress_data = data if data.ndim == 2 else data.reshape(-1, 1)
        history = FieldHistory(FieldType.STRESS, "Stress", stress_data)
        step_iter = driver.iter_run(history, raise_on_nonconverged=False)
    else:
        raise ValueError(f"control must be 'strain' or 'stress', got {control!r}")

    completed = []
    converged = True
    failed_step: int | None = None
    try:
        for step in step_iter:
            if not step.converged:
                converged = False
                failed_step = step.i
                break
            completed.append(step)
            if progress is not None:
                progress(step.i + 1, n_total)
    except (np.linalg.LinAlgError, FloatingPointError, ZeroDivisionError):
        # A singular NR Jacobian or numerical blow-up means these parameters
        # are unsolvable at this step — same information as non-convergence.
        converged = False
        failed_step = len(completed)

    if completed:
        stress = np.stack([np.asarray(s.result.stress) for s in completed])
        strain = np.stack([s.strain for s in completed])
        fields = {
            k: np.stack([np.asarray(s.result.state[k]) for s in completed])
            for k in include_fields
        }
    else:
        stress = np.zeros((0, ntens))
        strain = np.zeros((0, ntens))
        fields = {k: np.zeros((0,)) for k in include_fields}

    return SimulationOutput(
        stress=stress,
        strain=strain,
        fields=fields,
        converged=converged,
        failed_step=failed_step,
        n_steps=len(completed),
    )
