"""Simulation drivers: strain-controlled and stress-controlled loading.

All drivers share the same interface via :class:`DriverBase`:

* Input  — the driver holds a :class:`~manforge.simulation.integrator.StressIntegrator`
  (passed at construction) and receives a
  :class:`~manforge.simulation.types.FieldHistory` containing the prescribed
  loading history (strain or stress) at each :meth:`run` / :meth:`iter_run` call.
* Output — :class:`~manforge.simulation.types.DriverResult` containing
  stress and strain at every step, plus any explicitly requested state
  variables.

Conventions
-----------
- Stress and strain arrays use the engineering-shear Voigt convention:
    σ, ε = [11, 22, 33, 12, 13, 23]
- *Cumulative* quantities are the input; increments are computed internally.
- Drivers require a :class:`~manforge.simulation.integrator.StressIntegrator`
  at construction time.  Wrap bare ``MaterialModel`` objects with
  :class:`~manforge.simulation.integrator.PythonIntegrator` (auto solver),
  :class:`~manforge.simulation.integrator.PythonNumericalIntegrator`
  (numerical Newton-Raphson), or
  :class:`~manforge.simulation.integrator.PythonAnalyticalIntegrator`
  (analytical / user-defined) before passing to a driver.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np

from manforge.simulation.types import DriverResult, DriverStep, FieldHistory, FieldType


def _check_integrator(obj) -> None:
    """Raise TypeError when obj is not a StressIntegrator."""
    if not (hasattr(obj, "stress_update") and callable(obj.stress_update)):
        raise TypeError(
            f"Driver expects a StressIntegrator, got {type(obj).__name__!r}.  "
            "Wrap a bare MaterialModel with PythonIntegrator(model) first."
        )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class DriverBase(ABC):
    """Abstract base for all simulation drivers.

    Parameters
    ----------
    integrator : StressIntegrator
        Constitutive integrator — use
        :class:`~manforge.simulation.integrator.PythonIntegrator`,
        :class:`~manforge.simulation.integrator.PythonNumericalIntegrator`,
        :class:`~manforge.simulation.integrator.PythonAnalyticalIntegrator`, or
        :class:`~manforge.simulation.integrator.FortranIntegrator`.
    """

    def __init__(self, integrator) -> None:
        _check_integrator(integrator)
        self.integrator = integrator

    @abstractmethod
    def iter_run(
        self,
        load: FieldHistory,
    ) -> Iterator[DriverStep]:
        """Yield per-step results as a generator.

        Parameters
        ----------
        load : FieldHistory
            Loading history (same requirements as :meth:`run`).

        Yields
        ------
        DriverStep
            Snapshot after each step: cumulative strain, full
            :class:`~manforge.core.stress_update.StressUpdateResult`, and
            (for :class:`StressDriver`) outer-NR diagnostics.

        Examples
        --------
        Break on plasticity onset::

            driver = StrainDriver(PythonIntegrator(model))
            for step in driver.iter_run(load):
                if step.result.is_plastic:
                    print(f"Plasticity at step {step.i}")
                    break
        """

    def run(
        self,
        load: FieldHistory,
        *,
        collect_state: dict[str, FieldType] | None = None,
    ) -> DriverResult:
        """Run the loading simulation.

        Parameters
        ----------
        load : FieldHistory
            Loading history.  The ``type`` and shape of ``load.data`` must
            match the driver's expectations (see subclass documentation).
        collect_state : dict[str, FieldType] or None, optional
            Explicitly request state-variable histories to be included in the
            result.  Keys are model state keys (e.g. ``"ep"``); values specify
            the :class:`~manforge.simulation.types.FieldType` to assign to the
            resulting :class:`~manforge.simulation.types.FieldHistory`.

            If ``None`` (default), no state variables are collected and
            ``DriverResult.fields`` contains only ``"Stress"`` and ``"Strain"``.

        Returns
        -------
        DriverResult
        """
        step_results = []
        strain_rows = []
        for step in self.iter_run(load):
            step_results.append(step.result)
            strain_rows.append(step.strain)
        strain_out = (
            np.stack(strain_rows)
            if strain_rows
            else np.zeros((0, self.integrator.ntens))
        )
        return DriverResult(
            step_results=step_results,
            strain=strain_out,
            collect_state=collect_state,
        )


# ---------------------------------------------------------------------------
# Strain-driven driver
# ---------------------------------------------------------------------------

class StrainDriver(DriverBase):
    """Strain-controlled loading driver.

    Accepts both uniaxial (1-D) and general multi-component (2-D) strain
    histories in a single class.

    Parameters
    ----------
    integrator : StressIntegrator
        Constitutive integrator.
    """

    def iter_run(
        self,
        load: FieldHistory,
    ) -> Iterator[DriverStep]:
        """Yield per-step results for the strain-controlled loading history.

        Parameters
        ----------
        load : FieldHistory
            Must have ``type = FieldType.STRAIN``.  ``load.data`` shape:

            * ``(N,)``       — uniaxial: only ε11 varies, lateral strains zero.
            * ``(N, ntens)`` — general: all components prescribed.

        Yields
        ------
        DriverStep
        """
        integrator = self.integrator
        data = np.asarray(load.data, dtype=float)
        uniaxial = data.ndim == 1
        ntens = integrator.ntens

        stress_n = np.zeros(ntens)
        state_n = integrator.initial_state()

        for i in range(len(data)):
            if uniaxial:
                deps11 = data[i] - (data[i - 1] if i > 0 else 0.0)
                strain_inc = np.zeros(ntens)
                strain_inc[0] = deps11
                strain_cum = np.zeros(ntens)
                strain_cum[0] = data[i]
            else:
                prev = data[i - 1] if i > 0 else np.zeros(ntens)
                strain_inc = np.array(data[i] - prev)
                strain_cum = np.array(data[i])

            rm = integrator.stress_update(strain_inc, stress_n, state_n)
            stress_n = rm.stress
            state_n = rm.state
            yield DriverStep(i=i, strain=strain_cum.copy(), result=rm)


# ---------------------------------------------------------------------------
# Stress-driven driver
# ---------------------------------------------------------------------------

class StressDriver(DriverBase):
    """Stress-controlled loading driver.

    Prescribes a target stress history and solves for the corresponding
    strain increments using Newton-Raphson iteration with the consistent
    tangent (ddsdde).  Useful for simulating uniaxial stress loading in a
    multi-axial model (e.g. σ11 ramping, all other components zero) where
    the lateral strains adjust freely.

    Parameters
    ----------
    integrator : StressIntegrator
        Constitutive integrator.
    max_iter : int, optional
        Maximum Newton-Raphson iterations per step (default 20).
    tol : float, optional
        Convergence tolerance on the infinity norm of the stress residual
        (default 1e-8).
    """

    def __init__(self, integrator, *, max_iter: int = 20, tol: float = 1e-8):
        super().__init__(integrator)
        self.max_iter = max_iter
        self.tol = tol

    def iter_run(
        self,
        load: FieldHistory,
        *,
        raise_on_nonconverged: bool = True,
    ) -> Iterator[DriverStep]:
        """Yield per-step results for the stress-controlled loading history.

        Parameters
        ----------
        load : FieldHistory
            Must have ``type = FieldType.STRESS`` and
            ``load.data`` shape ``(N, ntens)`` — cumulative target stress
            tensor (Voigt) at each step.
        raise_on_nonconverged : bool, optional
            If ``True`` (default), raise :exc:`RuntimeError` when Newton-Raphson
            does not converge.  If ``False``, yield a :class:`DriverStep` with
            ``converged=False`` instead; internal state is *not* advanced for
            that step, so the caller should ``break`` immediately.

        Yields
        ------
        DriverStep
            Includes ``n_outer_iter`` and ``residual_inf`` for NR diagnostics.

        Raises
        ------
        RuntimeError
            If NR does not converge and ``raise_on_nonconverged=True``.
        """
        integrator = self.integrator
        stress_history = np.asarray(load.data, dtype=float)
        ntens = integrator.ntens

        stress_n = np.zeros(ntens)
        state_n = integrator.initial_state()
        eps_total = np.zeros(ntens)

        C = integrator.elastic_stiffness()
        S = np.linalg.inv(np.array(C))

        for i in range(stress_history.shape[0]):
            sigma_target = np.array(stress_history[i])
            deps = S @ (sigma_target - stress_n)

            converged = False
            residual = np.full(ntens, np.inf)
            rm = None
            k = 0
            for k in range(self.max_iter):
                rm = integrator.stress_update(deps, stress_n, state_n)
                residual = sigma_target - np.array(rm.stress)
                if float(np.max(np.abs(residual))) < self.tol:
                    converged = True
                    break
                deps = deps + np.linalg.solve(np.array(rm.ddsdde), residual)

            residual_inf = float(np.max(np.abs(residual)))

            if not converged:
                if raise_on_nonconverged:
                    raise RuntimeError(
                        f"StressDriver: NR did not converge at step {i} "
                        f"(||residual||_inf = {residual_inf:.3e}, "
                        f"tol = {self.tol:.3e})"
                    )
                yield DriverStep(
                    i=i,
                    strain=eps_total.copy(),
                    result=rm,
                    converged=False,
                    n_outer_iter=k + 1,
                    residual_inf=residual_inf,
                )
                return

            eps_total = eps_total + np.array(deps)
            stress_n = rm.stress
            state_n = rm.state
            yield DriverStep(
                i=i,
                strain=eps_total.copy(),
                result=rm,
                converged=True,
                n_outer_iter=k + 1,
                residual_inf=residual_inf,
            )

    def run(
        self,
        load: FieldHistory,
        *,
        collect_state: dict[str, FieldType] | None = None,
    ) -> DriverResult:
        """Run the stress-controlled loading history.

        Parameters
        ----------
        load : FieldHistory
            Must have ``type = FieldType.STRESS`` and
            ``load.data`` shape ``(N, ntens)`` — cumulative target stress
            tensor (Voigt) at each step.
        collect_state : dict[str, FieldType] or None, optional
            State variables to include in the result.

        Returns
        -------
        DriverResult

        Raises
        ------
        RuntimeError
            If Newton-Raphson does not converge within ``max_iter`` iterations
            at any step.
        """
        step_results = []
        strain_rows = []
        for step in self.iter_run(load, raise_on_nonconverged=True):
            step_results.append(step.result)
            strain_rows.append(step.strain)
        strain_out = (
            np.stack(strain_rows)
            if strain_rows
            else np.zeros((0, self.integrator.ntens))
        )
        return DriverResult(
            step_results=step_results,
            strain=strain_out,
            collect_state=collect_state,
        )
