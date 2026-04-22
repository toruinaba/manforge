"""Simulation drivers: strain-controlled and stress-controlled loading.

All drivers share the same interface via :class:`DriverBase`:

* Input  — :class:`~manforge.simulation.types.FieldHistory` containing the
  prescribed loading history (strain or stress).
* Output — :class:`~manforge.simulation.types.DriverResult` containing
  stress and strain at every step, plus any explicitly requested state
  variables.

Conventions
-----------
- Stress and strain arrays use the engineering-shear Voigt convention:
    σ, ε = [11, 22, 33, 12, 13, 23]
- *Cumulative* quantities are the input; increments are computed internally.

Backward-compatibility aliases
-------------------------------
``UniaxialDriver`` and ``GeneralDriver`` are aliases for :class:`StrainDriver`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from manforge.core.stress_update import stress_update
from manforge.simulation.types import DriverResult, FieldHistory, FieldType


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class DriverBase(ABC):
    """Abstract base for all simulation drivers.

    Subclasses implement :meth:`run` and must accept a
    :class:`~manforge.simulation.types.FieldHistory` as the loading
    specification and return a :class:`~manforge.simulation.types.DriverResult`.
    """

    @abstractmethod
    def run(
        self,
        model,
        load: FieldHistory,
        collect_state: dict[str, FieldType] | None = None,
    ) -> DriverResult:
        """Run the loading simulation.

        Parameters
        ----------
        model : MaterialModel
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


# ---------------------------------------------------------------------------
# Strain-driven driver
# ---------------------------------------------------------------------------

class StrainDriver(DriverBase):
    """Strain-controlled loading driver.

    Accepts both uniaxial (1-D) and general multi-component (2-D) strain
    histories in a single class.

    Parameters
    ----------
    (none — stateless, all inputs passed to :meth:`run`)

    Notes
    -----
    ``UniaxialDriver`` and ``GeneralDriver`` are aliases for this class.
    """

    def run(
        self,
        model,
        load: FieldHistory,
        collect_state: dict[str, FieldType] | None = None,
        method: str = "auto",
    ) -> DriverResult:
        """Run the strain-controlled loading history.

        Parameters
        ----------
        model : MaterialModel
            Constitutive model instance.
        load : FieldHistory
            Must have ``type = FieldType.STRAIN``.  ``load.data`` shape:

            * ``(N,)``       — uniaxial: only ε11 varies, lateral strains zero.
            * ``(N, ntens)`` — general: all components prescribed.

        collect_state : dict[str, FieldType] or None, optional
            State variables to include in the result.  Example::

                collect_state={"ep": FieldType.STRAIN}

        method : {"auto", "autodiff", "analytical"}, optional
            Passed to :func:`~manforge.core.return_mapping.return_mapping`
            at every step (default ``"auto"``).

        Returns
        -------
        DriverResult
            Always contains ``"Stress"`` (N, ntens) and ``"Strain"``
            (N, ntens).  State-variable fields are added when
            ``collect_state`` is provided.
        """
        data = np.asarray(load.data, dtype=float)
        uniaxial = data.ndim == 1
        ntens = model.ntens
        N = len(data)

        stress_n = np.zeros(ntens)
        state_n = model.initial_state()
        strain_out = np.zeros((N, ntens))
        step_results = []

        for i in range(N):
            if uniaxial:
                deps11 = data[i] - (data[i - 1] if i > 0 else 0.0)
                strain_inc = np.zeros(ntens)
                strain_inc[0] = deps11
                strain_out[i, 0] = data[i]
            else:
                prev = data[i - 1] if i > 0 else np.zeros(ntens)
                strain_inc = np.array(data[i] - prev)
                strain_out[i] = data[i]

            rm = stress_update(model, strain_inc, stress_n, state_n, method=method)
            stress_n = rm.stress
            state_n = rm.state
            step_results.append(rm)

        return DriverResult(
            step_results=step_results,
            strain=strain_out,
            collect_state=collect_state,
        )


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
    max_iter : int, optional
        Maximum Newton-Raphson iterations per step (default 20).
    tol : float, optional
        Convergence tolerance on the infinity norm of the stress residual
        (default 1e-8).
    """

    def __init__(self, max_iter: int = 20, tol: float = 1e-8):
        self.max_iter = max_iter
        self.tol = tol

    def run(
        self,
        model,
        load: FieldHistory,
        collect_state: dict[str, FieldType] | None = None,
        method: str = "auto",
    ) -> DriverResult:
        """Run the stress-controlled loading history.

        Parameters
        ----------
        model : MaterialModel
            Constitutive model instance.
        load : FieldHistory
            Must have ``type = FieldType.STRESS`` and
            ``load.data`` shape ``(N, ntens)`` — cumulative target stress
            tensor (Voigt) at each step.
        collect_state : dict[str, FieldType] or None, optional
            State variables to include in the result.  Example::

                collect_state={"ep": FieldType.STRAIN}

        method : {"auto", "numerical_newton", "user_defined"}, optional
            Passed to :func:`~manforge.core.stress_update.stress_update`
            at every inner iteration (default ``"auto"``).

        Returns
        -------
        DriverResult
            Always contains ``"Stress"`` (N, ntens) and ``"Strain"``
            (N, ntens).  State-variable fields are added when
            ``collect_state`` is provided.

        Raises
        ------
        RuntimeError
            If Newton-Raphson does not converge within ``max_iter`` iterations
            at any step.
        """
        stress_history = np.asarray(load.data, dtype=float)
        N = stress_history.shape[0]
        ntens = model.ntens

        stress_n = np.zeros(ntens)
        state_n = model.initial_state()
        eps_total = np.zeros(ntens)
        strain_out = np.zeros((N, ntens))
        step_results = []

        # Elastic compliance for the initial strain-increment guess
        C = model.elastic_stiffness()
        S = np.linalg.inv(np.array(C))

        for i in range(N):
            sigma_target = np.array(stress_history[i])

            # Initial guess: elastic compliance applied to stress increment
            deps = S @ (sigma_target - stress_n)

            converged = False
            residual = np.full(ntens, np.inf)
            rm = None
            for _ in range(self.max_iter):
                rm = stress_update(model, deps, stress_n, state_n, method=method)
                residual = sigma_target - np.array(rm.stress)
                if float(np.max(np.abs(residual))) < self.tol:
                    converged = True
                    break
                deps = deps + np.linalg.solve(np.array(rm.ddsdde), residual)

            if not converged:
                raise RuntimeError(
                    f"StressDriver: NR did not converge at step {i} "
                    f"(||residual||_inf = {float(np.max(np.abs(residual))):.3e}, "
                    f"tol = {self.tol:.3e})"
                )

            stress_n = rm.stress
            state_n = rm.state
            eps_total = eps_total + np.array(deps)
            strain_out[i] = eps_total.copy()
            step_results.append(rm)

        return DriverResult(
            step_results=step_results,
            strain=strain_out,
            collect_state=collect_state,
        )


# ---------------------------------------------------------------------------
# Backward-compatibility aliases
# ---------------------------------------------------------------------------

#: Alias for :class:`StrainDriver` (formerly separate uniaxial driver).
UniaxialDriver = StrainDriver

#: Alias for :class:`StrainDriver` (formerly separate general driver).
GeneralDriver = StrainDriver
