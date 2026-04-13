"""Simulation drivers: strain-controlled and stress-controlled loading.

All drivers share the same interface via :class:`DriverBase`:

* Input  — :class:`~manforge.simulation.types.FieldHistory` containing the
  prescribed loading history (strain or stress).
* Output — :class:`~manforge.simulation.types.DriverResult` containing
  stress, strain, and all model state variables at every step.

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
import jax.numpy as jnp

from manforge.core.return_mapping import return_mapping
from manforge.simulation.types import DriverResult, FieldHistory, FieldType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_state(state_n: dict) -> dict[str, np.ndarray]:
    """Convert a model state dict to a dict of 1-D-or-2-D numpy arrays."""
    out = {}
    for k, v in state_n.items():
        arr = np.array(v)
        if arr.ndim == 0:
            arr = arr.reshape(1)   # scalar → (1,)
        out[k] = arr
    return out


def _finalise_state_fields(state_accum: dict[str, list]) -> dict[str, FieldHistory]:
    """Stack per-step state lists into FieldHistory objects."""
    fields = {}
    for k, rows in state_accum.items():
        stacked = np.stack(rows, axis=0)          # (N,) or (N, ntens)
        if stacked.ndim == 2 and stacked.shape[1] == 1:
            stacked = stacked[:, 0]               # (N, 1) → (N,)
        fields[k] = FieldHistory(FieldType.SCALAR, k, stacked)
    return fields


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
    def run(self, model, load: FieldHistory, params: dict) -> DriverResult:
        """Run the loading simulation.

        Parameters
        ----------
        model : MaterialModel
        load : FieldHistory
            Loading history.  The ``type`` and shape of ``load.data`` must
            match the driver's expectations (see subclass documentation).
        params : dict
            Material parameters.

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

    def run(self, model, load: FieldHistory, params: dict) -> DriverResult:
        """Run the strain-controlled loading history.

        Parameters
        ----------
        model : MaterialModel
            Constitutive model instance.
        load : FieldHistory
            Must have ``type = FieldType.STRAIN``.  ``load.data`` shape:

            * ``(N,)``       — uniaxial: only ε11 varies, lateral strains zero.
            * ``(N, ntens)`` — general: all components prescribed.

        params : dict
            Material parameters.

        Returns
        -------
        DriverResult
            Fields: ``"Stress"`` (N, ntens), ``"Strain"`` (N, ntens), plus
            one ``FieldType.SCALAR`` field per model state variable.
        """
        data = np.asarray(load.data, dtype=float)
        uniaxial = data.ndim == 1
        ntens = model.ntens
        N = len(data)

        stress_n = jnp.zeros(ntens)
        state_n = model.initial_state()
        state_accum: dict[str, list] = {k: [] for k in state_n}

        stress_out = np.zeros((N, ntens))
        strain_out = np.zeros((N, ntens))

        for i in range(N):
            if uniaxial:
                deps11 = data[i] - (data[i - 1] if i > 0 else 0.0)
                strain_inc = jnp.zeros(ntens).at[0].set(deps11)
                strain_out[i, 0] = data[i]
            else:
                prev = data[i - 1] if i > 0 else np.zeros(ntens)
                strain_inc = jnp.array(data[i] - prev)
                strain_out[i] = data[i]

            stress_n, state_n, _ = return_mapping(
                model, strain_inc, stress_n, state_n, params
            )
            stress_out[i] = np.array(stress_n)
            for k, row in _collect_state(state_n).items():
                state_accum[k].append(row)

        fields: dict[str, FieldHistory] = {
            "Stress": FieldHistory(FieldType.STRESS, "Stress", stress_out),
            "Strain": FieldHistory(FieldType.STRAIN, "Strain", strain_out),
        }
        fields.update(_finalise_state_fields(state_accum))
        return DriverResult(fields=fields)


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

    def run(self, model, load: FieldHistory, params: dict) -> DriverResult:
        """Run the stress-controlled loading history.

        Parameters
        ----------
        model : MaterialModel
            Constitutive model instance.
        load : FieldHistory
            Must have ``type = FieldType.STRESS`` and
            ``load.data`` shape ``(N, ntens)`` — cumulative target stress
            tensor (Voigt) at each step.
        params : dict
            Material parameters.

        Returns
        -------
        DriverResult
            Fields: ``"Stress"`` (N, ntens), ``"Strain"`` (N, ntens), plus
            one ``FieldType.SCALAR`` field per model state variable.

        Raises
        ------
        RuntimeError
            If Newton-Raphson does not converge within ``max_iter`` iterations
            at any step.
        """
        stress_history = np.asarray(load.data, dtype=float)
        N = stress_history.shape[0]
        ntens = model.ntens

        stress_n = jnp.zeros(ntens)
        state_n = model.initial_state()
        eps_total = np.zeros(ntens)
        state_accum: dict[str, list] = {k: [] for k in state_n}

        # Elastic compliance for the initial strain-increment guess
        C = model.elastic_stiffness(params)
        S = jnp.linalg.inv(C)

        stress_out = np.zeros((N, ntens))
        strain_out = np.zeros((N, ntens))

        for i in range(N):
            sigma_target = jnp.array(stress_history[i])

            # Initial guess: elastic compliance applied to stress increment
            deps = S @ (sigma_target - stress_n)

            converged = False
            residual = jnp.full(ntens, jnp.inf)
            for _ in range(self.max_iter):
                stress_new, state_new, ddsdde = return_mapping(
                    model, deps, stress_n, state_n, params
                )
                residual = sigma_target - stress_new
                if float(jnp.max(jnp.abs(residual))) < self.tol:
                    converged = True
                    break
                deps = deps + jnp.linalg.solve(ddsdde, residual)

            if not converged:
                raise RuntimeError(
                    f"StressDriver: NR did not converge at step {i} "
                    f"(||residual||_inf = {float(jnp.max(jnp.abs(residual))):.3e}, "
                    f"tol = {self.tol:.3e})"
                )

            stress_n = stress_new
            state_n = state_new
            eps_total = eps_total + np.array(deps)
            stress_out[i] = np.array(stress_new)
            strain_out[i] = eps_total.copy()
            for k, row in _collect_state(state_n).items():
                state_accum[k].append(row)

        fields: dict[str, FieldHistory] = {
            "Stress": FieldHistory(FieldType.STRESS, "Stress", stress_out),
            "Strain": FieldHistory(FieldType.STRAIN, "Strain", strain_out),
        }
        fields.update(_finalise_state_fields(state_accum))
        return DriverResult(fields=fields)


# ---------------------------------------------------------------------------
# Backward-compatibility aliases
# ---------------------------------------------------------------------------

#: Alias for :class:`StrainDriver` (formerly separate uniaxial driver).
UniaxialDriver = StrainDriver

#: Alias for :class:`StrainDriver` (formerly separate general driver).
GeneralDriver = StrainDriver
