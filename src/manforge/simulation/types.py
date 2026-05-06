"""Unified data types for simulation input and output.

All physical quantities in a driver simulation are represented as
:class:`FieldHistory` instances tagged with a :class:`FieldType`.  The
two types correspond to the physical space of the quantity:

* ``STRESS`` — stress-space quantity  (e.g. Cauchy stress, back stress, yield stress)
* ``STRAIN`` — strain-space quantity  (e.g. total strain, plastic strain, equivalent plastic strain)

Whether a quantity is a full tensor (shape ``(N, ntens)``) or a scalar
(shape ``(N,)``) is conveyed by ``data.shape``, not by ``FieldType``.

:class:`DriverResult` collects all output fields from a driver run.
Its ``stress`` and ``strain`` properties provide shortcut access to the
primary outputs; arbitrary fields (e.g. state variables) are accessible
via ``result.fields["name"]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from manforge.core.result import StressUpdateResult


def _piecewise_linear(
    peaks: Sequence[float],
    n_per_segment: int,
    start: float = 0.0,
) -> np.ndarray:
    """Concatenate linear segments [start→peaks[0]→peaks[1]→...].

    Returns a 1-D array of length ``len(peaks) * n_per_segment`` using the
    same ``np.linspace(...)[1:]`` idiom as ``generate_strain_history``.
    """
    current = start
    parts: list[np.ndarray] = []
    for target in peaks:
        seg = np.linspace(current, target, n_per_segment + 1)[1:]
        parts.append(seg)
        current = float(target)
    return np.concatenate(parts)


def _broadcast_to_tensor(axial: np.ndarray, ntens: int, component: int) -> np.ndarray:
    """Map a 1-D axial history to shape ``(N, ntens)`` with one non-zero column."""
    if not (0 <= component < ntens):
        raise ValueError(
            f"component={component} is out of range for ntens={ntens}"
        )
    out = np.zeros((len(axial), ntens))
    out[:, component] = axial
    return out


class FieldType(Enum):
    """Physical space of a :class:`FieldHistory`.

    Classifies whether a quantity lives in stress space or strain space.
    The tensorial rank (full tensor vs scalar) is inferred from
    ``data.shape``: ``(N, ntens)`` for tensors, ``(N,)`` for scalars.
    """

    STRESS = "stress"
    """Stress-space quantity.

    Examples: Cauchy stress σ, back stress α (ntens,), yield stress σ_y (scalar).
    """

    STRAIN = "strain"
    """Strain-space quantity.

    Examples: total strain ε, plastic strain εp (ntens,), equivalent plastic strain ep (scalar).
    """


@dataclass
class FieldHistory:
    """Time-series of a named physical quantity.

    Parameters
    ----------
    type : FieldType
        Physical space of the quantity (STRESS or STRAIN).
    name : str
        Identifier (e.g. ``"Stress"``, ``"Strain"``, ``"ep"``).
    data : np.ndarray
        Array of shape ``(N, ntens)`` for tensor quantities or ``(N,)``
        for scalar quantities, where *N* is the number of steps.
    """

    type: FieldType
    name: str
    data: np.ndarray

    # ------------------------------------------------------------------
    # Cyclic loading constructors
    # ------------------------------------------------------------------

    @classmethod
    def cyclic_strain(
        cls,
        peaks: Sequence[float],
        *,
        n_per_segment: int = 20,
        component: int = 0,
        ntens: int = 1,
        name: str = "Strain",
        start: float = 0.0,
    ) -> "FieldHistory":
        """Piecewise-linear strain history through a list of peak values.

        Parameters
        ----------
        peaks:
            Ordered list of peak (reversal) values. The history goes
            ``start → peaks[0] → peaks[1] → ...``.
        n_per_segment:
            Number of steps per linear segment (endpoint included, start
            excluded — same as ``generate_strain_history``).
        component:
            Index of the non-zero column in the ``(N, ntens)`` output.
        ntens:
            Number of stress/strain components.
        name:
            Label stored in the :class:`FieldHistory`.
        start:
            Starting value (typically 0.0).
        """
        axial = _piecewise_linear(peaks, n_per_segment, start)
        data = _broadcast_to_tensor(axial, ntens, component)
        return cls(FieldType.STRAIN, name, data)

    @classmethod
    def cyclic_stress(
        cls,
        peaks: Sequence[float],
        *,
        n_per_segment: int = 20,
        component: int = 0,
        ntens: int = 1,
        name: str = "Stress",
        start: float = 0.0,
    ) -> "FieldHistory":
        """Piecewise-linear stress history through a list of peak values."""
        fh = cls.cyclic_strain(
            peaks,
            n_per_segment=n_per_segment,
            component=component,
            ntens=ntens,
            name=name,
            start=start,
        )
        return cls(FieldType.STRESS, name, fh.data)

    @classmethod
    def triangular_strain(
        cls,
        amplitude: float,
        n_cycles: int,
        *,
        n_per_segment: int = 20,
        component: int = 0,
        ntens: int = 1,
        name: str = "Strain",
        mean: float = 0.0,
    ) -> "FieldHistory":
        """Symmetric triangular-wave strain history.

        Generates ``n_cycles`` full cycles: ``mean → mean+amplitude →
        mean-amplitude → mean+amplitude → ...`` (each half-cycle is one
        segment of ``n_per_segment`` steps).
        """
        peaks: list[float] = []
        for _ in range(n_cycles):
            peaks.append(mean + amplitude)
            peaks.append(mean - amplitude)
        axial = _piecewise_linear(peaks, n_per_segment, mean)
        data = _broadcast_to_tensor(axial, ntens, component)
        return cls(FieldType.STRAIN, name, data)

    @classmethod
    def triangular_stress(
        cls,
        amplitude: float,
        n_cycles: int,
        *,
        n_per_segment: int = 20,
        component: int = 0,
        ntens: int = 1,
        name: str = "Stress",
        mean: float = 0.0,
    ) -> "FieldHistory":
        """Symmetric triangular-wave stress history."""
        fh = cls.triangular_strain(
            amplitude,
            n_cycles,
            n_per_segment=n_per_segment,
            component=component,
            ntens=ntens,
            name=name,
            mean=mean,
        )
        return cls(FieldType.STRESS, name, fh.data)

    @classmethod
    def sine_strain(
        cls,
        amplitude: float,
        n_cycles: int,
        *,
        n_per_cycle: int = 80,
        component: int = 0,
        ntens: int = 1,
        name: str = "Strain",
        mean: float = 0.0,
        phase: float = 0.0,
    ) -> "FieldHistory":
        """Sinusoidal strain history.

        Parameters
        ----------
        amplitude:
            Half-amplitude of the sine wave.
        n_cycles:
            Number of complete cycles.
        n_per_cycle:
            Steps per cycle (controls smoothness).
        mean:
            DC offset added to the sine wave.
        phase:
            Phase offset in radians (default 0 starts from ``mean``).
        """
        _broadcast_to_tensor(np.zeros(1), ntens, component)  # validate component
        N = n_cycles * n_per_cycle
        t = np.linspace(0.0, 2.0 * np.pi * n_cycles, N, endpoint=False)
        axial = mean + amplitude * np.sin(t + phase)
        data = _broadcast_to_tensor(axial, ntens, component)
        return cls(FieldType.STRAIN, name, data)

    @classmethod
    def sine_stress(
        cls,
        amplitude: float,
        n_cycles: int,
        *,
        n_per_cycle: int = 80,
        component: int = 0,
        ntens: int = 1,
        name: str = "Stress",
        mean: float = 0.0,
        phase: float = 0.0,
    ) -> "FieldHistory":
        """Sinusoidal stress history."""
        fh = cls.sine_strain(
            amplitude,
            n_cycles,
            n_per_cycle=n_per_cycle,
            component=component,
            ntens=ntens,
            name=name,
            mean=mean,
            phase=phase,
        )
        return cls(FieldType.STRESS, name, fh.data)

    @classmethod
    def decaying_cyclic_strain(
        cls,
        amplitude: float,
        n_cycles: int,
        *,
        decay: float = 0.5,
        n_per_segment: int = 20,
        component: int = 0,
        ntens: int = 1,
        name: str = "Strain",
    ) -> "FieldHistory":
        """Cyclic strain history with exponentially varying amplitude.

        Peak amplitude of cycle *k* (0-indexed) is ``amplitude * decay**k``.
        Setting ``decay=1.0`` gives a constant-amplitude triangular wave.
        Values ``decay < 1.0`` produce a decaying envelope; ``decay > 1.0``
        produce a growing envelope.
        """
        peaks: list[float] = []
        for k in range(n_cycles):
            a = amplitude * (decay ** k)
            peaks.append(+a)
            peaks.append(-a)
        axial = _piecewise_linear(peaks, n_per_segment, 0.0)
        data = _broadcast_to_tensor(axial, ntens, component)
        return cls(FieldType.STRAIN, name, data)

    @classmethod
    def decaying_cyclic_stress(
        cls,
        amplitude: float,
        n_cycles: int,
        *,
        decay: float = 0.5,
        n_per_segment: int = 20,
        component: int = 0,
        ntens: int = 1,
        name: str = "Stress",
    ) -> "FieldHistory":
        """Cyclic stress history with exponentially varying amplitude."""
        fh = cls.decaying_cyclic_strain(
            amplitude,
            n_cycles,
            decay=decay,
            n_per_segment=n_per_segment,
            component=component,
            ntens=ntens,
            name=name,
        )
        return cls(FieldType.STRESS, name, fh.data)


@dataclass
class DriverResult:
    """Results from a single driver simulation run.

    Parameters
    ----------
    step_results : list
        Per-step :class:`~manforge.core.result.StressUpdateResult`
        objects, one per increment.  Provides access to converged stress,
        state, tangent (ddsdde), dlambda, stress_trial, and is_plastic for every step.
    strain : np.ndarray, shape (N, ntens)
        Cumulative strain history computed by the driver.
    collect_state : dict[str, FieldType] or None
        State-variable fields requested at driver construction time.

    Examples
    --------
    Access common outputs::

        result = driver.run(model, load)
        result.stress          # np.ndarray (N, ntens)
        result.strain          # np.ndarray (N, ntens)

    Access a specific step's details::

        rm = result.step_results[15]
        rm.dlambda             # plastic multiplier at step 15
        rm.is_plastic          # whether step 15 was plastic

    Access state-variable history via fields::

        result.fields["ep"].data   # np.ndarray (N,) — scalar, strain-space
    """

    step_results: list
    strain: np.ndarray
    collect_state: dict[str, FieldType] | None = None

    @property
    def stress(self) -> np.ndarray:
        """Stress history, shape ``(N, ntens)``."""
        return np.array([np.asarray(r.stress) for r in self.step_results])

    @property
    def fields(self) -> dict[str, FieldHistory]:
        """All output fields constructed from step_results."""
        out: dict[str, FieldHistory] = {
            "Stress": FieldHistory(FieldType.STRESS, "Stress", self.stress),
            "Strain": FieldHistory(FieldType.STRAIN, "Strain", self.strain),
        }
        if self.collect_state:
            for k, ft in self.collect_state.items():
                data = np.stack(
                    [np.asarray(r.state[k]) for r in self.step_results]
                )
                out[k] = FieldHistory(ft, k, data)
        return out


@dataclass
class DriverStep:
    """Per-step snapshot yielded by :meth:`~manforge.simulation.driver.DriverBase.iter_run`.

    Parameters
    ----------
    i : int
        0-based step index.
    strain : np.ndarray, shape (ntens,)
        Cumulative strain at this step (copied — safe to retain across steps).
    result : StressUpdateResult
        Full stress-update result: stress, state, ddsdde, dlambda, is_plastic, etc.
    converged : bool
        Outer NR convergence flag (StressDriver only; always ``True`` for StrainDriver).
    n_outer_iter : int
        Number of outer NR iterations (StressDriver only; ``1`` for StrainDriver).
    residual_inf : float
        Final L∞ residual of outer NR (StressDriver only; ``0.0`` for StrainDriver).

    Examples
    --------
    Break on the first plastic step::

        for step in driver.iter_run(model, load):
            if step.result.is_plastic:
                print(f"Plasticity onset at step {step.i}")
                break

    Inspect StressDriver convergence details::

        for step in driver.iter_run(model, load):
            print(step.i, step.n_outer_iter, step.residual_inf)
    """

    i: int
    strain: np.ndarray
    result: StressUpdateResult
    converged: bool = True
    n_outer_iter: int = 1
    residual_inf: float = 0.0
