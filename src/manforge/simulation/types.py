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

from dataclasses import dataclass
from enum import Enum

import numpy as np


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


@dataclass
class DriverResult:
    """Results from a single driver simulation run.

    Parameters
    ----------
    step_results : list
        Per-step :class:`~manforge.core.stress_update.StressUpdateResult`
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
    collect_state: "dict[str, FieldType] | None" = None

    @property
    def stress(self) -> np.ndarray:
        """Stress history, shape ``(N, ntens)``."""
        return np.array([np.asarray(r.stress) for r in self.step_results])

    @property
    def fields(self) -> "dict[str, FieldHistory]":
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
