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
    fields : dict[str, FieldHistory]
        All output fields keyed by name.  At minimum contains ``"Stress"``
        and ``"Strain"``.  State variables (e.g. ``"ep"``) are included
        when ``collect_state`` is passed to the driver's ``run()`` method.

    Examples
    --------
    Access common outputs via convenience properties::

        result = driver.run(model, load)
        result.stress          # np.ndarray (N, ntens)
        result.strain          # np.ndarray (N, ntens)

    Access state-variable history directly::

        result.fields["ep"].data          # np.ndarray (N,)  — scalar, strain-space
        result.fields["ep"].type          # FieldType.STRAIN
    """

    fields: dict[str, FieldHistory]

    @property
    def stress(self) -> np.ndarray:
        """Stress history, shape ``(N, ntens)``."""
        return self.fields["Stress"].data

    @property
    def strain(self) -> np.ndarray:
        """Strain history, shape ``(N, ntens)``."""
        return self.fields["Strain"].data
