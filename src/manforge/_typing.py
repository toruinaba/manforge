"""Shared type aliases for manforge public API annotations.

All numeric operations use ``autograd.numpy``, but ``autograd.numpy.ndarray``
is the same object as ``numpy.ndarray`` at runtime (autograd wraps numpy via
namespace copying).  Static type-checkers only understand ``numpy.ndarray`` /
``NDArray``, so this module provides properly-typed aliases that the rest of
the codebase imports for annotations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Array aliases
# ---------------------------------------------------------------------------

FloatArray = NDArray[np.float64]
"""Generic floating-point ndarray — use for arrays whose shape is unspecified."""

Scalar = FloatArray
"""0-d or scalar ndarray (shape () or ())."""

StressVec = FloatArray
"""1-D stress/strain vector, shape (ntens,)."""

Stiffness = FloatArray
"""2-D stiffness matrix, shape (ntens, ntens)."""

StateDict = dict[str, FloatArray]
"""Internal state as a plain dict mapping field name → array."""

if TYPE_CHECKING:
    pass

__all__ = [
    "FloatArray",
    "Scalar",
    "StressVec",
    "Stiffness",
    "StateDict",
]
