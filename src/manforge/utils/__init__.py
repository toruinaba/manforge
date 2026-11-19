"""Helpers for model code: smooth approximations and spectral functions.

Both families are differentiable everywhere, including at the points where
the functions they stand in for are not.
"""

from manforge.utils.smooth import (
    smooth_sqrt,
    smooth_abs,
    smooth_norm,
    smooth_macaulay,
    smooth_direction,
    smooth_heaviside,
    smooth_min,
    smooth_max,
)
from manforge.utils.spectral import (
    DEFAULT_HEAVISIDE_AT_ZERO,
    voigt_stress_to_tensor,
    principal_values,
    matrix_function,
    spectral_trace,
    positive_part,
    negative_part,
    positive_part_sq,
    negative_part_sq,
    macaulay_sq,
    neg_macaulay_sq,
)

__all__ = [
    # smooth approximations
    "smooth_sqrt",
    "smooth_abs",
    "smooth_norm",
    "smooth_macaulay",
    "smooth_direction",
    "smooth_heaviside",
    "smooth_min",
    "smooth_max",
    # spectral (principal-value) functions of a stress
    "DEFAULT_HEAVISIDE_AT_ZERO",
    "voigt_stress_to_tensor",
    "principal_values",
    "matrix_function",
    "spectral_trace",
    "positive_part",
    "negative_part",
    "positive_part_sq",
    "negative_part_sq",
    "macaulay_sq",
    "neg_macaulay_sq",
]
