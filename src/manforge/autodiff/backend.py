"""Autograd backend utilities."""

import autograd.numpy as anp  # noqa: F401  (re-exported for convenience)

__all__ = ["anp", "check_float64"]


def check_float64() -> None:
    """Verify that numpy is operating in float64 (always true by default)."""
    if anp.zeros(1).dtype != anp.float64:
        raise RuntimeError(
            "numpy float64 is not the default dtype. "
            "This should not occur in a standard numpy installation."
        )
