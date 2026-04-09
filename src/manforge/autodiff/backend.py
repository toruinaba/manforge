"""JAX backend configuration and utilities."""

import jax
import jax.numpy as jnp  # noqa: F401  (re-exported for convenience)

__all__ = ["jnp", "check_float64"]


def check_float64() -> None:
    """Verify that JAX float64 mode is active.

    Raises
    ------
    RuntimeError
        If jax_enable_x64 has not been set before importing JAX arrays.
    """
    if jnp.zeros(1).dtype != jnp.float64:
        raise RuntimeError(
            "JAX float64 is not enabled. "
            "Ensure 'import manforge' is executed before any jax imports, "
            "or call jax.config.update('jax_enable_x64', True) at startup."
        )
