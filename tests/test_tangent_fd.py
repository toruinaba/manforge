"""Verify consistent tangent against central-difference finite differences.

For each test case, DDSDDE is computed two ways:
1. Analytically via consistent_tangent (implicit differentiation)
2. Numerically via central differences:
     DDSDDE_fd[i,j] ≈ (σ(Δε + h e_j)[i] - σ(Δε - h e_j)[i]) / (2h)

The two must agree to rtol = 1e-5.
"""

import numpy as np
import jax.numpy as jnp

from manforge.core.return_mapping import return_mapping


def _fd_tangent(model, strain_inc, stress_n, state_n, h=1e-7):
    """Compute DDSDDE by central differences."""
    ntens = model.ntens
    ddsdde_fd = jnp.zeros((ntens, ntens))

    for j in range(ntens):
        e_j = jnp.zeros(ntens).at[j].set(1.0)

        s_fwd, _, _ = return_mapping(
            model, strain_inc + h * e_j, stress_n, state_n
        )
        s_bwd, _, _ = return_mapping(
            model, strain_inc - h * e_j, stress_n, state_n
        )
        col = (s_fwd - s_bwd) / (2.0 * h)
        ddsdde_fd = ddsdde_fd.at[:, j].set(col)

    return ddsdde_fd


# ---------------------------------------------------------------------------
# Elastic domain
# ---------------------------------------------------------------------------

def test_tangent_fd_elastic(model):
    """Elastic domain: AD tangent == FD tangent."""
    strain_inc = jnp.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])
    stress_n = jnp.zeros(6)
    state_n = model.initial_state()

    _, _, ddsdde_ad = return_mapping(
        model, strain_inc, stress_n, state_n
    )
    ddsdde_fd = _fd_tangent(model, strain_inc, stress_n, state_n)

    np.testing.assert_allclose(
        np.asarray(ddsdde_ad), np.asarray(ddsdde_fd), rtol=1e-5, atol=1e-3,
        err_msg=f"Max rel err: {float(jnp.max(jnp.abs(ddsdde_ad - ddsdde_fd) / (jnp.abs(ddsdde_fd) + 1e-12))):.3e}",
    )


# ---------------------------------------------------------------------------
# Plastic domain (uniaxial)
# ---------------------------------------------------------------------------

def test_tangent_fd_plastic_uniaxial(model):
    """Plastic uniaxial domain: AD tangent == FD tangent."""
    strain_inc = jnp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    stress_n = jnp.zeros(6)
    state_n = model.initial_state()

    _, _, ddsdde_ad = return_mapping(
        model, strain_inc, stress_n, state_n
    )
    ddsdde_fd = _fd_tangent(model, strain_inc, stress_n, state_n)

    # Relative tolerance on the dominant non-zero entries
    denom = jnp.abs(ddsdde_fd) + 1e-2  # add small offset to avoid /0 on near-zero entries
    rel_err = jnp.max(jnp.abs(ddsdde_ad - ddsdde_fd) / denom)
    assert rel_err < 1e-5, f"Max rel err: {float(rel_err):.3e}"


# ---------------------------------------------------------------------------
# Plastic domain (multiaxial)
# ---------------------------------------------------------------------------

def test_tangent_fd_plastic_multiaxial(model):
    """Plastic multiaxial domain: AD tangent == FD tangent."""
    strain_inc = jnp.array([1.5e-3, -0.5e-3, -0.5e-3, 0.5e-3, 0.0, 0.0])
    stress_n = jnp.zeros(6)
    state_n = model.initial_state()

    _, _, ddsdde_ad = return_mapping(
        model, strain_inc, stress_n, state_n
    )
    ddsdde_fd = _fd_tangent(model, strain_inc, stress_n, state_n)

    denom = jnp.abs(ddsdde_fd) + 1e-2
    rel_err = jnp.max(jnp.abs(ddsdde_ad - ddsdde_fd) / denom)
    assert rel_err < 1e-5, f"Max rel err: {float(rel_err):.3e}"


# ---------------------------------------------------------------------------
# Plastic domain from non-zero pre-stress
# ---------------------------------------------------------------------------

def test_tangent_fd_plastic_prestress(model):
    """Plastic step from pre-stressed state: AD tangent == FD tangent."""
    sigma_y0 = model.sigma_y0
    stress_n = jnp.array([sigma_y0 / 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    state_n = model.initial_state()

    # Now push into plastic domain
    strain_inc = jnp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])

    _, _, ddsdde_ad = return_mapping(
        model, strain_inc, stress_n, state_n
    )
    ddsdde_fd = _fd_tangent(model, strain_inc, stress_n, state_n)

    denom = jnp.abs(ddsdde_fd) + 1e-2
    rel_err = jnp.max(jnp.abs(ddsdde_ad - ddsdde_fd) / denom)
    assert rel_err < 1e-5, f"Max rel err: {float(rel_err):.3e}"
