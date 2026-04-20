"""Consistent (algorithmic) tangent computation via implicit differentiation.

At the converged return-mapping point the residual system is

    R1 = σ - σ_trial + Δλ C n(σ, state(Δλ, σ)) = 0    (ntens equations)
    R2 = f(σ, state(Δλ, σ))                      = 0    (1 equation)

where state may depend on both Δλ and σ (e.g. kinematic hardening with
backstress updated via the flow direction).

Differentiating implicitly with respect to Δε (where σ_trial = σ_n + C Δε):

    ∂R/∂x · dx/dε = −∂R/∂ε

    ∂R/∂ε = ┌ −C ┐   (ntens+1, ntens)
             └  0 ┘

    dσ/dε  = first ntens rows of  (∂R/∂x)^{−1} C

The Jacobian ∂R/∂x is computed automatically by JAX, capturing all
cross-coupling terms when state depends on σ (as in kinematic hardening).
"""

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from manforge.core.residual import make_augmented_residual, make_explicit_residual


def consistent_tangent(model, stress, state, dlambda, stress_n, state_n):
    """Compute the consistent (algorithmic) tangent dσ_{n+1}/dΔε.

    Uses implicit differentiation at the converged return-mapping point;
    does **not** differentiate through the Newton-Raphson iteration.

    Parameters
    ----------
    model : MaterialModel
        Constitutive model instance.
    stress : jnp.ndarray, shape (ntens,)
        Converged stress σ_{n+1}.
    state : dict
        Converged internal state at step n+1.
    dlambda : jnp.ndarray, scalar
        Converged plastic multiplier increment Δλ.
    stress_n : jnp.ndarray, shape (ntens,)
        Stress at step n (kept for API symmetry).
    state_n : dict
        Internal state at step n (needed to evaluate ∂state/∂(Δλ, σ)).

    Returns
    -------
    jnp.ndarray, shape (ntens, ntens)
        Consistent tangent dσ_{n+1}/dΔε.
    """
    ntens = model.ntens
    C = model.elastic_stiffness()

    # Reconstruct σ_trial from converged quantities.
    # At convergence R1 = 0  ⟹  σ_trial = σ + Δλ C n.
    n_conv = jax.grad(lambda s: model.yield_function(s, state))(stress)
    stress_trial = stress + dlambda * (C @ n_conv)

    residual_fn = make_explicit_residual(model, stress_trial, C, state_n)

    # ∂R/∂x  — JAX computes all coupling terms automatically, including
    # ∂(state)/∂σ that arises with stress-dependent hardening increments.
    x_conv = jnp.concatenate([stress, dlambda.reshape(1)])
    A = jax.jacobian(residual_fn)(x_conv)  # (ntens+1, ntens+1)

    # ∂R/∂ε  →  right-hand side  (−∂R/∂ε = [C; 0])
    rhs = jnp.vstack([C, jnp.zeros((1, ntens))])  # (ntens+1, ntens)

    # Solve:  dx/dε = A^{-1} rhs
    dxde = jnp.linalg.solve(A, rhs)  # (ntens+1, ntens)

    # Extract dσ/dε (top ntens rows)
    return dxde[:ntens, :]


def augmented_consistent_tangent(model, stress, state, dlambda, stress_n, state_n):
    """Consistent tangent for implicit-state models (augmented residual system).

    The residual system is (ntens+1+n_state) equations in
    (ntens+1+n_state) unknowns x = [σ, Δλ, q_flat].  Implicit
    differentiation at the converged point gives dσ/dε.

    Parameters
    ----------
    model : MaterialModel
        Model with ``hardening_type == 'implicit'``.
    stress : jnp.ndarray, shape (ntens,)
        Converged stress σ_{n+1}.
    state : dict
        Converged internal state at step n+1.
    dlambda : jnp.ndarray, scalar
        Converged plastic multiplier increment Δλ.
    stress_n : jnp.ndarray, shape (ntens,)
        Stress at step n (used only to reconstruct σ_trial).
    state_n : dict
        Internal state at step n.

    Returns
    -------
    jnp.ndarray, shape (ntens, ntens)
        Consistent tangent dσ_{n+1}/dΔε.
    """
    ntens = model.ntens
    C = model.elastic_stiffness()

    # Reconstruct σ_trial from converged quantities (R1 = 0 at convergence).
    n_conv = jax.grad(lambda s: model.yield_function(s, state))(stress)
    stress_trial = stress + dlambda * (C @ n_conv)

    residual_fn, n_state, _ = make_augmented_residual(model, stress_trial, C, state_n)

    # Converged augmented unknown vector
    flat_state, _ = ravel_pytree(state)
    x_conv = jnp.concatenate([stress, dlambda.reshape(1), flat_state])

    # ∂R/∂x — full (ntens+1+n_state) × (ntens+1+n_state) Jacobian
    A = jax.jacobian(residual_fn)(x_conv)

    # ∂R/∂ε → RHS = [C; 0]  (shape: ntens+1+n_state, ntens)
    total = ntens + 1 + n_state
    rhs = jnp.vstack([C, jnp.zeros((1 + n_state, ntens))])  # (total, ntens)

    # Solve: dx/dε = A^{-1} rhs
    dxde = jnp.linalg.solve(A, rhs)  # (total, ntens)

    # Extract dσ/dε (top ntens rows)
    return dxde[:ntens, :]


def _select_tangent(model):
    """Return the consistent-tangent function appropriate for *model*."""
    if model.hardening_type == "implicit":
        return augmented_consistent_tangent
    return consistent_tangent
