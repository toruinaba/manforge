"""Augmented residual system for implicit hardening laws.

When a model overrides ``hardening_residual`` (i.e. ``uses_implicit_state``
is True), the return-mapping residual is extended from (ntens+1) to
(ntens+1+n_state) equations by treating the state variables as independent
unknowns:

    x = [σ  (ntens),  Δλ (1),  q_flat (n_state)]

    R(x) = [ R_stress ]  ntens equations
            [ R_yield  ]  1 equation
            [ R_state  ]  n_state equations

where

    R_stress = σ - σ_trial + Δλ C n(σ, q)
    R_yield  = f(σ, q, params)
    R_state  = flatten(model.hardening_residual(q, Δλ, σ, q_n, params))

This module provides a factory function that builds the residual callable.
The same callable is used by both the Newton-Raphson solver
(:mod:`manforge.core.return_mapping`) and the consistent tangent computation
(:mod:`manforge.core.tangent`).
"""

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def make_augmented_residual(model, stress_trial, C, state_n, params):
    """Build the augmented residual function for implicit-state models.

    Parameters
    ----------
    model : MaterialModel
        Constitutive model with ``uses_implicit_state == True``.
    stress_trial : jnp.ndarray, shape (ntens,)
        Elastic trial stress σ_trial = σ_n + C Δε.
    C : jnp.ndarray, shape (ntens, ntens)
        Elastic stiffness tensor.
    state_n : dict
        Internal state at the beginning of the increment.
    params : dict
        Material parameters.

    Returns
    -------
    residual_fn : callable
        ``residual_fn(x) -> jnp.ndarray`` of shape ``(ntens+1+n_state,)``.
    n_state : int
        Number of scalar state DOFs (length of the flattened state vector).
    unflatten_fn : callable
        ``unflatten_fn(q_flat) -> dict`` reconstructs the state dict from a
        flat 1-D array of length ``n_state``.
    """
    ntens = model.ntens

    # Establish the flatten/unflatten pair from state_n so that ordering
    # (alphabetical by key, JAX convention) is fixed for this increment.
    flat_state_n, unflatten_fn = ravel_pytree(state_n)
    n_state = flat_state_n.shape[0]

    def residual_fn(x):
        sig = x[:ntens]
        dlambda = x[ntens]
        q_flat = x[ntens + 1 :]
        state_new = unflatten_fn(q_flat)

        # Flow direction n = ∂f/∂σ at (σ, q)
        n = jax.grad(lambda s: model.yield_function(s, state_new, params))(sig)

        # Stress residual: R1 = σ - σ_trial + Δλ C n
        R_stress = sig - stress_trial + dlambda * (C @ n)

        # Yield residual: R2 = f(σ, q)
        R_yield = model.yield_function(sig, state_new, params)

        # State residual: R_h = hardening_residual(q_{n+1}, Δλ, σ, q_n, params)
        R_state_dict = model.hardening_residual(
            state_new, dlambda, sig, state_n, params
        )
        R_state_flat, _ = ravel_pytree(R_state_dict)

        return jnp.concatenate([R_stress, R_yield.reshape(1), R_state_flat])

    return residual_fn, n_state, unflatten_fn
