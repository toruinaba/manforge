"""Residual builders for return-mapping systems.

Two factory functions are provided:

- :func:`make_explicit_residual` — for ``hardening_type == 'explicit'``.
  Builds the (ntens+1) residual ``[R_stress, R_yield]`` with x = [σ, Δλ].

- :func:`make_augmented_residual` — for ``hardening_type == 'implicit'``.
  Extends the system to (ntens+1+n_state) equations by treating state
  variables as additional unknowns:

      x = [σ  (ntens),  Δλ (1),  q_flat (n_state)]

      R(x) = [ R_stress ]  ntens equations
              [ R_yield  ]  1 equation
              [ R_state  ]  n_state equations

  where

      R_stress = σ - σ_trial + Δλ C n(σ, q)
      R_yield  = f(σ, q)
      R_state  = flatten(model.hardening_residual(q, Δλ, σ, q_n))

Both callables are used by the Newton-Raphson solver
(:mod:`manforge.core.solver`) and the consistent tangent computation
(:mod:`manforge.core.tangent`).
"""

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def make_explicit_residual(model, stress_trial, C, state_n):
    """Build the residual function for explicit-state models.

    Parameters
    ----------
    model : MaterialModel
        Constitutive model with ``hardening_type == 'explicit'``.
    stress_trial : jnp.ndarray, shape (ntens,)
        Elastic trial stress σ_trial = σ_n + C Δε.
    C : jnp.ndarray, shape (ntens, ntens)
        Elastic stiffness tensor.
    state_n : dict
        Internal state at the beginning of the increment.

    Returns
    -------
    residual_fn : callable
        ``residual_fn(x) -> jnp.ndarray`` of shape ``(ntens+1,)``
        where x = [σ (ntens), Δλ (1)].
    """
    ntens = model.ntens

    def residual_fn(x):
        sig = x[:ntens]
        dl = x[ntens]
        st = model.hardening_increment(dl, sig, state_n)
        n = jax.grad(lambda s: model.yield_function(s, st))(sig)
        R_stress = sig - stress_trial + dl * (C @ n)
        R_yield = model.yield_function(sig, st)
        return jnp.concatenate([R_stress, R_yield.reshape(1)])

    return residual_fn


def make_augmented_residual(model, stress_trial, C, state_n):
    """Build the augmented residual function for implicit-state models.

    Parameters
    ----------
    model : MaterialModel
        Constitutive model with ``hardening_type == 'implicit'``.
    stress_trial : jnp.ndarray, shape (ntens,)
        Elastic trial stress σ_trial = σ_n + C Δε.
    C : jnp.ndarray, shape (ntens, ntens)
        Elastic stiffness tensor.
    state_n : dict
        Internal state at the beginning of the increment.

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
        n = jax.grad(lambda s: model.yield_function(s, state_new))(sig)

        # Stress residual: R1 = σ - σ_trial + Δλ C n
        R_stress = sig - stress_trial + dlambda * (C @ n)

        # Yield residual: R2 = f(σ, q)
        R_yield = model.yield_function(sig, state_new)

        # State residual: R_h = hardening_residual(q_{n+1}, Δλ, σ, q_n)
        R_state_dict = model.hardening_residual(state_new, dlambda, sig, state_n)
        R_state_flat, _ = ravel_pytree(R_state_dict)

        return jnp.concatenate([R_stress, R_yield.reshape(1), R_state_flat])

    return residual_fn, n_state, unflatten_fn


def select_residual_builder(model):
    """Return the appropriate residual builder for *model*.

    Returns :func:`make_explicit_residual` for ``hardening_type == 'explicit'``
    and :func:`make_augmented_residual` for ``hardening_type == 'implicit'``.
    """
    if model.hardening_type == "implicit":
        return make_augmented_residual
    return make_explicit_residual
