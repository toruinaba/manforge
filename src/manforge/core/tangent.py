"""Consistent (algorithmic) tangent computation via implicit differentiation.

At the converged return-mapping point the residual system is

    R1 = σ - σ_trial + Δλ C n(σ, state(Δλ, σ)) = 0    (ntens equations)
    R2 = f(σ, state(Δλ, σ), params)              = 0    (1 equation)

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


def consistent_tangent(model, stress, state, dlambda, stress_n, state_n, params):
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
        Stress at step n (used only to identify σ_trial = σ_n + C Δε,
        not needed here — kept for API symmetry).
    state_n : dict
        Internal state at step n (needed to evaluate ∂state/∂(Δλ, σ)).
    params : dict
        Material parameters.

    Returns
    -------
    jnp.ndarray, shape (ntens, ntens)
        Consistent tangent dσ_{n+1}/dΔε.
    """
    ntens = model.ntens
    C = model.elastic_stiffness(params)

    # Reconstruct σ_trial from converged quantities.
    # At convergence R1 = 0  ⟹  σ_trial = σ + Δλ C n.
    n_conv = jax.grad(lambda s: model.yield_function(s, state, params))(stress)
    stress_trial = stress + dlambda * (C @ n_conv)

    # --- Full coupled residual as function of x = [σ (ntens), Δλ (1)] ---
    def _residual_vec(x):
        sig = x[:ntens]
        dl = x[ntens]
        st = model.hardening_increment(dl, sig, state_n, params)
        nn = jax.grad(lambda s: model.yield_function(s, st, params))(sig)
        R1 = sig - stress_trial + dl * (C @ nn)
        R2 = model.yield_function(sig, st, params)
        return jnp.concatenate([R1, R2.reshape(1)])

    # ∂R/∂x  — JAX computes all coupling terms automatically, including
    # ∂(state)/∂σ that arises with stress-dependent hardening increments.
    x_conv = jnp.concatenate([stress, dlambda.reshape(1)])
    A = jax.jacobian(_residual_vec)(x_conv)  # (ntens+1, ntens+1)

    # ∂R/∂ε  →  right-hand side  (−∂R/∂ε = [C; 0])
    rhs = jnp.vstack([C, jnp.zeros((1, ntens))])  # (ntens+1, ntens)

    # Solve:  dx/dε = A^{-1} rhs
    dxde = jnp.linalg.solve(A, rhs)  # (ntens+1, ntens)

    # Extract dσ/dε (top ntens rows)
    return dxde[:ntens, :]
