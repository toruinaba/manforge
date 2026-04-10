"""Consistent (algorithmic) tangent computation via implicit differentiation.

At the converged return-mapping point the residual system is

    R1 = σ - σ_trial + Δλ C n(σ, state) = 0    (ntens equations)
    R2 = f(σ, state(Δλ), params)         = 0    (1 equation)

Differentiating implicitly with respect to Δε (where σ_trial = σ_n + C Δε):

    ∂R/∂x · dx/dε = −∂R/∂ε

    ∂R/∂x = ┌ I + Δλ C ∂²f/∂σ²   C n ┐   (ntens+1, ntens+1)
             └       n^T             h  ┘

    ∂R/∂ε = ┌ −C ┐   (ntens+1, ntens)
             └  0 ┘

    dσ/dε  = first ntens rows of  (∂R/∂x)^{−1} C

where h = (∂f/∂state)·(∂state/∂Δλ)  (scalar hardening modulus derivative).
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
        Internal state at step n (needed to evaluate ∂state/∂Δλ).
    params : dict
        Material parameters.

    Returns
    -------
    jnp.ndarray, shape (ntens, ntens)
        Consistent tangent dσ_{n+1}/dΔε.
    """
    ntens = model.ntens
    C = model.elastic_stiffness(params)

    # --- Gradient quantities at the converged point ---

    # n = ∂f/∂σ  (flow direction, shape ntens)
    n = jax.grad(lambda s: model.yield_function(s, state, params))(stress)

    # ∂²f/∂σ²  (Hessian of yield function w.r.t. stress, shape ntens×ntens)
    H_f = jax.hessian(lambda s: model.yield_function(s, state, params))(stress)

    # h = d/dΔλ [ f(σ, state(Δλ), params) ]  (hardening stiffness, scalar)
    # = (∂f/∂state) · (∂state/∂Δλ)
    def _f_of_dl(dl):
        st = model.hardening_increment(dl, state_n, params)
        return model.yield_function(stress, st, params)

    h = jax.grad(_f_of_dl)(dlambda)

    # --- Assemble the 7×7 (ntens+1 × ntens+1) system ---
    # ∂R/∂x
    A_top_left = jnp.eye(ntens) + dlambda * C @ H_f   # (ntens, ntens)
    A_top_right = (C @ n).reshape(ntens, 1)            # (ntens, 1)
    A_bot_left = n.reshape(1, ntens)                    # (1, ntens)
    A_bot_right = jnp.array([[h]])                      # (1, 1)

    A = jnp.block([
        [A_top_left, A_top_right],
        [A_bot_left, A_bot_right],
    ])  # (ntens+1, ntens+1)

    # ∂R/∂ε  →  right-hand side  (−∂R/∂ε = [C; 0])
    rhs = jnp.vstack([C, jnp.zeros((1, ntens))])  # (ntens+1, ntens)

    # Solve:  dx/dε = A^{-1} rhs
    dxde = jnp.linalg.solve(A, rhs)  # (ntens+1, ntens)

    # Extract dσ/dε (top ntens rows)
    return dxde[:ntens, :]
