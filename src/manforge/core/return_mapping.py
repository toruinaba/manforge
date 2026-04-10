"""Generic return mapping solver (closest point projection, associative flow).

Algorithm
---------
1. Elastic trial:  σ_trial = σ_n + C Δε
2. Yield check:    f_trial = f(σ_trial, state_n)
   → elastic if f_trial ≤ 0
3. Plastic correction — scalar NR on Δλ:
     state_new = hardening_increment(Δλ, state_n)
     n         = ∂f/∂σ  at (σ, state_new)              [jax.grad]
     σ         = σ_trial − Δλ C n
     f         = yield_function(σ, state_new)
     h         = d/dΔλ f(σ, state(Δλ))                 [jax.grad]
     dg/dΔλ    = −n^T C n + h
     Δλ        ← Δλ − f / (dg/dΔλ)
4. Consistent tangent via implicit differentiation (see core/tangent.py)

Notes
-----
- Python if/break is used for the elastic check and NR convergence.
  jax.jit compatibility is left as a TODO.
- For J2 with linear isotropic hardening the NR converges in a single step
  (the classic radial-return closed form).
"""

import jax
import jax.numpy as jnp

from manforge.core.tangent import consistent_tangent


def return_mapping(
    model,
    strain_inc: jnp.ndarray,
    stress_n: jnp.ndarray,
    state_n: dict,
    params: dict,
    max_iter: int = 50,
    tol: float = 1e-10,
):
    """Perform one load increment via closest point projection.

    Parameters
    ----------
    model : MaterialModel
        Constitutive model instance.
    strain_inc : jnp.ndarray, shape (ntens,)
        Strain increment Δε (engineering shear convention).
    stress_n : jnp.ndarray, shape (ntens,)
        Stress at the beginning of the increment.
    state_n : dict
        Internal state at the beginning of the increment.
    params : dict
        Material parameters.
    max_iter : int, optional
        Maximum Newton-Raphson iterations (default 50).
    tol : float, optional
        Absolute convergence tolerance on |f| (default 1e-10).

    Returns
    -------
    stress_new : jnp.ndarray, shape (ntens,)
        Updated stress σ_{n+1}.
    state_new : dict
        Updated internal state.
    ddsdde : jnp.ndarray, shape (ntens, ntens)
        Consistent tangent dσ_{n+1}/dΔε.

    Raises
    ------
    RuntimeError
        If the NR iteration does not converge within ``max_iter`` steps.
    """
    # ------------------------------------------------------------------
    # Step 1 — elastic stiffness and trial stress
    # ------------------------------------------------------------------
    C = model.elastic_stiffness(params)
    stress_trial = stress_n + C @ strain_inc

    # ------------------------------------------------------------------
    # Step 2 — elastic check
    # TODO: replace Python if with jax.lax.cond for jit compatibility
    # ------------------------------------------------------------------
    f_trial = model.yield_function(stress_trial, state_n, params)

    if f_trial <= 0.0:
        return stress_trial, state_n, C

    # ------------------------------------------------------------------
    # Step 3 — plastic correction: scalar NR on Δλ
    # ------------------------------------------------------------------
    dlambda = jnp.array(0.0)
    stress = stress_trial
    state_new = state_n

    for iteration in range(max_iter):
        # Update state and flow direction
        state_new = model.hardening_increment(dlambda, state_n, params)
        n = jax.grad(lambda s: model.yield_function(s, state_new, params))(stress)

        # Stress correction (radial return for fixed n)
        stress = stress_trial - dlambda * (C @ n)

        # Re-evaluate after stress update
        state_new = model.hardening_increment(dlambda, state_n, params)
        f = model.yield_function(stress, state_new, params)

        if jnp.abs(f) < tol:
            break

        # df/dΔλ  (total derivative, freezing n direction at current stress)
        #   = −n^T C n  +  h
        # where h = (∂f/∂state)(∂state/∂Δλ)  computed via jax.grad
        def _f_residual(dl, _stress=stress):
            st = model.hardening_increment(dl, state_n, params)
            nn = jax.grad(lambda s: model.yield_function(s, st, params))(_stress)
            s_upd = stress_trial - dl * (C @ nn)
            return model.yield_function(s_upd, st, params)

        dfddl = jax.grad(_f_residual)(dlambda)

        dlambda = dlambda - f / dfddl

    else:
        raise RuntimeError(
            f"return_mapping: NR did not converge in {max_iter} iterations "
            f"(|f| = {float(jnp.abs(f)):.3e}, tol = {tol:.3e})"
        )

    # ------------------------------------------------------------------
    # Step 4 — consistent tangent (implicit differentiation)
    # ------------------------------------------------------------------
    ddsdde = consistent_tangent(
        model, stress, state_new, dlambda, stress_n, state_n, params
    )

    return stress, state_new, ddsdde
