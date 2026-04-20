"""Newton-Raphson solvers for return-mapping systems.

Two solvers are provided:

- :func:`_explicit_nr` — scalar NR on Δλ for ``hardening_type == 'explicit'``.
- :func:`_augmented_nr` — vector NR on [σ, Δλ, q_flat] for
  ``hardening_type == 'implicit'``.

Both return a 5-tuple ``(stress, state_new, dlambda, n_iterations,
residual_history)`` consumed by :func:`~manforge.core.stress_update.return_mapping`.
"""

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from manforge.core.residual import make_augmented_residual, make_explicit_residual


def _explicit_nr(model, stress_trial, C, state_n, max_iter, tol):
    """Newton-Raphson solver for the explicit (ntens+1) residual system.

    Scalar NR on Δλ: evaluates the full residual at each step and uses
    ``jax.grad`` to compute dR/dΔλ for the update.

    Parameters
    ----------
    model : MaterialModel
        Constitutive model with ``hardening_type == 'explicit'``.
    stress_trial : jnp.ndarray, shape (ntens,)
    C : jnp.ndarray, shape (ntens, ntens)
    state_n : dict
    max_iter : int
    tol : float

    Returns
    -------
    stress : jnp.ndarray, shape (ntens,)
    state_new : dict
    dlambda : jnp.ndarray, scalar
    n_iterations : int
    residual_history : list[float]
    """
    ntens = model.ntens

    dlambda = jnp.array(0.0)
    stress = stress_trial
    state_new = state_n

    residual_history = []
    n_iterations = 0

    for _iteration in range(max_iter):
        # Evaluate state and flow direction at current stress, then update stress,
        # then re-evaluate state at the new stress (two calls is intentional).
        state_new = model.hardening_increment(dlambda, stress, state_n)
        n = jax.grad(lambda s: model.yield_function(s, state_new))(stress)
        stress = stress_trial - dlambda * (C @ n)
        state_new = model.hardening_increment(dlambda, stress, state_n)
        f = model.yield_function(stress, state_new)
        residual_history.append(float(jnp.abs(f)))

        if jnp.abs(f) < tol:
            break

        def _f_residual(dl, _stress=stress):
            st = model.hardening_increment(dl, _stress, state_n)
            nn = jax.grad(lambda s: model.yield_function(s, st))(_stress)
            s_upd = stress_trial - dl * (C @ nn)
            return model.yield_function(s_upd, st)

        dfddl = jax.grad(_f_residual)(dlambda)
        dlambda = dlambda - f / dfddl
        n_iterations += 1

    else:
        raise RuntimeError(
            f"_explicit_nr: NR did not converge in {max_iter} iterations "
            f"(|f| = {float(jnp.abs(f)):.3e}, tol = {tol:.3e})"
        )

    return stress, state_new, dlambda, n_iterations, residual_history


def _augmented_nr(model, stress_trial, C, state_n, max_iter, tol):
    """Newton-Raphson solver for the augmented (ntens+1+n_state) residual system.

    Used when ``model.hardening_type == 'implicit'``.

    Parameters
    ----------
    model : MaterialModel
    stress_trial : jnp.ndarray, shape (ntens,)
    C : jnp.ndarray, shape (ntens, ntens)
    state_n : dict
    max_iter : int
    tol : float

    Returns
    -------
    stress : jnp.ndarray, shape (ntens,)
    state_new : dict
    dlambda : jnp.ndarray, scalar
    n_iterations : int
    residual_history : list[float]
    """
    ntens = model.ntens
    residual_fn, n_state, unflatten_fn = make_augmented_residual(
        model, stress_trial, C, state_n
    )

    flat_state_n, _ = ravel_pytree(state_n)
    x = jnp.concatenate([stress_trial, jnp.array([0.0]), flat_state_n])

    residual_history = []
    n_iterations = 0
    for _iteration in range(max_iter):
        R = residual_fn(x)
        res_norm = float(jnp.max(jnp.abs(R)))
        residual_history.append(res_norm)
        if res_norm < tol:
            break
        J = jax.jacobian(residual_fn)(x)
        dx = jnp.linalg.solve(J, R)
        x = x - dx
        n_iterations += 1
    else:
        raise RuntimeError(
            f"_augmented_nr: NR did not converge in {max_iter} iterations "
            f"(||R||_inf = {float(jnp.max(jnp.abs(R))):.3e}, tol = {tol:.3e})"
        )

    stress = x[:ntens]
    dlambda = jnp.asarray(x[ntens])
    state_new = unflatten_fn(x[ntens + 1:])
    return stress, state_new, dlambda, n_iterations, residual_history


def _select_nr(model):
    """Return the NR solver function appropriate for *model*."""
    if model.hardening_type == "implicit":
        return _augmented_nr
    return _explicit_nr
