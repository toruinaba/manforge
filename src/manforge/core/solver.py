"""Newton-Raphson solvers for return-mapping systems."""

import autograd
import autograd.numpy as anp
import numpy as np

from manforge.core.residual import (
    make_augmented_residual,
    make_reduced_residual,
    _flatten_state,
)


def _reduced_nr(model, stress_trial, C, state_n, max_iter, tol):
    """Newton-Raphson solver for the reduced (ntens+1) residual system."""
    ntens = model.ntens

    dlambda = anp.array(0.0)
    stress = anp.array(stress_trial)
    state_new = state_n

    residual_history = []
    n_iterations = 0

    for _iteration in range(max_iter):
        state_new = model.update_state(dlambda, stress, state_n)
        n = autograd.grad(lambda s: model.yield_function(s, state_new))(stress)
        stress = stress_trial - dlambda * (C @ n)
        state_new = model.update_state(dlambda, stress, state_n)
        f = model.yield_function(stress, state_new)
        residual_history.append(float(np.abs(float(f))))

        if abs(float(f)) < tol:
            break

        def _f_residual(dl, _stress=stress):
            st = model.update_state(dl, _stress, state_n)
            nn = autograd.grad(lambda s: model.yield_function(s, st))(_stress)
            s_upd = stress_trial - dl * (C @ nn)
            return model.yield_function(s_upd, st)

        dfddl = autograd.grad(_f_residual)(dlambda)
        dlambda = dlambda - f / dfddl
        n_iterations += 1

    else:
        raise RuntimeError(
            f"_reduced_nr: NR did not converge in {max_iter} iterations "
            f"(|f| = {float(np.abs(float(f))):.3e}, tol = {tol:.3e})"
        )

    return stress, state_new, dlambda, n_iterations, residual_history


def _augmented_nr(model, stress_trial, C, state_n, max_iter, tol):
    """Newton-Raphson solver for the augmented (ntens+1+n_state) residual system."""
    ntens = model.ntens
    residual_fn, n_state, unflatten_fn = make_augmented_residual(
        model, stress_trial, C, state_n
    )

    flat_state_n, _ = _flatten_state(state_n)
    x = anp.concatenate([anp.array(stress_trial), anp.array([0.0]), flat_state_n])

    residual_history = []
    n_iterations = 0
    for _iteration in range(max_iter):
        R = residual_fn(x)
        res_norm = float(np.max(np.abs(np.array(R))))
        residual_history.append(res_norm)
        if res_norm < tol:
            break
        J = autograd.jacobian(residual_fn)(x)
        dx = np.linalg.solve(np.array(J), np.array(R))
        x = x - anp.array(dx)
        n_iterations += 1
    else:
        raise RuntimeError(
            f"_augmented_nr: NR did not converge in {max_iter} iterations "
            f"(||R||_inf = {float(np.max(np.abs(np.array(R)))):.3e}, tol = {tol:.3e})"
        )

    stress = x[:ntens]
    dlambda = float(x[ntens])
    state_new = unflatten_fn(x[ntens + 1:])
    return stress, state_new, anp.array(dlambda), n_iterations, residual_history


def _select_nr(model):
    """Return the NR solver function appropriate for *model*."""
    if model.hardening_type == "augmented":
        return _augmented_nr
    return _reduced_nr
