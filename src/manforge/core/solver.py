"""Newton-Raphson solver for return-mapping systems."""

import autograd
import autograd.numpy as anp
import numpy as np

from manforge.core.residual import (
    make_nr_residual,
    _flatten_state,
    _normalise_update,
)


def _numerical_newton(model, stress_trial, state_n, max_iter, tol,
                      raise_on_nonconverged=True):
    """Unified Newton-Raphson solver for return mapping.

    Dispatches automatically based on ``model.implicit_state_names`` and
    ``model.implicit_stress``:

    * ``implicit_state_names=[]``, ``implicit_stress=False``  →  scalar NR on Δλ
      (uses ``autograd.grad`` for efficiency)
    * otherwise  →  vector NR on ``[σ?] + [Δλ] + [q_implicit?]``
      (uses ``autograd.jacobian`` + ``np.linalg.solve``)

    Returns
    -------
    tuple
        ``(stress, state_new, dlambda, n_iterations, residual_history, converged)``
    """
    residual_fn, unknowns_meta, unflatten_implicit = make_nr_residual(
        model, stress_trial, state_n
    )
    do_implicit_stress = unknowns_meta["implicit_stress"]
    n_implicit = unknowns_meta["n_implicit"]
    scalar_nr = (not do_implicit_stress) and (n_implicit == 0)

    if scalar_nr:
        return _scalar_nr(
            model, stress_trial, state_n, max_iter, tol, raise_on_nonconverged,
        )
    else:
        return _vector_nr(
            model, stress_trial, state_n, residual_fn, unknowns_meta,
            unflatten_implicit, max_iter, tol, raise_on_nonconverged,
        )


def _scalar_nr(model, stress_trial, state_n, max_iter, tol, raise_on_nonconverged):
    """Scalar NR on Δλ (implicit_state_names=[], implicit_stress=False)."""
    dlambda = anp.array(0.0)
    stress = anp.array(stress_trial)
    state_new = state_n

    residual_history = []
    n_iterations = 0
    f = anp.array(0.0)

    model_name = type(model).__name__
    explicit_keys = set(model.state_names)  # all explicit for scalar NR

    for _iteration in range(max_iter):
        state_new = _normalise_update(model.update_state(dlambda, stress, state_n), explicit_keys, model_name)
        C_new = model.elastic_stiffness(state_new)
        n = autograd.grad(lambda s: model.yield_function(s, state_new))(stress)
        stress = anp.array(stress_trial) - dlambda * (C_new @ n)
        state_new = _normalise_update(model.update_state(dlambda, stress, state_n), explicit_keys, model_name)
        f = model.yield_function(stress, state_new)
        residual_history.append(float(np.abs(float(f))))

        if abs(float(f)) < tol:
            return stress, state_new, dlambda, n_iterations, residual_history, True

        def _f_residual(dl, _stress=stress):
            st = _normalise_update(model.update_state(dl, _stress, state_n), explicit_keys, model_name)
            nn = autograd.grad(lambda s: model.yield_function(s, st))(_stress)
            s_upd = anp.array(stress_trial) - dl * (model.elastic_stiffness(st) @ nn)
            return model.yield_function(s_upd, st)

        dfddl = autograd.grad(_f_residual)(dlambda)
        dlambda = dlambda - f / dfddl
        n_iterations += 1

    if raise_on_nonconverged:
        raise RuntimeError(
            f"_scalar_nr: NR did not converge in {max_iter} iterations "
            f"(|f| = {float(np.abs(float(f))):.3e}, tol = {tol:.3e})"
        )
    return stress, state_new, dlambda, n_iterations, residual_history, False


def _vector_nr(model, stress_trial, state_n, residual_fn, unknowns_meta,
               unflatten_implicit, max_iter, tol, raise_on_nonconverged):
    """Vector NR on [σ?] + [Δλ] + [q_implicit?]."""
    ntens = unknowns_meta["ntens"]
    do_implicit_stress = unknowns_meta["implicit_stress"]
    n_implicit = unknowns_meta["n_implicit"]
    implicit_keys = sorted(model.implicit_state_names)
    explicit_keys = set(model.state_names) - set(implicit_keys)

    # Build initial guess x.
    implicit_state_n = {k: state_n[k] for k in implicit_keys}
    flat_impl_n, _ = _flatten_state(implicit_state_n)
    parts = []
    if do_implicit_stress:
        parts.append(anp.array(stress_trial))
    parts.append(anp.array([0.0]))  # Δλ
    if n_implicit > 0:
        parts.append(flat_impl_n)
    x = anp.concatenate(parts)

    if do_implicit_stress:
        _dl_idx = ntens
        _q_sl = slice(ntens + 1, ntens + 1 + n_implicit)
    else:
        _dl_idx = 0
        _q_sl = slice(1, 1 + n_implicit)

    residual_history = []
    n_iterations = 0
    R = residual_fn(x)
    converged = False

    for _iteration in range(max_iter):
        R = residual_fn(x)
        res_norm = float(np.linalg.norm(np.array(R)))
        residual_history.append(res_norm)
        if res_norm < tol:
            converged = True
            break
        J = autograd.jacobian(residual_fn)(x)
        dx = np.linalg.solve(np.array(J), np.array(R))
        x = x - anp.array(dx)
        n_iterations += 1

    if not converged and raise_on_nonconverged:
        raise RuntimeError(
            f"_vector_nr: NR did not converge in {max_iter} iterations "
            f"(||R||_2 = {float(np.linalg.norm(np.array(R))):.3e}, tol = {tol:.3e})"
        )

    # Extract results from x.
    dlambda_val = x[_dl_idx]
    vr_model_name = type(model).__name__

    if do_implicit_stress:
        stress = x[:ntens]
    else:
        q_imp = unflatten_implicit(x[_q_sl]) if n_implicit > 0 else {}
        if explicit_keys:
            q_exp = _normalise_update(
                model.update_state(dlambda_val, anp.array(stress_trial), state_n),
                explicit_keys, vr_model_name,
            )
            q_full = {**q_imp, **q_exp}
        else:
            q_full = q_imp if q_imp else dict(state_n)
        C = model.elastic_stiffness(q_full)
        n_dir = autograd.grad(lambda s: model.yield_function(s, q_full))(anp.array(stress_trial))
        stress = anp.array(stress_trial) - dlambda_val * (C @ n_dir)

    q_imp = unflatten_implicit(x[_q_sl]) if n_implicit > 0 else {}
    if explicit_keys:
        q_exp = _normalise_update(
            model.update_state(dlambda_val, stress, state_n), explicit_keys, vr_model_name,
        )
        state_new = {**q_imp, **q_exp}
    else:
        state_new = q_imp if q_imp else dict(state_n)

    return stress, state_new, anp.array(dlambda_val), n_iterations, residual_history, converged
