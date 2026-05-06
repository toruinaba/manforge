"""Newton-Raphson solver for return-mapping systems."""

import autograd
import autograd.numpy as anp
import numpy as np

from manforge.core.residual import (
    make_nr_residual,
    _flatten_state,
    _wrap_state,
    _call_update_state,
    _call_state_residual,
)
from manforge.core.state import _state_with_stress as _state_with_stress_fn


def _numerical_newton(model, stress_trial, state_n, max_iter, tol,
                      raise_on_nonconverged=True):
    """Unified Newton-Raphson solver for return mapping.

    Dispatches automatically based on ``model.state_fields["stress"].kind``
    and ``model.implicit_state_names``:

    * stress is Explicit, ``implicit_state_names=[]``  →  scalar NR on Δλ
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
    """Scalar NR on Δλ (stress=Explicit, no implicit non-stress states)."""
    from manforge.core.material import MaterialModel as _MaterialModel
    dlambda = anp.array(0.0)
    stress = anp.array(stress_trial)
    state_new = state_n

    residual_history = []
    n_iterations = 0
    f = anp.array(0.0)

    model_name = type(model).__name__
    explicit_keys_non_stress = set(
        k for k in model.state_names
        if k != "stress" and k not in model.implicit_state_names
    )
    user_has_state_residual = (
        type(model).state_residual is not _MaterialModel.state_residual
    )

    def _eval_R_dl(dl, state_new_dict):
        """Return R_dl: user override if present, else yield_function."""
        if user_has_state_residual:
            # state_trial["stress"] = current stress for the Explicit case
            trial_dict = dict(state_new_dict)
            _, r_dl = _call_state_residual(
                model, state_new_dict, dl, state_n, trial_dict,
                set(), model_name  # no implicit non-stress keys in scalar-NR
            )
            if r_dl is not None:
                return r_dl
        return model.yield_function(_wrap_state(state_new_dict, model))

    for _iteration in range(max_iter):
        # Update explicit non-stress states
        if explicit_keys_non_stress:
            state_trial_dict = dict(state_n)
            state_trial_dict["stress"] = stress
            q_exp = _call_update_state(
                model, dlambda, state_n, state_trial_dict,
                explicit_keys_non_stress, model_name
            )
            stress = q_exp["stress"]
            state_new = {"stress": stress,
                         **{k: v for k, v in q_exp.items() if k != "stress"}}
        else:
            state_new = {"stress": stress}

        # Compute flow direction and update stress
        state_new_state = _wrap_state(state_new, model)
        C_new = model.elastic_stiffness(state_new_state)
        n = autograd.grad(
            lambda s: model.yield_function(_state_with_stress_fn(state_new_state, s))
        )(stress)
        stress = anp.array(stress_trial) - dlambda * (C_new @ n)

        # Re-evaluate with updated stress
        if explicit_keys_non_stress:
            state_trial_dict = dict(state_n)
            state_trial_dict["stress"] = stress
            q_exp = _call_update_state(
                model, dlambda, state_n, state_trial_dict,
                explicit_keys_non_stress, model_name
            )
            stress = q_exp["stress"]
            state_new = {"stress": stress,
                         **{k: v for k, v in q_exp.items() if k != "stress"}}
        else:
            state_new = {"stress": stress}

        f = _eval_R_dl(dlambda, state_new)
        residual_history.append(float(np.abs(float(f))))

        if abs(float(f)) < tol:
            return stress, state_new, dlambda, n_iterations, residual_history, True

        def _f_residual(dl, _stress=stress):
            trial_dict = dict(state_n)
            trial_dict["stress"] = _stress
            if explicit_keys_non_stress:
                q_st = _call_update_state(
                    model, dl, state_n, trial_dict, explicit_keys_non_stress, model_name
                )
                _stress2 = q_st["stress"]
                st = {"stress": _stress2, **{k: v for k, v in q_st.items() if k != "stress"}}
            else:
                st = {"stress": _stress}
            st_state = _wrap_state(st, model)
            nn = autograd.grad(
                lambda s: model.yield_function(_state_with_stress_fn(st_state, s))
            )(_stress)
            s_upd = anp.array(stress_trial) - dl * (model.elastic_stiffness(st_state) @ nn)
            st2 = {"stress": s_upd, **{k: v for k, v in st.items() if k != "stress"}}
            return _eval_R_dl(dl, st2)

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
    implicit_keys_non_stress = sorted([k for k in model.implicit_state_names if k != "stress"])
    explicit_keys_non_stress = set(
        k for k in model.state_names
        if k != "stress" and k not in model.implicit_state_names
    )

    # Build initial guess x.
    implicit_state_n = {k: state_n[k] for k in implicit_keys_non_stress}
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

    dlambda_val = x[_dl_idx]
    model_name = type(model).__name__

    if do_implicit_stress:
        stress = x[:ntens]
    else:
        q_imp = unflatten_implicit(x[_q_sl]) if n_implicit > 0 else {}
        if explicit_keys_non_stress:
            # Compute σ from final Δλ and implicit state
            q_approx = {"stress": anp.array(stress_trial), **q_imp}
            q_approx.update({k: state_n[k] for k in explicit_keys_non_stress})
            q_approx_state = _wrap_state(q_approx, model)
            C = model.elastic_stiffness(q_approx_state)
            n_dir = autograd.grad(
                lambda s: model.yield_function(_state_with_stress_fn(q_approx_state, s))
            )(anp.array(stress_trial))
            stress = anp.array(stress_trial) - dlambda_val * (C @ n_dir)
            # Try to get explicit state
            state_trial_dict = dict(state_n)
            state_trial_dict["stress"] = stress
            q_exp = _call_update_state(
                model, dlambda_val, state_n, state_trial_dict,
                explicit_keys_non_stress, model_name
            )
            stress = q_exp["stress"]
        else:
            q_imp_full = {"stress": anp.array(stress_trial), **q_imp}
            q_imp_state = _wrap_state(q_imp_full, model)
            C = model.elastic_stiffness(q_imp_state)
            n_dir = autograd.grad(
                lambda s: model.yield_function(_state_with_stress_fn(q_imp_state, s))
            )(anp.array(stress_trial))
            stress = anp.array(stress_trial) - dlambda_val * (C @ n_dir)

    # Final state extraction
    q_imp = unflatten_implicit(x[_q_sl]) if n_implicit > 0 else {}
    if explicit_keys_non_stress:
        state_trial_dict = dict(state_n)
        state_trial_dict["stress"] = stress
        q_exp = _call_update_state(
            model, dlambda_val, state_n, state_trial_dict,
            explicit_keys_non_stress, model_name,
            require_stress=(not do_implicit_stress),
        )
        if not do_implicit_stress:
            stress = q_exp["stress"]
        state_new = {"stress": stress,
                     **q_imp,
                     **{k: v for k, v in q_exp.items() if k != "stress"}}
    else:
        state_new = {"stress": stress, **q_imp} if q_imp else {"stress": stress}

    return stress, state_new, anp.array(dlambda_val), n_iterations, residual_history, converged
