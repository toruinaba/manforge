"""Newton-Raphson solver for return-mapping systems."""

import autograd
import autograd.numpy as anp
import numpy as np

from manforge.core.residual import (
    build_residual,
    _flatten_state,
    _wrap_state,
    _call_update_state,
)


def _numerical_newton(model, stress_trial, state_n, max_iter, tol,
                      raise_on_nonconverged=True):
    """Unified Newton-Raphson solver for return mapping.

    σ is always included as an independent variable.  Unknown vector layout::

        x = [σ (ntens)] + [Δλ (1)] + [q_implicit_non_stress (n_imp)]?

    Returns
    -------
    tuple
        ``(stress, state_new, dlambda, n_iterations, residual_history, converged)``
    """
    ntens = model.ntens
    implicit_keys_non_stress = sorted(k for k in model.implicit_state_names if k != "stress")
    explicit_keys_non_stress = set(
        k for k in model.state_names
        if k != "stress" and k not in model.implicit_state_names
    )
    do_implicit_stress = model.state_fields["stress"].kind == "implicit"

    residual_fn, n_unknown, unflatten_implicit = build_residual(model, stress_trial, state_n)
    n_implicit = n_unknown - ntens - 1

    implicit_state_n = {k: state_n[k] for k in implicit_keys_non_stress}
    flat_impl_n, _ = _flatten_state(implicit_state_n)
    x = anp.concatenate([anp.array(stress_trial), anp.array([0.0]), flat_impl_n])

    residual_history = []
    n_iterations = 0
    converged = False

    for _ in range(max_iter):
        R = residual_fn(x)
        norm = float(np.linalg.norm(np.array(R)))
        residual_history.append(norm)
        if norm < tol:
            converged = True
            break
        J = autograd.jacobian(residual_fn)(x)
        dx = np.linalg.solve(np.array(J), np.array(R))
        x = x - anp.array(dx)
        n_iterations += 1

    if not converged and raise_on_nonconverged:
        raise RuntimeError(
            f"_numerical_newton: NR did not converge in {max_iter} iterations "
            f"(||R||_2 = {float(np.linalg.norm(np.array(residual_fn(x)))):.3e}, tol = {tol:.3e})"
        )

    stress = x[:ntens]
    dlambda_val = x[ntens]
    q_imp = unflatten_implicit(x[ntens + 1:]) if n_implicit > 0 else {}
    model_name = type(model).__name__

    if explicit_keys_non_stress:
        state_trial_dict = dict(state_n)
        state_trial_dict["stress"] = stress
        q_exp = _call_update_state(
            model, dlambda_val, state_n, state_trial_dict,
            explicit_keys_non_stress, model_name,
            require_stress=(not do_implicit_stress),
        )
        if not do_implicit_stress and "stress" in q_exp:
            stress = q_exp["stress"]
        state_new = {"stress": stress, **q_imp,
                     **{k: v for k, v in q_exp.items() if k != "stress"}}
    else:
        state_new = {"stress": stress, **q_imp}

    return stress, state_new, anp.array(dlambda_val), n_iterations, residual_history, converged
