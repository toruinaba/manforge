"""Consistent (algorithmic) tangent computation via implicit differentiation."""

import autograd
import autograd.numpy as anp
import numpy as np

from manforge.core.residual import make_tangent_residual, _flatten_state, _wrap_state
from manforge.core.state import _state_with_stress


def consistent_tangent(model, stress, state, dlambda, stress_n, state_n):
    """Compute the consistent (algorithmic) tangent dσ_{n+1}/dΔε.

    σ is always included as an independent variable in the linear system.
    The linear system solved is::

        A · dx/dε = rhs
        A = ∂R/∂x  (ntens + 1 + n_implicit) × (ntens + 1 + n_implicit)
        rhs[:,j] = [C_n[:,j], 0, ..., 0]
        dσ/dε = dx[:ntens, :]

    Parameters
    ----------
    model : MaterialModel
    stress : array, shape (ntens,)
        Converged stress σ_{n+1}.
    state : dict
        Converged state at step n+1.
    dlambda : scalar
        Converged Δλ.
    stress_n : array
        Stress at increment start (unused, kept for API compatibility).
    state_n : dict
        State at increment start.

    Returns
    -------
    anp.ndarray, shape (ntens, ntens)
    """
    ntens = model.ntens
    state_with_stress = dict(state)
    state_with_stress["stress"] = stress
    state_full = _wrap_state(state_with_stress, model)

    C_n = model.elastic_stiffness(state_n)
    C_conv = model.elastic_stiffness(state_full)

    n_conv = autograd.grad(
        lambda s: model.yield_function(_state_with_stress(state_full, s))
    )(anp.array(stress))
    stress_trial = anp.array(stress) + float(dlambda) * (C_conv @ n_conv)

    # state_n must include "stress" for tangent residual
    state_n_full = dict(state_n)
    if "stress" not in state_n_full:
        state_n_full["stress"] = anp.zeros(ntens)

    residual_fn, n_implicit, _ = make_tangent_residual(model, stress_trial, state_n_full)

    implicit_keys_non_stress = sorted([k for k in model.implicit_state_names if k != "stress"])
    implicit_state = {k: state[k] for k in implicit_keys_non_stress}
    flat_impl, _ = _flatten_state(implicit_state)

    x_conv = anp.concatenate([
        anp.array(stress),
        anp.array([float(dlambda)]),
        flat_impl,
    ])

    A = autograd.jacobian(residual_fn)(x_conv)
    rhs = np.vstack([np.array(C_n), np.zeros((1 + n_implicit, ntens))])
    dxde = np.linalg.solve(np.array(A), rhs)

    return anp.array(dxde[:ntens, :])
