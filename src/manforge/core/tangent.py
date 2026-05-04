"""Consistent (algorithmic) tangent computation via implicit differentiation."""

import autograd
import autograd.numpy as anp
import numpy as np

from manforge.core.residual import (
    make_augmented_residual,
    make_reduced_residual,
    _flatten_state,
)


def consistent_tangent(model, stress, state, dlambda, stress_n, state_n):
    """Compute the consistent (algorithmic) tangent dσ_{n+1}/dΔε (reduced path)."""
    ntens = model.ntens
    C_n = model.elastic_stiffness(state_n)
    C_conv = model.elastic_stiffness(state)

    n_conv = autograd.grad(lambda s: model.yield_function(s, state))(stress)
    # Reconstruct stress_trial from the converged state equation:
    # stress = stress_trial - dlambda * C(state_new) @ n  →  stress_trial = stress + dlambda * C_conv @ n
    stress_trial = stress + float(dlambda) * (C_conv @ n_conv)

    residual_fn = make_reduced_residual(model, stress_trial, state_n)

    x_conv = anp.concatenate([anp.array(stress), anp.array([float(dlambda)])])
    A = autograd.jacobian(residual_fn)(x_conv)  # (ntens+1, ntens+1)

    rhs = np.vstack([np.array(C_n), np.zeros((1, ntens))])  # (ntens+1, ntens)
    dxde = np.linalg.solve(np.array(A), rhs)                # (ntens+1, ntens)

    return anp.array(dxde[:ntens, :])


def augmented_consistent_tangent(model, stress, state, dlambda, stress_n, state_n):
    """Consistent tangent for implicit-state models (augmented residual system)."""
    ntens = model.ntens
    C_n = model.elastic_stiffness(state_n)
    C_conv = model.elastic_stiffness(state)

    n_conv = autograd.grad(lambda s: model.yield_function(s, state))(stress)
    stress_trial = stress + float(dlambda) * (C_conv @ n_conv)

    residual_fn, n_state, _ = make_augmented_residual(model, stress_trial, state_n)

    flat_state, _ = _flatten_state(state)
    x_conv = anp.concatenate([
        anp.array(stress),
        anp.array([float(dlambda)]),
        flat_state,
    ])

    A = autograd.jacobian(residual_fn)(x_conv)

    total = ntens + 1 + n_state
    rhs = np.vstack([np.array(C_n), np.zeros((1 + n_state, ntens))])  # (total, ntens)
    dxde = np.linalg.solve(np.array(A), rhs)                        # (total, ntens)

    return anp.array(dxde[:ntens, :])


def _select_tangent(model):
    """Return the consistent-tangent function appropriate for *model*."""
    if model.hardening_type == "augmented":
        return augmented_consistent_tangent
    return consistent_tangent
