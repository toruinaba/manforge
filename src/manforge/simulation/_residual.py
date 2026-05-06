"""Residual builder for return-mapping systems."""

import autograd.numpy as anp
import numpy as np

from manforge.core.state import (
    StateResidual, StateUpdate, _validate_state_items, State,
)
from manforge.simulation._layout import ResidualLayout


# ---------------------------------------------------------------------------
# State wrapping helper
# ---------------------------------------------------------------------------

def _wrap_state(data: dict, model) -> State:
    """Wrap a plain dict in a State with the model's field ordering."""
    return State(data, tuple(model.state_names))


# ---------------------------------------------------------------------------
# Unified residual builder
# ---------------------------------------------------------------------------

def build_residual(model, stress_trial, state_n):
    """Build the unified NR/tangent residual function.

    Unknown / residual vector layout (declaration order, stress-first)::

        x = [σ (ntens)] + [Δλ (1)] + [q_implicit_non_stress (declaration order)]
        R = [R_σ (ntens)] + [R_Δλ (1)] + [R_q_non_stress (declaration order)]

    σ is always an independent variable.  This function serves both the NR
    phase (iterated until convergence) and the tangent/Jacobian phase
    (Jacobian evaluated at the converged point).

    ``state_trial["stress"]`` passed to ``update_state`` and ``state_residual``
    is always the **current σ NR iterate**.  For models with
    ``stress = Implicit``, the fixed elastic predictor σ_trial is provided
    separately via the ``stress_trial`` keyword argument to ``state_residual``.

    Returns
    -------
    residual_fn : callable(x) -> array of size layout.n_unknown
        Autograd-differentiable residual function.
    layout : ResidualLayout
        Layout descriptor (pack/unpack/slice helpers).
    """
    from manforge.core.material import MaterialModel as _MaterialModel
    layout = ResidualLayout.from_model(model)
    model_name = type(model).__name__
    stress_trial_arr = anp.array(stress_trial)
    user_has_state_residual = (type(model).state_residual is not _MaterialModel.state_residual)

    if layout.is_stress_implicit and not user_has_state_residual:
        raise ValueError(
            f"{model_name}: stress is Implicit but state_residual() is not implemented "
            "— this should not happen."
        )

    def residual_fn(x):
        sigma, dlambda, q_imp = layout.unpack(x)

        # state_trial["stress"] = current σ iterate for update_state
        state_trial_for_update = dict(state_n)
        state_trial_for_update["stress"] = sigma

        # Explicit non-stress states
        q_exp: dict = {}
        if layout.explicit_keys:
            expected_explicit = set(layout.explicit_keys)
            state_n_wrapped = _wrap_state(state_n, model)
            trial_wrapped = _wrap_state(state_trial_for_update, model)
            returned = model.update_state(dlambda, state_n_wrapped, trial_wrapped)
            q_exp = _validate_state_items(
                returned, expected_explicit, StateUpdate,
                "update_state", model_name,
            )

        # Assemble full state
        q_full = {"stress": sigma, **q_imp, **q_exp}
        q_full_state = _wrap_state(q_full, model)

        # Collect R_state and optional R_Δλ from state_residual
        R_state_dict: dict = {}
        r_dl_override = None
        if user_has_state_residual:
            expected_implicit = set(layout.implicit_keys)
            if layout.is_stress_implicit:
                expected_implicit = expected_implicit | {"stress"}
            state_new_wrapped = _wrap_state(q_full, model)
            state_n_wrapped = _wrap_state(state_n, model)
            # state_trial["stress"] = current σ iterate (for flow direction / backstress)
            state_trial_for_residual = dict(state_n)
            state_trial_for_residual["stress"] = sigma
            trial_wrapped = _wrap_state(state_trial_for_residual, model)
            returned = model.state_residual(
                state_new_wrapped, dlambda, state_n_wrapped, trial_wrapped,
                stress_trial=stress_trial_arr,
            )
            hint = (
                "add self.stress(self.default_stress_residual(state_new, dlambda, stress_trial))"
                " for associative flow, or self.stress(R_custom) for non-associative"
                if layout.is_stress_implicit else ""
            )
            R_state_dict, r_dl_override = _validate_state_items(
                returned, expected_implicit, StateResidual,
                "state_residual", model_name,
                extract_dlambda=True,
                hint=hint,
            )

        # Δλ row
        R_dl = r_dl_override if r_dl_override is not None else model.yield_function(q_full_state)

        # σ row
        if layout.is_stress_implicit:
            R_stress = R_state_dict["stress"]
        else:
            R_stress = model.default_stress_residual(
                q_full_state, dlambda, stress_trial_arr
            )

        # Non-stress implicit state rows
        R_q = {k: R_state_dict[k] for k in layout.implicit_keys}

        return layout.pack_residual(R_stress, R_dl, R_q)

    return residual_fn, layout


# ---------------------------------------------------------------------------
# Post-convergence state reconstruction
# ---------------------------------------------------------------------------

def build_state_from_x(model, x_conv, state_n, layout: ResidualLayout) -> dict:
    """Reconstruct the full state dict from a converged NR solution vector.

    Calls ``update_state`` once to obtain explicit non-stress states, then
    assembles ``{"stress": σ, **q_imp, **q_exp}``.

    Parameters
    ----------
    model : MaterialModel
    x_conv : array-like
        Converged NR unknown vector.
    state_n : dict
        State at the beginning of the increment.
    layout : ResidualLayout

    Returns
    -------
    dict
        Full state dict including ``"stress"``.
    """
    sigma, dlambda, q_imp = layout.unpack(np.asarray(x_conv))
    q_exp: dict = {}
    if layout.explicit_keys:
        state_trial_dict = dict(state_n)
        state_trial_dict["stress"] = sigma
        state_n_wrapped = _wrap_state(state_n, model)
        trial_wrapped = _wrap_state(state_trial_dict, model)
        returned = model.update_state(dlambda, state_n_wrapped, trial_wrapped)
        q_exp = _validate_state_items(
            returned, set(layout.explicit_keys), StateUpdate,
            "update_state", type(model).__name__,
        )
    return {"stress": sigma, **q_imp, **q_exp}
