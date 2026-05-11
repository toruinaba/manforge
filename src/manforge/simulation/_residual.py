"""Residual builder for return-mapping systems."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import autograd.numpy as anp
import numpy as np

from manforge.core.state import (
    StateResidual, StateUpdate, _validate_state_items, State,
)
from manforge.simulation._layout import ResidualLayout
from manforge._typing import FloatArray, StressVec, StateDict

if TYPE_CHECKING:
    from manforge.core.material import MaterialModel


# ---------------------------------------------------------------------------
# State wrapping helper
# ---------------------------------------------------------------------------

def _wrap_state(data: StateDict, model: "MaterialModel") -> State:
    """Wrap a plain dict in a State with the model's field ordering."""
    return State(data, tuple(model.state_names))


# ---------------------------------------------------------------------------
# Unified residual builder
# ---------------------------------------------------------------------------

def build_residual(
    model: "MaterialModel",
    stress_trial: StressVec,
    state_n: StateDict,
    strain_inc: "FloatArray | None" = None,
) -> tuple[Callable[[FloatArray], FloatArray], ResidualLayout]:
    """Build the unified NR/tangent residual function.

    Unknown / residual vector layout (declaration order, stress-first)::

        x = [σ (ntens)] + [Δλ (1)] + [q_implicit_non_stress (declaration order)]
        R = [R_σ (ntens)] + [R_Δλ (1)] + [R_q_non_stress (declaration order)]

    σ is always an independent variable.  This function serves both the NR
    phase (iterated until convergence) and the tangent/Jacobian phase
    (Jacobian evaluated at the converged point).

    ``state_new`` passed to ``update_state`` and ``state_residual`` holds the
    **current NR iterate**: ``state_new["stress"] = σ_k``,
    ``state_new[implicit_key] = current iterate``.  Explicit non-stress keys
    carry ``state_n`` values when passed to ``update_state``; after
    ``update_state`` returns, its results are merged before calling
    ``state_residual``.  The fixed elastic predictor ``stress_trial`` and the
    strain increment ``strain_inc`` are forwarded as keyword arguments to both
    methods.

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
    strain_inc_arr = anp.array(strain_inc) if strain_inc is not None else None
    user_has_state_residual = (type(model).state_residual is not _MaterialModel.state_residual)

    if layout.is_stress_implicit and not user_has_state_residual:
        raise ValueError(
            f"{model_name}: stress is Implicit but state_residual() is not implemented "
            "— this should not happen."
        )

    def residual_fn(x):
        sigma, dlambda, q_imp = layout.unpack(x)

        # state_new for update_state: σ_k + implicit iterates + state_n for explicit keys
        state_new_for_update: StateDict = {**state_n, "stress": sigma, **q_imp}

        # Explicit non-stress states
        q_exp: StateDict = {}
        if layout.explicit_keys:
            expected_explicit = set(layout.explicit_keys)
            state_n_wrapped = _wrap_state(state_n, model)
            state_new_wrapped = _wrap_state(state_new_for_update, model)
            returned = model.update_state(
                dlambda, state_new_wrapped, state_n_wrapped,
                stress_trial=stress_trial_arr,
                strain_inc=strain_inc_arr,
            )
            raw = _validate_state_items(
                returned, expected_explicit, StateUpdate,
                "update_state", model_name,
            )
            assert isinstance(raw, dict)
            q_exp = raw

        # Assemble full state (explicit keys now hold updated values)
        q_full = {"stress": sigma, **q_imp, **q_exp}
        q_full_state = _wrap_state(q_full, model)

        # Collect R_state and optional R_Δλ from state_residual
        R_state_dict: StateDict = {}
        r_dl_override = None
        if user_has_state_residual:
            expected_implicit = set(layout.implicit_keys)
            if layout.is_stress_implicit:
                expected_implicit = expected_implicit | {"stress"}
            state_new_full_wrapped = _wrap_state(q_full, model)
            state_n_wrapped = _wrap_state(state_n, model)
            returned = model.state_residual(
                state_new_full_wrapped, dlambda, state_n_wrapped,
                stress_trial=stress_trial_arr,
                strain_inc=strain_inc_arr,
            )
            hint = (
                "add self.stress(self.default_stress_residual(state_new, dlambda, stress_trial))"
                " for associative flow, or self.stress(R_custom) for non-associative"
                if layout.is_stress_implicit else ""
            )
            raw2 = _validate_state_items(
                returned, expected_implicit, StateResidual,
                "state_residual", model_name,
                extract_dlambda=True,
                hint=hint,
            )
            assert isinstance(raw2, tuple)
            R_state_dict, r_dl_override = raw2

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

def build_state_from_x(
    model: "MaterialModel",
    x_conv: FloatArray,
    state_n: StateDict,
    layout: ResidualLayout,
    stress_trial: "FloatArray | None" = None,
    strain_inc: "FloatArray | None" = None,
) -> StateDict:
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
    stress_trial : array-like, optional
        Fixed elastic predictor forwarded to ``update_state``.
    strain_inc : array-like, optional
        Strain increment forwarded to ``update_state``.

    Returns
    -------
    dict
        Full state dict including ``"stress"``.
    """
    sigma, dlambda, q_imp = layout.unpack(np.asarray(x_conv))
    q_exp: StateDict = {}
    if layout.explicit_keys:
        state_new_dict: StateDict = {**state_n, "stress": sigma, **q_imp}
        state_n_wrapped = _wrap_state(state_n, model)
        state_new_wrapped = _wrap_state(state_new_dict, model)
        stress_trial_arr = np.asarray(stress_trial) if stress_trial is not None else None
        strain_inc_arr = np.asarray(strain_inc) if strain_inc is not None else None
        returned = model.update_state(
            dlambda, state_new_wrapped, state_n_wrapped,
            stress_trial=stress_trial_arr,
            strain_inc=strain_inc_arr,
        )
        raw = _validate_state_items(
            returned, set(layout.explicit_keys), StateUpdate,
            "update_state", type(model).__name__,
        )
        assert isinstance(raw, dict)
        q_exp = raw
    return {"stress": sigma, **q_imp, **q_exp}
