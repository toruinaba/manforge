"""Residual builder for return-mapping systems."""

import autograd.numpy as anp
import numpy as np

from manforge.core.state import (
    StateResidual, StateUpdate, DlambdaResidual, State,
)


# ---------------------------------------------------------------------------
# State flatten / unflatten helpers
# ---------------------------------------------------------------------------

def _flatten_state(state: dict):
    """Flatten a state dict to a 1-D array (alphabetical key order).

    Returns
    -------
    vec : array, shape (n_state,)
    shapes : list of (key, shape) pairs — shapes are plain Python tuples,
             safe to inspect outside of autograd's tracing context.
    """
    keys = sorted(state.keys())
    shapes = [(k, np.asarray(state[k]).shape) for k in keys]
    parts = []
    for k, shp in shapes:
        v = state[k]
        parts.append(anp.reshape(v, (-1,)) if shp else anp.atleast_1d(v))
    vec = anp.concatenate(parts) if parts else anp.zeros(0)
    return vec, shapes


def _unflatten_state(vec, shapes: list) -> dict:
    """Reconstruct a state dict from a flat array and shape metadata."""
    out = {}
    idx = 0
    for k, shp in shapes:
        size = int(np.prod(shp)) if shp else 1
        chunk = vec[idx: idx + size]
        out[k] = chunk.reshape(shp) if shp else chunk[0]
        idx += size
    return out


# ---------------------------------------------------------------------------
# State wrapping helper
# ---------------------------------------------------------------------------

def _wrap_state(data: dict, model) -> State:
    """Wrap a plain dict in a State with the model's field ordering."""
    return State(data, tuple(model.state_names))


# ---------------------------------------------------------------------------
# User method call helpers
# ---------------------------------------------------------------------------

def _call_update_state(model, dlambda, state_n_dict, state_trial_dict,
                       expected_explicit_keys, model_name, require_stress=True):
    """Call model.update_state(dlambda, state_n, state_trial) → dict.

    *expected_explicit_keys* must NOT include "stress".  When *require_stress*
    is True (the default), the returned list MUST include "stress".  Pass
    ``require_stress=False`` when stress is Implicit and update_state is only
    called for non-stress explicit states.
    """
    state_n = _wrap_state(state_n_dict, model)
    state_trial = _wrap_state(state_trial_dict, model)
    returned = model.update_state(dlambda, state_n, state_trial)
    if not isinstance(returned, list):
        raise TypeError(
            f"{model_name}.update_state must return a list of StateUpdate "
            f"(use `self.<field>(value)`), got {type(returned).__name__}"
        )
    for item in returned:
        if isinstance(item, DlambdaResidual):
            raise TypeError(
                f"{model_name}.update_state: self.dlambda(...) is not allowed in "
                "update_state (Δλ has no explicit update rule). "
                "Use state_residual to override the Δλ residual instead."
            )
        if not isinstance(item, StateUpdate):
            raise TypeError(
                f"{model_name}.update_state: every item must be StateUpdate, "
                f"got {type(item).__name__}"
            )
    expected_all = (expected_explicit_keys | {"stress"}) if require_stress else expected_explicit_keys
    actual_keys = {item.name for item in returned}
    if actual_keys != expected_all:
        extra = actual_keys - expected_all
        missing = expected_all - actual_keys
        parts = []
        if extra:
            parts.append(f"unexpected keys: {sorted(extra)}")
        if missing:
            parts.append(f"missing keys: {sorted(missing)}")
            if "stress" in missing:
                parts.append(
                    "Hint: add self.stress(self.default_stress_update(dlambda, state_n, state_trial))"
                    " for associative flow, or self.stress(sig_new) for custom update"
                )
        raise ValueError(
            f"{model_name}.update_state() returned wrong keys "
            f"({'; '.join(parts)}). Expected: {sorted(expected_all)}"
        )
    return {item.name: item.value for item in returned}


def _call_state_residual(model, state_new_dict, dlambda, state_n_dict, state_trial_dict,
                         expected_implicit_keys, model_name):
    """Call model.state_residual(state_new, dlambda, state_n, state_trial).

    *expected_implicit_keys* must NOT include "stress".  The returned list
    MUST include "stress" when stress is Implicit.

    Returns
    -------
    tuple
        ``(state_dict, r_dlambda_or_None)``
    """
    stress_field = model.state_fields["stress"]
    do_implicit_stress = stress_field.kind == "implicit"
    expected_all = (expected_implicit_keys | {"stress"}) if do_implicit_stress else expected_implicit_keys

    state_new = _wrap_state(state_new_dict, model)
    state_n = _wrap_state(state_n_dict, model)
    state_trial = _wrap_state(state_trial_dict, model)
    returned = model.state_residual(state_new, dlambda, state_n, state_trial)

    if not isinstance(returned, list):
        raise TypeError(
            f"{model_name}.state_residual must return a list of StateResidual "
            f"(use `self.<field>(value)`, optionally `self.dlambda(R)`), "
            f"got {type(returned).__name__}"
        )
    state_items = []
    dl_items = []
    for item in returned:
        if isinstance(item, DlambdaResidual):
            dl_items.append(item)
        elif isinstance(item, StateResidual):
            state_items.append(item)
        else:
            raise TypeError(
                f"{model_name}.state_residual: every item must be StateResidual "
                f"or DlambdaResidual (self.dlambda(R)), "
                f"got {type(item).__name__}"
            )
    if len(dl_items) > 1:
        raise ValueError(
            f"{model_name}.state_residual: duplicate self.dlambda(...) entries"
        )
    r_dl = dl_items[0].value if dl_items else None

    actual_keys = {item.name for item in state_items}
    if actual_keys != expected_all:
        extra = actual_keys - expected_all
        missing = expected_all - actual_keys
        parts = []
        if extra:
            parts.append(f"unexpected keys: {sorted(extra)}")
        if missing:
            parts.append(f"missing keys: {sorted(missing)}")
            if "stress" in missing:
                parts.append(
                    "Hint: add self.stress(self.default_stress_residual(state_new, dlambda, state_trial))"
                    " for associative flow, or self.stress(R_custom) for non-associative"
                )
        raise ValueError(
            f"{model_name}.state_residual() returned wrong keys "
            f"({'; '.join(parts)}). Expected: {sorted(expected_all)}"
        )
    return {item.name: item.value for item in state_items}, r_dl


# ---------------------------------------------------------------------------
# Unified residual builder
# ---------------------------------------------------------------------------

def build_residual(model, stress_trial, state_n):
    """Build the unified NR/tangent residual function.

    σ is always an independent variable.  Unknown vector layout::

        x = [σ (ntens)] + [Δλ (1)] + [q_implicit_non_stress (n_imp)]?

    This function serves both the NR phase (iterated until convergence) and
    the tangent/Jacobian phase (Jacobian evaluated at the converged point).

    For ``stress = Explicit`` models, R_stress is the default associative
    residual (σ − σ_trial + Δλ·C·∂f/∂σ = 0).  The user's ``update_state``
    result for stress is accepted (for validation) but ignored; σ_NR drives
    the iteration.

    Returns
    -------
    residual_fn : callable(x) -> R  of size n_unknown
    n_unknown : int  (= ntens + 1 + n_implicit)
    unflatten_implicit : callable(flat) -> dict
    """
    from manforge.core.material import MaterialModel as _MaterialModel
    ntens = model.ntens
    stress_field = model.state_fields["stress"]
    do_implicit_stress = (stress_field.kind == "implicit")
    implicit_keys_non_stress = sorted(k for k in model.implicit_state_names if k != "stress")
    explicit_keys_non_stress = set(
        k for k in model.state_names
        if k != "stress" and k not in model.implicit_state_names
    )
    model_name = type(model).__name__
    user_has_state_residual = (type(model).state_residual is not _MaterialModel.state_residual)

    implicit_state_n = {k: state_n[k] for k in implicit_keys_non_stress}
    flat_impl_n, implicit_shapes = _flatten_state(implicit_state_n)
    n_implicit = len(flat_impl_n)

    stress_trial_arr = anp.array(stress_trial)

    def unflatten_implicit(flat):
        return _unflatten_state(flat, implicit_shapes)

    n_unknown = ntens + 1 + n_implicit

    def residual_fn(x):
        sig = x[:ntens]
        dlambda = x[ntens]
        q_imp = unflatten_implicit(x[ntens + 1:]) if n_implicit > 0 else {}

        # state_trial for update_state: stress = current σ iterate
        state_trial_for_update = dict(state_n)
        state_trial_for_update["stress"] = sig

        # Compute non-stress explicit states
        if explicit_keys_non_stress:
            q_exp = _call_update_state(
                model, dlambda, state_n, state_trial_for_update,
                explicit_keys_non_stress, model_name,
                require_stress=(not do_implicit_stress),
            )
            q_full = {"stress": sig, **q_imp,
                      **{k: v for k, v in q_exp.items() if k != "stress"}}
        else:
            q_full = {"stress": sig, **q_imp}

        q_full_state = _wrap_state(q_full, model)

        # Collect optional R_dλ override and implicit-state residuals from state_residual
        R_state_dict = {}
        r_dl_override = None
        if user_has_state_residual:
            state_trial_for_residual = dict(state_n)
            if do_implicit_stress:
                # Implicit stress: state_trial["stress"] must be the fixed σ_trial so that
                # R_stress = σ − σ_trial + ... differentiates correctly w.r.t. the iterate σ.
                state_trial_for_residual["stress"] = stress_trial_arr
            else:
                # Explicit stress: pass current σ iterate so the user can evaluate
                # flow direction / backstress direction at the current state.
                state_trial_for_residual["stress"] = sig
            R_state_dict, r_dl_override = _call_state_residual(
                model, q_full, dlambda, state_n, state_trial_for_residual,
                set(implicit_keys_non_stress), model_name,
            )
        elif do_implicit_stress:
            raise ValueError(
                f"{model_name}: stress is Implicit but state_residual() is not implemented "
                "— this should not happen."
            )

        # Δλ row: user override if provided, else yield_function default
        R_dl = r_dl_override if r_dl_override is not None else model.yield_function(q_full_state)

        # Stress residual row
        if do_implicit_stress:
            R_stress = R_state_dict["stress"]
        else:
            R_stress = model.default_stress_residual(
                q_full_state, dlambda, {"stress": stress_trial_arr}
            )

        R_state_non_stress = {k: v for k, v in R_state_dict.items() if k != "stress"}
        parts = [R_stress, anp.atleast_1d(R_dl)]
        if R_state_non_stress:
            R_flat, _ = _flatten_state(R_state_non_stress)
            parts.append(R_flat)

        return anp.concatenate(parts)

    return residual_fn, n_unknown, unflatten_implicit
