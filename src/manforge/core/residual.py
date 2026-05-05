"""Residual builders for return-mapping systems."""

import autograd
import autograd.numpy as anp
import numpy as np

from manforge.core.state import (
    StateResidual, StateUpdate, State, _validate_state_items, _state_with_stress,
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
# Return-value normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_update(returned, explicit_keys: set, model_name: str) -> dict:
    """Accept list[StateUpdate] or legacy dict from update_state; return a plain dict."""
    if isinstance(returned, list):
        return _validate_state_items(returned, explicit_keys, StateUpdate, "update_state", model_name)
    if isinstance(returned, dict):
        actual = set(returned.keys())
        if actual != explicit_keys:
            extra = actual - explicit_keys
            missing = explicit_keys - actual
            parts = []
            if extra:
                parts.append(f"unexpected keys: {sorted(extra)}")
            if missing:
                parts.append(f"missing keys: {sorted(missing)}")
            raise ValueError(
                f"{model_name}.update_state() returned wrong keys "
                f"({'; '.join(parts)}). Expected: {sorted(explicit_keys)}"
            )
        return returned
    raise TypeError(
        f"{model_name}.update_state must return a list of StateUpdate "
        f"(use `self.<field>(value)`) or a dict, got {type(returned).__name__}"
    )


def _normalise_residual(returned, implicit_keys: set, model_name: str) -> dict:
    """Accept list[StateResidual] or legacy dict from state_residual; return a plain dict."""
    if isinstance(returned, list):
        return _validate_state_items(returned, implicit_keys, StateResidual, "state_residual", model_name)
    if isinstance(returned, dict):
        actual = set(returned.keys())
        if actual != implicit_keys:
            extra = actual - implicit_keys
            missing = implicit_keys - actual
            parts = []
            if extra:
                parts.append(f"unexpected keys: {sorted(extra)}")
            if missing:
                parts.append(f"missing keys: {sorted(missing)}")
            raise ValueError(
                f"{model_name}.state_residual() returned wrong keys "
                f"({'; '.join(parts)}). Expected: {sorted(implicit_keys)}"
            )
        return returned
    raise TypeError(
        f"{model_name}.state_residual must return a list of StateResidual "
        f"(use `self.<field>(value)`) or a dict, got {type(returned).__name__}"
    )


# ---------------------------------------------------------------------------
# User method call helpers (new API: state_n/state_trial as State)
# ---------------------------------------------------------------------------

def _call_update_state(model, dlambda, state_n_dict, state_trial_dict,
                        expected_explicit_keys, model_name):
    """Call model.update_state(dlambda, state_n, state_trial) → dict.

    *expected_explicit_keys* must NOT include "stress" — stress is handled
    separately by the framework.
    """
    state_n = _wrap_state(state_n_dict, model)
    state_trial = _wrap_state(state_trial_dict, model)
    returned = model.update_state(dlambda, state_n, state_trial)
    # Accept list[StateUpdate] with only the non-stress explicit keys
    if isinstance(returned, list):
        # User may optionally include "stress" in the list — accept it
        actual_keys = {item.name for item in returned if isinstance(item, StateUpdate)}
        stress_provided = "stress" in actual_keys
        non_stress_keys = actual_keys - {"stress"}
        if non_stress_keys != expected_explicit_keys:
            extra = non_stress_keys - expected_explicit_keys
            missing = expected_explicit_keys - non_stress_keys
            parts = []
            if extra:
                parts.append(f"unexpected keys: {sorted(extra)}")
            if missing:
                parts.append(f"missing keys: {sorted(missing)}")
            raise ValueError(
                f"{model_name}.update_state() returned wrong non-stress keys "
                f"({'; '.join(parts)}). Expected: {sorted(expected_explicit_keys)}"
            )
        # Validate types
        for item in returned:
            if not isinstance(item, StateUpdate):
                raise TypeError(
                    f"{model_name}.update_state: every item must be StateUpdate, "
                    f"got {type(item).__name__}"
                )
        return {item.name: item.value for item in returned}
    if isinstance(returned, dict):
        stress_provided = "stress" in returned
        actual_non_stress = set(returned.keys()) - {"stress"}
        if actual_non_stress != expected_explicit_keys:
            extra = actual_non_stress - expected_explicit_keys
            missing = expected_explicit_keys - actual_non_stress
            parts = []
            if extra:
                parts.append(f"unexpected keys: {sorted(extra)}")
            if missing:
                parts.append(f"missing keys: {sorted(missing)}")
            raise ValueError(
                f"{model_name}.update_state() returned wrong keys "
                f"({'; '.join(parts)}). Expected: {sorted(expected_explicit_keys)}"
            )
        return returned
    raise TypeError(
        f"{model_name}.update_state must return a list of StateUpdate "
        f"(use `self.<field>(value)`) or a dict, got {type(returned).__name__}"
    )


def _call_state_residual(model, state_new_dict, dlambda, state_n_dict, state_trial_dict,
                          expected_implicit_keys, model_name):
    """Call model.state_residual(state_new, dlambda, state_n, state_trial) → dict.

    *expected_implicit_keys* must NOT include "stress".  The user may optionally
    include "stress" in the returned list to provide a custom R_stress.
    """
    state_new = _wrap_state(state_new_dict, model)
    state_n = _wrap_state(state_n_dict, model)
    state_trial = _wrap_state(state_trial_dict, model)
    returned = model.state_residual(state_new, dlambda, state_n, state_trial)
    if isinstance(returned, list):
        actual_keys = {item.name for item in returned if isinstance(item, StateResidual)}
        non_stress_keys = actual_keys - {"stress"}
        if non_stress_keys != expected_implicit_keys:
            extra = non_stress_keys - expected_implicit_keys
            missing = expected_implicit_keys - non_stress_keys
            parts = []
            if extra:
                parts.append(f"unexpected keys: {sorted(extra)}")
            if missing:
                parts.append(f"missing keys: {sorted(missing)}")
            raise ValueError(
                f"{model_name}.state_residual() returned wrong non-stress keys "
                f"({'; '.join(parts)}). Expected: {sorted(expected_implicit_keys)}"
            )
        for item in returned:
            if not isinstance(item, StateResidual):
                raise TypeError(
                    f"{model_name}.state_residual: every item must be StateResidual, "
                    f"got {type(item).__name__}"
                )
        return {item.name: item.value for item in returned}
    if isinstance(returned, dict):
        actual_non_stress = set(returned.keys()) - {"stress"}
        if actual_non_stress != expected_implicit_keys:
            extra = actual_non_stress - expected_implicit_keys
            missing = expected_implicit_keys - actual_non_stress
            parts = []
            if extra:
                parts.append(f"unexpected keys: {sorted(extra)}")
            if missing:
                parts.append(f"missing keys: {sorted(missing)}")
            raise ValueError(
                f"{model_name}.state_residual() returned wrong keys "
                f"({'; '.join(parts)}). Expected: {sorted(expected_implicit_keys)}"
            )
        return returned
    raise TypeError(
        f"{model_name}.state_residual must return a list of StateResidual "
        f"(use `self.<field>(value)`) or a dict, got {type(returned).__name__}"
    )


# ---------------------------------------------------------------------------
# Residual builders
# ---------------------------------------------------------------------------

def make_nr_residual(model, stress_trial, state_n):
    """Build the NR residual function.

    stress field kind determines the NR unknown vector layout:

    * stress is Explicit: σ derived via fixed-point each iteration (scalar NR
      when no other implicit states; vector NR when implicit states exist)
    * stress is Implicit: σ is an independent NR unknown (same layout as
      old ``implicit_stress=True``)

    Unknown vector layout::

        x_NR = [σ (ntens)]?                      if stress is Implicit
             + [Δλ (1)]
             + [q_implicit_non_stress (n_imp)]?

    Returns
    -------
    residual_fn : callable(x) -> R
    unknowns_meta : dict
        Keys: ``implicit_stress`` (bool), ``ntens`` (int),
        ``n_implicit`` (int), ``implicit_shapes`` (list).
    unflatten_implicit : callable(flat) -> dict
    """
    ntens = model.ntens
    stress_field = model.state_fields["stress"]
    do_implicit_stress = (stress_field.kind == "implicit")
    model_name = type(model).__name__

    # Keys excluding "stress"
    implicit_keys_non_stress = sorted([k for k in model.implicit_state_names if k != "stress"])
    explicit_keys_non_stress = set(
        k for k in model.state_names
        if k != "stress" and k not in model.implicit_state_names
    )

    implicit_state_n = {k: state_n[k] for k in implicit_keys_non_stress}
    flat_impl_n, implicit_shapes = _flatten_state(implicit_state_n)
    n_implicit = len(flat_impl_n)

    def unflatten_implicit(flat):
        return _unflatten_state(flat, implicit_shapes)

    unknowns_meta = {
        "implicit_stress": do_implicit_stress,
        "ntens": ntens,
        "n_implicit": n_implicit,
        "implicit_shapes": implicit_shapes,
    }

    if do_implicit_stress:
        _sig_sl = slice(0, ntens)
        _dl_idx = ntens
        _q_sl = slice(ntens + 1, ntens + 1 + n_implicit)
    else:
        _sig_sl = None
        _dl_idx = 0
        _q_sl = slice(1, 1 + n_implicit)

    def residual_fn(x):
        dlambda = x[_dl_idx]
        q_imp = unflatten_implicit(x[_q_sl]) if n_implicit > 0 else {}

        if do_implicit_stress:
            sig = x[_sig_sl]
            # state_trial: current sig + state_n for others
            state_trial_dict = dict(state_n)
            state_trial_dict["stress"] = sig
            if explicit_keys_non_stress:
                q_exp = _call_update_state(
                    model, dlambda, state_n, state_trial_dict,
                    explicit_keys_non_stress, model_name
                )
                # q_exp may include "stress" if user provided custom update
                sig = q_exp.get("stress", sig)
                q_full = {"stress": sig, **q_imp,
                          **{k: v for k, v in q_exp.items() if k != "stress"}}
            else:
                q_full = {"stress": sig, **q_imp}
        else:
            # Derive σ via associative formula using current implicit state
            q_approx = {"stress": anp.array(stress_trial), **q_imp}
            if explicit_keys_non_stress:
                q_approx.update({k: state_n[k] for k in explicit_keys_non_stress})
            q_approx_state = _wrap_state(q_approx, model)
            C_approx = model.elastic_stiffness(q_approx_state)
            n_approx = autograd.grad(
                lambda s: model.yield_function(_state_with_stress(q_approx_state, s))
            )(anp.array(stress_trial))
            sig = anp.array(stress_trial) - dlambda * (C_approx @ n_approx)

            state_trial_dict = dict(state_n)
            state_trial_dict["stress"] = sig
            if explicit_keys_non_stress:
                q_exp = _call_update_state(
                    model, dlambda, state_n, state_trial_dict,
                    explicit_keys_non_stress, model_name
                )
                # User may provide custom stress via update_state
                sig = q_exp.get("stress", sig)
                q_full = {"stress": sig, **q_imp,
                          **{k: v for k, v in q_exp.items() if k != "stress"}}
            else:
                q_full = {"stress": sig, **q_imp}

        q_full_state = _wrap_state(q_full, model)
        R_yield = model.yield_function(q_full_state)

        parts = []
        if do_implicit_stress:
            stress_trial_arr = anp.array(stress_trial)
            # Default associative R_stress (overridden below if user provides it)
            R_stress = model._default_stress_residual(
                q_full["stress"], dlambda, q_full_state, stress_trial_arr
            )
            parts.append(R_stress)

        parts.append(anp.atleast_1d(R_yield))

        if n_implicit > 0:
            state_trial_for_residual = dict(state_n)
            if do_implicit_stress:
                # When stress is Implicit, state_trial["stress"] must be the fixed
                # elastic trial stress so that R_stress = σ − σ_trial + ... differentiates
                # correctly w.r.t. the current iterate σ.
                state_trial_for_residual["stress"] = anp.array(stress_trial)
            else:
                # When stress is Explicit, pass the current derived stress as the trial
                # so that user's state_residual can access it (e.g. AF flow direction).
                state_trial_for_residual["stress"] = q_full.get("stress", anp.array(stress_trial))
            R_state_dict = _call_state_residual(
                model, q_full, dlambda, state_n, state_trial_for_residual,
                set(implicit_keys_non_stress), model_name
            )
            # If user provided stress residual, replace default
            if "stress" in R_state_dict and do_implicit_stress:
                parts[0] = R_state_dict["stress"]
            R_state_non_stress = {k: v for k, v in R_state_dict.items() if k != "stress"}
            if R_state_non_stress:
                R_state_flat, _ = _flatten_state(R_state_non_stress)
                parts.append(R_state_flat)

        return anp.concatenate(parts)

    return residual_fn, unknowns_meta, unflatten_implicit


def make_tangent_residual(model, stress_trial, state_n):
    """Build the tangent/Jacobian residual function.

    σ is always an independent variable.  Unknown vector layout::

        x_tan = [σ (ntens)] + [Δλ (1)] + [q_implicit_non_stress (n_imp)]?

    Returns
    -------
    residual_fn : callable(x) -> R  of size ntens + 1 + n_implicit
    n_implicit : int
    unflatten_implicit : callable(flat) -> dict
    """
    ntens = model.ntens
    stress_field = model.state_fields["stress"]
    do_implicit_stress = (stress_field.kind == "implicit")
    implicit_keys_non_stress = sorted([k for k in model.implicit_state_names if k != "stress"])
    explicit_keys_non_stress = set(
        k for k in model.state_names
        if k != "stress" and k not in model.implicit_state_names
    )
    model_name = type(model).__name__

    implicit_state_n = {k: state_n[k] for k in implicit_keys_non_stress}
    _, implicit_shapes = _flatten_state(implicit_state_n)
    n_implicit = sum(int(np.prod(shp)) if shp else 1 for _, shp in implicit_shapes)

    def unflatten_implicit(flat):
        return _unflatten_state(flat, implicit_shapes)

    def residual_fn(x):
        sig = x[:ntens]
        dlambda = x[ntens]
        q_imp = unflatten_implicit(x[ntens + 1:]) if n_implicit > 0 else {}

        state_trial_dict = dict(state_n)
        state_trial_dict["stress"] = sig
        if explicit_keys_non_stress:
            q_exp = _call_update_state(
                model, dlambda, state_n, state_trial_dict,
                explicit_keys_non_stress, model_name
            )
            q_full = {"stress": sig, **q_imp,
                      **{k: v for k, v in q_exp.items() if k != "stress"}}
        else:
            q_full = {"stress": sig, **q_imp}

        q_full_state = _wrap_state(q_full, model)
        stress_trial_arr = anp.array(stress_trial)

        R_stress = model._default_stress_residual(sig, dlambda, q_full_state, stress_trial_arr)
        R_yield = model.yield_function(q_full_state)

        parts = [R_stress, anp.atleast_1d(R_yield)]

        if n_implicit > 0:
            state_trial_for_residual = dict(state_n)
            if do_implicit_stress:
                # When stress is Implicit, state_trial["stress"] must be the fixed elastic
                # trial stress so that R_stress = σ − σ_trial + ... differentiates correctly
                # w.r.t. the tangent variable σ.
                state_trial_for_residual["stress"] = stress_trial_arr
            else:
                # When stress is Explicit, pass the tangent variable sig as the trial
                # so that user's state_residual captures ∂R_state/∂σ correctly.
                state_trial_for_residual["stress"] = sig
            R_state_dict = _call_state_residual(
                model, q_full, dlambda, state_n, state_trial_for_residual,
                set(implicit_keys_non_stress), model_name
            )
            # If user provided stress residual, replace default
            if "stress" in R_state_dict:
                parts[0] = R_state_dict["stress"]
            R_state_non_stress = {k: v for k, v in R_state_dict.items() if k != "stress"}
            if R_state_non_stress:
                R_state_flat, _ = _flatten_state(R_state_non_stress)
                parts.append(R_state_flat)

        return anp.concatenate(parts)

    return residual_fn, n_implicit, unflatten_implicit
