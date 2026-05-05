"""Residual builders for return-mapping systems."""

import autograd
import autograd.numpy as anp
import numpy as np

from manforge.core.state import StateResidual, StateUpdate, _validate_state_items


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
# Return-value normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_update(returned, explicit_keys: set, model_name: str) -> dict:
    """Accept list[StateUpdate] or legacy dict from update_state; return a plain dict."""
    if isinstance(returned, list):
        return _validate_state_items(returned, explicit_keys, StateUpdate, "update_state", model_name)
    # Legacy / default: plain dict (e.g. default state_residual impl, or old-style models)
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
    # Legacy / default: plain dict
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
# Residual builders
# ---------------------------------------------------------------------------

def make_nr_residual(model, stress_trial, state_n):
    """Build the NR residual function.

    Unknown vector layout::

        x_NR = [σ (ntens)]?          if model.implicit_stress
             + [Δλ (1)]
             + [q_implicit (n_imp)]?  if model.implicit_state_names

    Returns
    -------
    residual_fn : callable(x) -> R
    unknowns_meta : dict
        Keys: ``implicit_stress`` (bool), ``ntens`` (int),
        ``n_implicit`` (int), ``implicit_shapes`` (list).
    unflatten_implicit : callable(flat) -> dict
    """
    ntens = model.ntens
    implicit_keys = sorted(model.implicit_state_names)
    explicit_keys = set(model.state_names) - set(implicit_keys)
    do_implicit_stress = model.implicit_stress
    model_name = type(model).__name__

    implicit_state_n = {k: state_n[k] for k in implicit_keys}
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

    # Index positions in x.
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

        if n_implicit > 0:
            q_imp = unflatten_implicit(x[_q_sl])
        else:
            q_imp = {}

        if do_implicit_stress:
            sig = x[_sig_sl]
            # Compute explicit states from update_state using current sig.
            if explicit_keys:
                q_exp = _normalise_update(
                    model.update_state(dlambda, sig, state_n), explicit_keys, model_name
                )
                q_full = {**q_imp, **q_exp}
            else:
                q_full = q_imp
        else:
            # σ is derived via one fixed-point step from σ_trial using current q.
            if explicit_keys:
                # Approximate explicit states using state_n (they are re-evaluated
                # after σ is determined inside the full solver iteration).
                q_exp_approx = {k: state_n[k] for k in explicit_keys}
                q_full_approx = {**q_imp, **q_exp_approx}
            else:
                q_full_approx = q_imp if q_imp else dict(state_n)
            C_approx = model.elastic_stiffness(q_full_approx)
            n_approx = autograd.grad(
                lambda s: model.yield_function(s, q_full_approx)
            )(anp.array(stress_trial))
            sig = anp.array(stress_trial) - dlambda * (C_approx @ n_approx)
            if explicit_keys:
                q_exp = _normalise_update(
                    model.update_state(dlambda, sig, state_n), explicit_keys, model_name
                )
                q_full = {**q_imp, **q_exp}
            else:
                q_full = q_imp if q_imp else {}

        R_yield = model.yield_function(sig, q_full)

        parts = []
        if do_implicit_stress:
            R_stress = model.stress_residual(sig, dlambda, q_full, anp.array(stress_trial), state_n)
            parts.append(R_stress)
        parts.append(anp.atleast_1d(R_yield))
        if n_implicit > 0:
            R_state_dict = _normalise_residual(
                model.state_residual(q_imp, dlambda, sig, state_n), set(implicit_keys), model_name
            )
            R_state_flat, _ = _flatten_state(R_state_dict)
            parts.append(R_state_flat)

        return anp.concatenate(parts)

    return residual_fn, unknowns_meta, unflatten_implicit


def make_tangent_residual(model, stress_trial, state_n):
    """Build the tangent/Jacobian residual function.

    σ is always an independent variable.  Unknown vector layout::

        x_tan = [σ (ntens)] + [Δλ (1)] + [q_implicit (n_imp)]?

    Returns
    -------
    residual_fn : callable(x) -> R  of size ntens + 1 + n_implicit
    n_implicit : int
    unflatten_implicit : callable(flat) -> dict
    """
    ntens = model.ntens
    implicit_keys = sorted(model.implicit_state_names)
    explicit_keys = set(model.state_names) - set(implicit_keys)
    model_name = type(model).__name__

    implicit_state_n = {k: state_n[k] for k in implicit_keys}
    _, implicit_shapes = _flatten_state(implicit_state_n)
    n_implicit = sum(int(np.prod(shp)) if shp else 1 for _, shp in implicit_shapes)

    def unflatten_implicit(flat):
        return _unflatten_state(flat, implicit_shapes)

    def residual_fn(x):
        sig = x[:ntens]
        dlambda = x[ntens]

        if n_implicit > 0:
            q_imp = unflatten_implicit(x[ntens + 1:])
        else:
            q_imp = {}

        if explicit_keys:
            q_exp = _normalise_update(
                model.update_state(dlambda, sig, state_n), explicit_keys, model_name
            )
            q_full = {**q_imp, **q_exp}
        else:
            q_full = q_imp

        R_stress = model.stress_residual(sig, dlambda, q_full, anp.array(stress_trial), state_n)
        R_yield = model.yield_function(sig, q_full)

        parts = [R_stress, anp.atleast_1d(R_yield)]
        if n_implicit > 0:
            R_state_dict = _normalise_residual(
                model.state_residual(q_imp, dlambda, sig, state_n), set(implicit_keys), model_name
            )
            R_state_flat, _ = _flatten_state(R_state_dict)
            parts.append(R_state_flat)

        return anp.concatenate(parts)

    return residual_fn, n_implicit, unflatten_implicit
