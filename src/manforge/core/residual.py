"""Residual builders for return-mapping systems."""

import autograd
import autograd.numpy as anp
import numpy as np


# ---------------------------------------------------------------------------
# State flatten / unflatten helpers (replaces jax.flatten_util.ravel_pytree)
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
    # Use np.asarray for shape inspection only (no gradient needed for shape).
    shapes = [(k, np.asarray(state[k]).shape) for k in keys]
    parts = []
    for k, shp in shapes:
        v = state[k]
        # Wrap scalar box values in a 1-element array so concatenate works.
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
# Residual builders
# ---------------------------------------------------------------------------

def make_reduced_residual(model, stress_trial, C, state_n):
    """Build the residual function for reduced-system models."""
    ntens = model.ntens

    def residual_fn(x):
        sig = x[:ntens]
        dl = x[ntens]
        st = model.update_state(dl, sig, state_n)
        n = autograd.grad(lambda s: model.yield_function(s, st))(sig)
        R_stress = sig - stress_trial + dl * (C @ n)
        R_yield = model.yield_function(sig, st)
        return anp.concatenate([R_stress, anp.atleast_1d(R_yield)])

    return residual_fn


def make_augmented_residual(model, stress_trial, C, state_n):
    """Build the augmented residual function for implicit-state models."""
    ntens = model.ntens

    _, shapes = _flatten_state(state_n)
    n_state = sum(int(np.prod(shp)) if shp else 1 for _, shp in shapes)

    def unflatten_fn(q_flat):
        return _unflatten_state(q_flat, shapes)

    def residual_fn(x):
        sig = x[:ntens]
        dlambda = x[ntens]
        q_flat = x[ntens + 1:]
        state_new = unflatten_fn(q_flat)

        n = autograd.grad(lambda s: model.yield_function(s, state_new))(sig)

        R_stress = sig - stress_trial + dlambda * (C @ n)
        R_yield = model.yield_function(sig, state_new)

        R_state_dict = model.state_residual(state_new, dlambda, sig, state_n)
        R_state_flat, _ = _flatten_state(R_state_dict)

        return anp.concatenate([R_stress, anp.atleast_1d(R_yield), R_state_flat])

    return residual_fn, n_state, unflatten_fn


def select_residual_builder(model):
    """Return the appropriate residual builder for *model*."""
    if model.hardening_type == "augmented":
        return make_augmented_residual
    return make_reduced_residual
