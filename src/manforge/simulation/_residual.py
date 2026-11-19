"""Residual builder for return-mapping systems."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from manforge.core.state import (
    DlambdaResidual, Increment, StateResidual, StateUpdate, _validate_state_items, State,
)
from manforge.simulation._layout import ResidualLayout
from manforge._typing import FloatArray, StateDict

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
    inc: Increment,
) -> tuple[Callable[..., FloatArray], ResidualLayout]:
    """Build the unified NR/tangent residual function for one increment.

    Unknown / residual vector layout (declaration order, stress-first)::

        x = [σ (ntens)] + [Δλ (1)] + [q_implicit_non_stress (declaration order)]
        R = [R_σ (ntens)] + [R_Δλ (1)] + [R_q_non_stress (declaration order)]

    σ is always an independent variable.  This function serves both the NR
    phase (iterated until convergence) and the tangent/Jacobian phase
    (Jacobian evaluated at the converged point).

    The state handed to ``update_state`` and ``state_residual`` holds the
    **current NR iterate**: ``new["stress"] = σ_k``,
    ``new[implicit_key] = current iterate``.  Explicit non-stress keys carry
    their ``inc.state_n`` values when passed to ``update_state``; after
    ``update_state`` returns, its results are merged before calling
    ``state_residual``.  Both receive the increment itself.

    ``residual_fn`` also accepts a keyword override ``inc=`` that replaces the
    increment bound here.  The consistent tangent uses it to differentiate R
    with respect to the load increment (through the whole trial state and Δε),
    so user residuals may depend on either in any row and with any coefficient
    — the tangent does not assume the ``R_σ = σ − σ_trial + …`` form.  The NR
    loop and the Jacobian inspection call ``residual_fn(x)`` and are unaffected.

    Returns
    -------
    residual_fn : callable(x, *, inc=None) -> array of size layout.n_unknown
        Autograd-differentiable residual function.
    layout : ResidualLayout
        Layout descriptor (pack/unpack/slice helpers).
    """
    from manforge.core.material import MaterialModel as _MaterialModel
    layout = ResidualLayout.from_model(model)
    model_name = type(model).__name__
    inc_bound = inc
    user_has_state_residual = (type(model).state_residual is not _MaterialModel.state_residual)

    if layout.is_stress_implicit and not user_has_state_residual:
        raise ValueError(
            f"{model_name}: stress is Implicit but state_residual() is not implemented "
            "— this should not happen."
        )

    # The user methods' return structure (item types, names, duplicates) is a
    # property of the model class, not of the NR iterate, so validating it on
    # every residual evaluation — including inside every autodiff trace —
    # repays nothing after the first pass.  Validate once, then switch to a
    # bare extraction.  Measured at ~40% of residual_fn time on YU 3D.
    validated = {"done": False}

    def _extract_updates(returned):
        return {item.name: item.value for item in returned}

    def _extract_residuals(returned):
        out, r_dl = {}, None
        for item in returned:
            if isinstance(item, DlambdaResidual):
                r_dl = item.value
            else:
                out[item.name] = item.value
        return out, r_dl

    def residual_fn(x, *, inc=None):
        inc_ = inc_bound if inc is None else inc
        state_n = inc_.state_n.as_dict()
        sigma, dlambda, q_imp = layout.unpack(x)
        first = not validated["done"]

        # iterate handed to update_state: σ_k + implicit iterates + state_n for explicit keys
        state_new_for_update: StateDict = {**state_n, "stress": sigma, **q_imp}

        # Explicit non-stress states
        q_exp: StateDict = {}
        if layout.explicit_keys:
            returned = model.update_state(
                _wrap_state(state_new_for_update, model), dlambda, inc_,
            )
            if first:
                raw = _validate_state_items(
                    returned, set(layout.explicit_keys), StateUpdate,
                    "update_state", model_name,
                )
                assert isinstance(raw, dict)
                q_exp = raw
            else:
                q_exp = _extract_updates(returned)

        # Assemble full state (explicit keys now hold updated values)
        q_full = {"stress": sigma, **q_imp, **q_exp}
        q_full_state = _wrap_state(q_full, model)

        # Collect R_state and optional R_Δλ from state_residual
        R_state_dict: StateDict = {}
        r_dl_override = None
        if user_has_state_residual:
            returned = model.state_residual(_wrap_state(q_full, model), dlambda, inc_)
            if first:
                expected_implicit = set(layout.implicit_keys)
                if layout.is_stress_implicit:
                    expected_implicit = expected_implicit | {"stress"}
                hint = (
                    "add self.stress(self.default_stress_residual(new, dlambda, "
                    "inc.trial[\"stress\"]))"
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
            else:
                R_state_dict, r_dl_override = _extract_residuals(returned)

        validated["done"] = True

        # Δλ row
        R_dl = r_dl_override if r_dl_override is not None else model.yield_function(q_full_state)

        # σ row
        if layout.is_stress_implicit:
            R_stress = R_state_dict["stress"]
        else:
            R_stress = model.default_stress_residual(
                q_full_state, dlambda, inc_.trial["stress"]
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
    inc: Increment,
    layout: ResidualLayout,
) -> StateDict:
    """Reconstruct the full state dict from a converged NR solution vector.

    Calls ``update_state`` once to obtain explicit non-stress states, then
    assembles ``{"stress": σ, **q_imp, **q_exp}``.

    Parameters
    ----------
    model : MaterialModel
    x_conv : array-like
        Converged NR unknown vector.
    inc : Increment
        The increment the vector was solved for.
    layout : ResidualLayout

    Returns
    -------
    dict
        Full state dict including ``"stress"``.
    """
    sigma, dlambda, q_imp = layout.unpack(np.asarray(x_conv))
    q_exp: StateDict = {}
    if layout.explicit_keys:
        state_new_dict: StateDict = {**inc.state_n.as_dict(), "stress": sigma, **q_imp}
        returned = model.update_state(_wrap_state(state_new_dict, model), dlambda, inc)
        raw = _validate_state_items(
            returned, set(layout.explicit_keys), StateUpdate,
            "update_state", type(model).__name__,
        )
        assert isinstance(raw, dict)
        q_exp = raw
    return {"stress": sigma, **q_imp, **q_exp}
