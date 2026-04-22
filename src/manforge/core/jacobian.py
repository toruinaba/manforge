"""Jacobian block decomposition utilities for return-mapping residual systems."""

from __future__ import annotations

from dataclasses import dataclass

import autograd
import autograd.numpy as anp
import numpy as np

from manforge.core.residual import _flatten_state


@dataclass
class JacobianBlocks:
    """Named blocks of the return-mapping residual Jacobian at the converged point."""

    dstress_dsigma: anp.ndarray
    dstress_ddlambda: anp.ndarray
    dyield_dsigma: anp.ndarray
    dyield_ddlambda: anp.ndarray
    dstress_dstate: dict | None
    dyield_dstate: dict | None
    dstate_dsigma: dict | None
    dstate_ddlambda: dict | None
    dstate_dstate: dict | None
    full: anp.ndarray


def ad_jacobian_blocks(
    model, result, state_n: dict, *, stress_trial=None
) -> JacobianBlocks:
    """Compute the residual Jacobian at the converged point and decompose into blocks."""
    from manforge.core.stress_update import StressUpdateResult

    if isinstance(result, StressUpdateResult):
        if result.return_mapping is None:
            stress = result.stress_trial
            dlambda = anp.array(0.0)
            stress_trial = result.stress_trial
            state = result.state
        else:
            stress = result.return_mapping.stress
            dlambda = result.return_mapping.dlambda
            stress_trial = result.stress_trial
            state = result.return_mapping.state
    else:
        stress = result.stress
        dlambda = result.dlambda
        state = result.state
        if stress_trial is None:
            raise ValueError(
                "stress_trial must be provided when passing a ReturnMappingResult "
                "to ad_jacobian_blocks(). Use stress_update() instead, or pass "
                "stress_trial=... explicitly."
            )
    ntens = model.ntens
    C = model.elastic_stiffness()

    _blocks_fn = (
        _jacobian_blocks_augmented
        if model.hardening_type == "augmented"
        else _jacobian_blocks_reduced
    )
    return _blocks_fn(model, stress, state, dlambda, stress_trial, C, state_n, ntens)


def _jacobian_blocks_reduced(
    model, stress, state, dlambda, stress_trial, C, state_n, ntens
):
    from manforge.core.residual import make_reduced_residual

    residual_fn = make_reduced_residual(model, stress_trial, C, state_n)
    x_conv = anp.concatenate([anp.array(stress), anp.array([float(dlambda)])])
    J = autograd.jacobian(residual_fn)(x_conv)

    return JacobianBlocks(
        dstress_dsigma=J[:ntens, :ntens],
        dstress_ddlambda=J[:ntens, ntens],
        dyield_dsigma=J[ntens, :ntens],
        dyield_ddlambda=J[ntens, ntens],
        dstress_dstate=None,
        dyield_dstate=None,
        dstate_dsigma=None,
        dstate_ddlambda=None,
        dstate_dstate=None,
        full=J,
    )


def _jacobian_blocks_augmented(
    model, stress, state, dlambda, stress_trial, C, state_n, ntens
):
    from manforge.core.residual import make_augmented_residual

    residual_fn, n_state, unflatten_fn = make_augmented_residual(
        model, stress_trial, C, state_n
    )

    flat_state, _ = _flatten_state(state)
    x_conv = anp.concatenate([
        anp.array(stress),
        anp.array([float(dlambda)]),
        flat_state,
    ])
    J = autograd.jacobian(residual_fn)(x_conv)

    state_keys = sorted(state.keys())
    slices = _build_state_slices(state, state_keys)

    dstress_dsigma   = J[:ntens, :ntens]
    dstress_ddlambda = J[:ntens, ntens]
    dyield_dsigma    = J[ntens, :ntens]
    dyield_ddlambda  = J[ntens, ntens]

    dstress_dstate: dict = {}
    dyield_dstate:  dict = {}
    dstate_dsigma:  dict = {}
    dstate_ddlambda: dict = {}
    dstate_dstate:  dict = {}

    q0 = ntens + 1
    for key, sl in slices.items():
        q_sl = slice(q0 + sl.start, q0 + sl.stop)
        dstress_dstate[key]  = J[:ntens, q_sl]
        dyield_dstate[key]   = J[ntens, q_sl]
        dstate_dsigma[key]   = J[q_sl, :ntens]
        dstate_ddlambda[key] = J[q_sl, ntens]
        dstate_dstate[key]   = {}
        for key2, sl2 in slices.items():
            q_sl2 = slice(q0 + sl2.start, q0 + sl2.stop)
            dstate_dstate[key][key2] = J[q_sl, q_sl2]

    return JacobianBlocks(
        dstress_dsigma=dstress_dsigma,
        dstress_ddlambda=dstress_ddlambda,
        dyield_dsigma=dyield_dsigma,
        dyield_ddlambda=dyield_ddlambda,
        dstress_dstate=dstress_dstate,
        dyield_dstate=dyield_dstate,
        dstate_dsigma=dstate_dsigma,
        dstate_ddlambda=dstate_ddlambda,
        dstate_dstate=dstate_dstate,
        full=J,
    )


def _build_state_slices(state: dict, keys: list) -> dict:
    slices = {}
    offset = 0
    for key in keys:
        val = np.asarray(state[key])
        size = val.size
        slices[key] = slice(offset, offset + size)
        offset += size
    return slices
