"""Jacobian block decomposition utilities for return-mapping residual systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import autograd
import autograd.numpy as anp
import numpy as np

from manforge.simulation._residual import build_residual
from manforge.simulation._layout import ResidualLayout


@dataclass
class JacobianBlocks:
    """Named blocks of the return-mapping residual Jacobian at the converged point.

    The residual / unknown vector layout follows :class:`~manforge.simulation._layout.ResidualLayout`:
    ``[σ (ntens) | Δλ (1) | q_implicit_non_stress (declaration order)]``.

    Block naming convention: ``dR<row>_d<col>`` where the row / column labels are:

    - ``sigma``   — the σ block (rows/cols 0 … ntens-1)
    - ``dlambda`` — the Δλ row/col (index ntens)
    - ``state``   — the implicit non-stress state block (rows/cols ntens+1 … )

    State blocks are always ``dict[str, ndarray]`` (empty dict when there are no
    implicit non-stress states), never ``None``.

    Attributes
    ----------
    layout : ResidualLayout
        Layout descriptor used to compute this Jacobian.
    dRsigma_dsigma : ndarray, shape (ntens, ntens)
        ∂R_σ / ∂σ
    dRsigma_ddlambda : ndarray, shape (ntens,)
        ∂R_σ / ∂Δλ
    dRsigma_dstate : dict[str, ndarray]
        ∂R_σ / ∂q_k for each implicit non-stress state key.
    dRdlambda_dsigma : ndarray, shape (ntens,)
        ∂R_Δλ / ∂σ  (gradient of the Δλ-row w.r.t. σ)
    dRdlambda_ddlambda : float
        ∂R_Δλ / ∂Δλ
    dRdlambda_dstate : dict[str, ndarray]
        ∂R_Δλ / ∂q_k for each implicit non-stress state key.
    dRstate_dsigma : dict[str, ndarray]
        ∂R_qj / ∂σ for each implicit non-stress key j.
    dRstate_ddlambda : dict[str, ndarray]
        ∂R_qj / ∂Δλ for each implicit non-stress key j.
    dRstate_dstate : dict[str, dict[str, ndarray]]
        ∂R_qj / ∂q_k (row j → col k → array).
    full : ndarray, shape (n_unknown, n_unknown)
        Full Jacobian matrix.
    """

    layout: ResidualLayout

    dRsigma_dsigma: anp.ndarray
    dRsigma_ddlambda: anp.ndarray
    dRsigma_dstate: dict

    dRdlambda_dsigma: anp.ndarray
    dRdlambda_ddlambda: float
    dRdlambda_dstate: dict

    dRstate_dsigma: dict
    dRstate_ddlambda: dict
    dRstate_dstate: dict

    full: anp.ndarray

    def iter_blocks(self) -> Iterator[tuple[str, anp.ndarray]]:
        """Iterate over all non-empty named blocks as ``(name, array)`` pairs.

        Scalar blocks are yielded as-is; dict blocks are expanded with
        ``"block_name::key"`` labels.

        Useful for :func:`compare_jacobians`.
        """
        ntens = self.layout.ntens

        yield "dRsigma_dsigma",   self.dRsigma_dsigma
        yield "dRsigma_ddlambda", self.dRsigma_ddlambda
        yield "dRdlambda_dsigma",    self.dRdlambda_dsigma
        yield "dRdlambda_ddlambda",  np.atleast_1d(self.dRdlambda_ddlambda)

        for name, d in [
            ("dRsigma_dstate",   self.dRsigma_dstate),
            ("dRdlambda_dstate",    self.dRdlambda_dstate),
            ("dRstate_dsigma",  self.dRstate_dsigma),
            ("dRstate_ddlambda", self.dRstate_ddlambda),
        ]:
            for key, arr in d.items():
                yield f"{name}::{key}", np.asarray(arr)

        for row_key, col_dict in self.dRstate_dstate.items():
            for col_key, arr in col_dict.items():
                yield f"dRstate_dstate::{row_key}::{col_key}", np.asarray(arr)


def ad_jacobian_blocks(
    model, result, state_n: dict, *, stress_trial=None
) -> JacobianBlocks:
    """Compute the residual Jacobian at the converged point and decompose into blocks.

    Parameters
    ----------
    model : MaterialModel
    result : StressUpdateResult or ReturnMappingResult
        Converged result from a stress integration step.
    state_n : dict
        State at the beginning of the increment (must include ``"stress"``).
    stress_trial : array-like, optional
        Required when *result* is a :class:`~manforge.core.result.ReturnMappingResult`.

    Returns
    -------
    JacobianBlocks
    """
    from manforge.core.result import StressUpdateResult

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

    residual_fn, layout = build_residual(model, stress_trial, state_n)
    ntens = layout.ntens

    # Build the converged x vector
    q_imp = {k: state[k] for k in layout.implicit_keys}
    x_conv = layout.pack(stress, dlambda, q_imp)

    J = autograd.jacobian(residual_fn)(anp.array(x_conv))

    dRsigma_dsigma    = J[:ntens, :ntens]
    dRsigma_ddlambda  = J[:ntens, ntens]
    dRdlambda_dsigma  = J[ntens, :ntens]
    dRdlambda_ddlambda = J[ntens, ntens]

    q0 = ntens + 1
    dRsigma_dstate:    dict = {}
    dRdlambda_dstate:  dict = {}
    dRstate_dsigma:    dict = {}
    dRstate_ddlambda:  dict = {}
    dRstate_dstate:    dict = {}

    for k in layout.implicit_keys:
        sl = layout.state_slice(k)
        q_sl = slice(q0 + sl.start, q0 + sl.stop)
        dRsigma_dstate[k]   = J[:ntens, q_sl]
        dRdlambda_dstate[k] = J[ntens, q_sl]
        dRstate_dsigma[k]   = J[q_sl, :ntens]
        dRstate_ddlambda[k] = J[q_sl, ntens]
        dRstate_dstate[k]   = {}
        for k2 in layout.implicit_keys:
            sl2 = layout.state_slice(k2)
            q_sl2 = slice(q0 + sl2.start, q0 + sl2.stop)
            dRstate_dstate[k][k2] = J[q_sl, q_sl2]

    return JacobianBlocks(
        layout=layout,
        dRsigma_dsigma=dRsigma_dsigma,
        dRsigma_ddlambda=dRsigma_ddlambda,
        dRsigma_dstate=dRsigma_dstate,
        dRdlambda_dsigma=dRdlambda_dsigma,
        dRdlambda_ddlambda=dRdlambda_ddlambda,
        dRdlambda_dstate=dRdlambda_dstate,
        dRstate_dsigma=dRstate_dsigma,
        dRstate_ddlambda=dRstate_ddlambda,
        dRstate_dstate=dRstate_dstate,
        full=J,
    )
