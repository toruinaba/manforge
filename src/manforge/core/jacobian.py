"""Jacobian block decomposition utilities for return-mapping residual systems.

Provides :class:`JacobianBlocks` and :func:`ad_jacobian_blocks`, which compute
the Jacobian of the return-mapping residual system at a converged point and
decompose it into named blocks indexed by physical quantity and state-variable
name.

This enables step-by-step verification of analytical derivatives against
AD-computed values during constitutive model development.

Usage
-----
::

    from manforge.core.stress_update import stress_update
    from manforge.core.jacobian import ad_jacobian_blocks
    import numpy.testing as npt

    result = stress_update(model, deps, stress_n, state_n)
    jac = ad_jacobian_blocks(model, result, state_n)

    # Inspect AD-computed blocks
    print(jac.dyield_dsigma)    # flow direction n = df/dσ
    print(jac.dstate_ddlambda)  # {"alpha": ..., "ep": ...}

    # Compare against your analytical formula
    npt.assert_allclose(jac.dyield_dsigma, my_n, rtol=1e-8)
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


@dataclass
class JacobianBlocks:
    """Named blocks of the return-mapping residual Jacobian at the converged point.

    The residual system has the block structure::

        x = [σ (ntens),  Δλ (1),  q_flat (n_state)]

        ∂R/∂x = ┌ dstress_dsigma    dstress_ddlambda    dstress_dstate ┐
                 │ dyield_dsigma     dyield_ddlambda     dyield_dstate  │
                 └ dstate_dsigma     dstate_ddlambda     dstate_dstate  ┘

    For reduced-hardening models (ntens+1 system), state blocks are ``None``.

    Attributes
    ----------
    dstress_dsigma : jnp.ndarray, shape (ntens, ntens)
        ∂R_stress/∂σ
    dstress_ddlambda : jnp.ndarray, shape (ntens,)
        ∂R_stress/∂Δλ  (= C n at convergence)
    dyield_dsigma : jnp.ndarray, shape (ntens,)
        ∂R_yield/∂σ  (= flow direction n = ∂f/∂σ)
    dyield_ddlambda : jnp.ndarray, scalar
        ∂R_yield/∂Δλ  (= ∂f/∂Δλ, related to hardening modulus)
    dstress_dstate : dict[str, jnp.ndarray] or None
        ∂R_stress/∂q per state variable.  None for explicit models.
    dyield_dstate : dict[str, jnp.ndarray] or None
        ∂R_yield/∂q per state variable.  None for explicit models.
    dstate_dsigma : dict[str, jnp.ndarray] or None
        ∂R_state/∂σ per state variable.  None for explicit models.
    dstate_ddlambda : dict[str, jnp.ndarray] or None
        ∂R_state/∂Δλ per state variable.  None for explicit models.
    dstate_dstate : dict[str, dict[str, jnp.ndarray]] or None
        ∂R_state_i/∂q_j per state-variable pair.  None for explicit models.
    full : jnp.ndarray, shape (N, N)
        The full assembled Jacobian matrix (N = ntens+1 or ntens+1+n_state).
    """

    dstress_dsigma: jnp.ndarray
    dstress_ddlambda: jnp.ndarray
    dyield_dsigma: jnp.ndarray
    dyield_ddlambda: jnp.ndarray
    dstress_dstate: dict | None
    dyield_dstate: dict | None
    dstate_dsigma: dict | None
    dstate_ddlambda: dict | None
    dstate_dstate: dict | None
    full: jnp.ndarray


def ad_jacobian_blocks(
    model, result, state_n: dict, *, stress_trial: jnp.ndarray | None = None
) -> JacobianBlocks:
    """Compute the residual Jacobian at the converged point and decompose into blocks.

    For reduced-hardening models, builds the (ntens+1)×(ntens+1) residual
    Jacobian.  For augmented-hardening models, builds the full
    (ntens+1+n_state)×(ntens+1+n_state) augmented system Jacobian.

    State-related blocks are decomposed by variable name using the ordering
    established by :func:`jax.flatten_util.ravel_pytree` (alphabetical).

    Parameters
    ----------
    model : MaterialModel
        Constitutive model instance.
    result : StressUpdateResult or ReturnMappingResult
        Converged result from :func:`~manforge.core.stress_update.stress_update`
        or :func:`~manforge.core.stress_update.return_mapping`.
        Must be a plastic step (``is_plastic=True`` / ``return_mapping`` not None).
    state_n : dict
        Internal state at the *beginning* of the increment (step n).
    stress_trial : jnp.ndarray, optional
        Elastic trial stress, shape ``(ntens,)``.  Required when *result* is a
        bare :class:`~manforge.core.stress_update.ReturnMappingResult`;
        automatically extracted when *result* is a
        :class:`~manforge.core.stress_update.StressUpdateResult`.

    Returns
    -------
    JacobianBlocks
        All Jacobian blocks at the converged point.
    """
    from manforge.core.stress_update import StressUpdateResult
    if isinstance(result, StressUpdateResult):
        if result.return_mapping is None:
            # Elastic step — evaluate Jacobian at trial stress with dlambda=0
            stress = result.stress_trial
            dlambda = jnp.zeros(())
            stress_trial = result.stress_trial
            state = result.state
        else:
            stress = result.return_mapping.stress
            dlambda = result.return_mapping.dlambda
            stress_trial = result.stress_trial
            state = result.return_mapping.state
    else:
        # ReturnMappingResult — stress_trial must be supplied explicitly
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


# ---------------------------------------------------------------------------
# Reduced path — (ntens+1) × (ntens+1) system
# ---------------------------------------------------------------------------

def _jacobian_blocks_reduced(
    model, stress, state, dlambda, stress_trial, C, state_n, ntens
):
    """Build JacobianBlocks for the reduced (ntens+1) residual system."""
    from manforge.core.residual import make_reduced_residual

    residual_fn = make_reduced_residual(model, stress_trial, C, state_n)
    x_conv = jnp.concatenate([stress, dlambda.reshape(1)])
    J = jax.jacobian(residual_fn)(x_conv)  # (ntens+1, ntens+1)

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


# ---------------------------------------------------------------------------
# Augmented path — (ntens+1+n_state) × (ntens+1+n_state) system
# ---------------------------------------------------------------------------

def _jacobian_blocks_augmented(
    model, stress, state, dlambda, stress_trial, C, state_n, ntens
):
    """Build JacobianBlocks for the augmented (ntens+1+n_state) residual system."""
    from manforge.core.residual import make_augmented_residual

    residual_fn, n_state, unflatten_fn = make_augmented_residual(
        model, stress_trial, C, state_n
    )

    flat_state, _ = ravel_pytree(state)
    x_conv = jnp.concatenate([stress, dlambda.reshape(1), flat_state])
    J = jax.jacobian(residual_fn)(x_conv)  # (ntens+1+n_state, ntens+1+n_state)

    # Build variable-name → slice mapping from state dict ordering
    # ravel_pytree uses alphabetical key order
    state_keys = sorted(state.keys())
    slices = _build_state_slices(state, state_keys)

    # Extract fixed blocks
    dstress_dsigma   = J[:ntens, :ntens]
    dstress_ddlambda = J[:ntens, ntens]
    dyield_dsigma    = J[ntens, :ntens]
    dyield_ddlambda  = J[ntens, ntens]

    # Extract state blocks decomposed by variable name
    dstress_dstate: dict[str, jnp.ndarray] = {}
    dyield_dstate:  dict[str, jnp.ndarray] = {}
    dstate_dsigma:  dict[str, jnp.ndarray] = {}
    dstate_ddlambda: dict[str, jnp.ndarray] = {}
    dstate_dstate:  dict[str, dict[str, jnp.ndarray]] = {}

    q0 = ntens + 1  # start of state block in x

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


def _build_state_slices(state: dict, keys: list[str]) -> dict[str, slice]:
    """Build a mapping from state variable name to its slice in the flat array."""
    slices = {}
    offset = 0
    for key in keys:
        val = jnp.asarray(state[key])
        size = val.size
        slices[key] = slice(offset, offset + size)
        offset += size
    return slices
