"""Jacobian block comparison for diagnosing failed crosschecks.

:func:`compare_jacobians` compares the residual Jacobian blocks computed by
:func:`~manforge.core.jacobian.ad_jacobian_blocks` for two
:class:`~manforge.core.result.StressUpdateResult` objects.  It is
intended for manual use when a crosscheck fails:

::

    cc = CrosscheckStrainDriver(int_a, int_b)
    for case in cc.iter_run(load):
        if not case.passed:
            jac = compare_jacobians(model, case.result_a, case.result_b, case.state_n)
            print(jac.blocks)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manforge.verification.comparator_base import _array_rel_err


@dataclass
class JacobianComparisonResult:
    """Result of comparing Jacobian blocks from two StressUpdateResults.

    Attributes
    ----------
    passed : bool
        ``True`` if every block is within ``rtol``.
    blocks : dict[str, float]
        Block name → maximum relative error.  Dict-valued blocks use
        ``"block_name::var"`` as the key.
    max_rel_err : float
        Maximum relative error across all blocks.
    """

    passed: bool
    blocks: dict
    max_rel_err: float


def compare_jacobians(
    model,
    result_a,
    result_b,
    state_n: dict,
    *,
    rtol: float = 1e-8,
) -> JacobianComparisonResult:
    """Compare Jacobian blocks from two StressUpdateResults.

    Parameters
    ----------
    model : MaterialModel
    result_a : StressUpdateResult
        First result (e.g. from ``PythonNumericalIntegrator``).
    result_b : StressUpdateResult
        Second result (e.g. from ``PythonAnalyticalIntegrator``).
    state_n : dict
        Initial state at the start of the step (before the increment).
    rtol : float, optional
        Relative tolerance for pass/fail (default 1e-8).

    Returns
    -------
    JacobianComparisonResult
    """
    from manforge.verification.jacobian import ad_jacobian_blocks

    jac_a = ad_jacobian_blocks(model, result_a, state_n)
    jac_b = ad_jacobian_blocks(model, result_b, state_n)

    _SCALAR_BLOCKS = [
        "dstress_dsigma",
        "dstress_ddlambda",
        "dyield_dsigma",
        "dyield_ddlambda",
    ]
    _DICT_BLOCKS = [
        "dstress_dstate",
        "dyield_dstate",
        "dstate_dsigma",
        "dstate_ddlambda",
        "dstate_dstate",
    ]

    block_errs: dict[str, float] = {}

    for name in _SCALAR_BLOCKS:
        va = np.asarray(getattr(jac_a, name), dtype=float)
        vb = np.asarray(getattr(jac_b, name), dtype=float)
        block_errs[name] = _array_rel_err(va, vb)

    for name in _DICT_BLOCKS:
        da = getattr(jac_a, name)
        db = getattr(jac_b, name)
        if da is None or db is None:
            continue
        for key in da:
            va = np.asarray(da[key], dtype=float)
            vb = np.asarray(db.get(key, np.zeros_like(va)), dtype=float)
            block_errs[f"{name}::{key}"] = _array_rel_err(va, vb)

    max_err = max(block_errs.values()) if block_errs else 0.0

    return JacobianComparisonResult(
        passed=max_err <= rtol,
        blocks=block_errs,
        max_rel_err=max_err,
    )
