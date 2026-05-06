"""Jacobian block comparison for diagnosing failed crosschecks.

:func:`compare_jacobians` compares the residual Jacobian blocks computed by
:func:`~manforge.verification.jacobian.ad_jacobian_blocks` for two
:class:`~manforge.core.result.StressUpdateResult` objects.  It is
intended for manual use when a crosscheck fails::

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
        Block label → maximum relative error.
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

    block_errs: dict[str, float] = {}
    blocks_b = {label: arr for label, arr in jac_b.iter_blocks()}

    for label, arr_a in jac_a.iter_blocks():
        arr_a = np.asarray(arr_a, dtype=float)
        arr_b = np.asarray(blocks_b.get(label, np.zeros_like(arr_a)), dtype=float)
        block_errs[label] = _array_rel_err(arr_a, arr_b)

    max_err = max(block_errs.values()) if block_errs else 0.0

    return JacobianComparisonResult(
        passed=max_err <= rtol,
        blocks=block_errs,
        max_rel_err=max_err,
    )
