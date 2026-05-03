"""Python-solver-vs-Python-solver crosscheck and Jacobian comparison.

:class:`SolverCrosscheck` compares two Python solvers across a set of
independent test cases.  It is a :class:`~manforge.verification.Comparator`
subclass: configuration (solver_a, solver_b, tolerances) is fixed in
``__init__``; ``iter_run(model, test_cases)`` drives the comparison.

A *solver* is any callable with the signature::

    solver(model, strain_inc, stress_n, state_n) -> StressUpdateResult

Usage
-----
::

    cs = SolverCrosscheck(solver_a, solver_b)
    result = cs.run(model, test_cases)        # → SolverCrosscheckResult
    for case in cs.iter_run(model, test_cases):   # step-by-step
        if not case.passed:
            break
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np

from manforge.verification.comparator_base import (
    CaseResult,
    ComparisonResult,
    Comparator,
    _array_rel_err,
    _state_rel_err,
    _stress_rel_err,
    _tangent_rel_err,
)

if TYPE_CHECKING:
    from manforge.core.stress_update import StressUpdateResult


# ---------------------------------------------------------------------------
# Per-case and aggregate result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SolverCaseResult(CaseResult):
    """Per-case result from :class:`SolverCrosscheck`.

    Extends :class:`~manforge.verification.CaseResult` with raw
    :class:`~manforge.core.stress_update.StressUpdateResult` objects so the
    caller can run :func:`compare_jacobians` on failing cases.
    """

    result_a: StressUpdateResult | None = None
    result_b: StressUpdateResult | None = None
    trial_rel_err: float = 0.0
    is_plastic_match: bool = True
    elastic_branch_match: bool = True
    # P2: inner-NR trajectory (a/b mirror result_a/result_b).
    # a_converged / b_converged live in base CaseResult (P3).
    a_n_iterations: int = 0
    a_residual_history: list = field(default_factory=list)
    b_n_iterations: int = 0
    b_residual_history: list = field(default_factory=list)


@dataclass
class SolverCrosscheckResult(ComparisonResult):
    """Aggregate result from :class:`SolverCrosscheck`.

    Extends :class:`~manforge.verification.ComparisonResult` with
    ``max_trial_rel_err``.
    """

    max_trial_rel_err: float = 0.0


# ---------------------------------------------------------------------------
# SolverCrosscheck
# ---------------------------------------------------------------------------

class SolverCrosscheck(Comparator):
    """Compare two Python solvers across a set of independent test cases.

    Parameters
    ----------
    solver_a : callable
        Reference solver: ``(model, strain_inc, stress_n, state_n)``
        → :class:`~manforge.core.stress_update.StressUpdateResult`.
    solver_b : callable
        Candidate solver with the same signature.
    stress_tol : float, optional
        Per-case pass threshold for relative stress error (default 1e-6).
    tangent_tol : float, optional
        Per-case pass threshold for relative tangent error (default 1e-5).
    state_tol : float, optional
        Per-case pass threshold for state relative error (default 1e-6).
    check_state : bool, optional
        Whether to include state error in pass/fail (default True).
    check_is_plastic : bool, optional
        Fail the case when ``is_plastic`` flags disagree (default True).

    Examples
    --------
    ::

        cs = SolverCrosscheck(solver_a, solver_b)
        result = cs.run(model, test_cases)
        assert result.passed

        # Step-by-step with early break
        for case in cs.iter_run(model, test_cases):
            if not case.passed:
                jac = compare_jacobians(model, case.result_a, case.result_b,
                                        test_cases[case.index]["state_n"])
                break
    """

    _result_cls = SolverCrosscheckResult

    def _aggregate_extra(self, cases):
        return {"max_trial_rel_err": max((c.trial_rel_err for c in cases), default=0.0)}

    def __init__(
        self,
        solver_a: Callable,
        solver_b: Callable,
        *,
        stress_tol: float = 1e-6,
        tangent_tol: float = 1e-5,
        state_tol: float = 1e-6,
        check_state: bool = True,
        check_is_plastic: bool = True,
    ) -> None:
        self.solver_a = solver_a
        self.solver_b = solver_b
        self.stress_tol = stress_tol
        self.tangent_tol = tangent_tol
        self.state_tol = state_tol
        self.check_state = check_state
        self.check_is_plastic = check_is_plastic

    def iter_run(
        self,
        model,
        test_cases: list,
    ) -> Iterator[SolverCaseResult]:
        """Yield per-case comparison results.

        Parameters
        ----------
        model : MaterialModel
        test_cases : list[dict]
            Each dict must have ``"strain_inc"``, ``"stress_n"``, ``"state_n"``.

        Yields
        ------
        SolverCaseResult
        """
        for idx, case in enumerate(test_cases):
            strain_inc = case["strain_inc"]
            stress_n   = case["stress_n"]
            state_n    = case["state_n"]

            ra = self.solver_a(model, strain_inc, stress_n, state_n)
            rb = self.solver_b(model, strain_inc, stress_n, state_n)

            stress_a = np.asarray(ra.stress, dtype=float)
            stress_b = np.asarray(rb.stress, dtype=float)
            trial_a  = np.asarray(ra.stress_trial, dtype=float)
            trial_b  = np.asarray(rb.stress_trial, dtype=float)

            stress_rel_err  = _stress_rel_err(stress_a, stress_b)
            tangent_rel_err = _tangent_rel_err(
                np.asarray(ra.ddsdde, dtype=float),
                np.asarray(rb.ddsdde, dtype=float),
            ) or 0.0
            trial_rel_err   = _array_rel_err(trial_a, trial_b)

            is_plastic_match     = bool(ra.is_plastic == rb.is_plastic)
            elastic_branch_match = bool((ra.return_mapping is None) == (rb.return_mapping is None))

            state_rel_err: dict[str, float] = (
                _state_rel_err(ra.state, rb.state) if self.check_state else {}
            )

            case_passed = stress_rel_err <= self.stress_tol and tangent_rel_err <= self.tangent_tol
            if self.check_state and state_rel_err:
                case_passed = case_passed and all(v <= self.state_tol for v in state_rel_err.values())
            if self.check_is_plastic:
                case_passed = case_passed and is_plastic_match

            yield SolverCaseResult(
                index=idx,
                stress_rel_err=stress_rel_err,
                tangent_rel_err=tangent_rel_err,
                state_rel_err=state_rel_err,
                passed=case_passed,
                result_a=ra,
                result_b=rb,
                trial_rel_err=trial_rel_err,
                is_plastic_match=is_plastic_match,
                elastic_branch_match=elastic_branch_match,
                a_n_iterations=ra.n_iterations,
                a_residual_history=list(ra.residual_history),
                a_converged=ra.converged,
                b_n_iterations=rb.n_iterations,
                b_residual_history=list(rb.residual_history),
                b_converged=rb.converged,
            )


# ---------------------------------------------------------------------------
# compare_jacobians
# ---------------------------------------------------------------------------

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
        First result (e.g. from ``method="numerical_newton"``).
    result_b : StressUpdateResult
        Second result (e.g. from ``method="user_defined"``).
    state_n : dict
        Initial state at the start of the step (before the increment).
    rtol : float, optional
        Relative tolerance for pass/fail (default 1e-8).

    Returns
    -------
    JacobianComparisonResult
    """
    from manforge.core.jacobian import ad_jacobian_blocks

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
