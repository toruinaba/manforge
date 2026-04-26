"""Solver-vs-solver and Jacobian comparison utilities.

:class:`SolverComparison` compares two Python solvers across a set of
independent test cases.  It is a :class:`~manforge.verification.Comparator`
subclass: configuration (solver_a, solver_b, tolerances) is fixed in
``__init__``; ``iter_run(model, test_cases)`` drives the comparison.

A *solver* is any callable with the signature::

    solver(model, strain_inc, stress_n, state_n) -> StressUpdateResult

Usage
-----
::

    cs = SolverComparison(solver_a, solver_b)
    result = cs.run(model, test_cases)        # → ComparisonResult
    for case in cs.iter_run(model, test_cases):   # step-by-step
        if not case.passed:
            break
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np

from manforge.verification.comparator import CaseResult, ComparisonResult, Comparator

if TYPE_CHECKING:
    from manforge.core.stress_update import StressUpdateResult

_EPS = 1e-300  # zero-division guard — no physical meaning


# ---------------------------------------------------------------------------
# Per-case and aggregate result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SolverCaseResult(CaseResult):
    """Per-case result from :class:`SolverComparison`.

    Extends :class:`~manforge.verification.CaseResult` with raw
    :class:`~manforge.core.stress_update.StressUpdateResult` objects so the
    caller can run :func:`compare_jacobians` on failing cases.
    """

    result_a: StressUpdateResult | None = None
    result_b: StressUpdateResult | None = None
    trial_rel_err: float = 0.0
    is_plastic_match: bool = True
    elastic_branch_match: bool = True


@dataclass
class SolverComparisonResult(ComparisonResult):
    """Aggregate result from :class:`SolverComparison`.

    Extends :class:`~manforge.verification.ComparisonResult` with
    ``max_trial_rel_err`` and a legacy ``details`` list for backwards
    compatibility with existing tests that inspect ``result.details``.
    """

    max_trial_rel_err: float = 0.0
    details: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# SolverComparison
# ---------------------------------------------------------------------------

class SolverComparison(Comparator):
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
    check_residual_history : bool, optional
        Include ``residual_history_a/b`` in ``details`` records (default False).
    check_n_iterations : bool, optional
        Include ``n_iterations_a/b`` in ``details`` records (default False).

    Examples
    --------
    ::

        cs = SolverComparison(solver_a, solver_b)
        result = cs.run(model, test_cases)
        assert result.passed

        # Step-by-step with early break
        for case in cs.iter_run(model, test_cases):
            if not case.passed:
                jac = compare_jacobians(model, case.result_a, case.result_b,
                                        test_cases[case.index]["state_n"])
                break
    """

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
        check_residual_history: bool = False,
        check_n_iterations: bool = False,
    ) -> None:
        self.solver_a = solver_a
        self.solver_b = solver_b
        self.stress_tol = stress_tol
        self.tangent_tol = tangent_tol
        self.state_tol = state_tol
        self.check_state = check_state
        self.check_is_plastic = check_is_plastic
        self.check_residual_history = check_residual_history
        self.check_n_iterations = check_n_iterations

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

            stress_a  = np.asarray(ra.stress,  dtype=float)
            stress_b  = np.asarray(rb.stress,  dtype=float)
            ddsdde_a  = np.asarray(ra.ddsdde,  dtype=float)
            ddsdde_b  = np.asarray(rb.ddsdde,  dtype=float)
            trial_a   = np.asarray(ra.stress_trial, dtype=float)
            trial_b   = np.asarray(rb.stress_trial, dtype=float)

            stress_rel_err  = float(np.max(np.abs(stress_b  - stress_a)  / (np.abs(stress_a)  + _EPS)))
            tangent_rel_err = float(np.max(np.abs(ddsdde_b  - ddsdde_a)  / (np.abs(ddsdde_a)  + _EPS)))
            trial_rel_err   = float(np.max(np.abs(trial_b   - trial_a)   / (np.abs(trial_a)   + _EPS)))

            is_plastic_match     = bool(ra.is_plastic == rb.is_plastic)
            elastic_branch_match = bool((ra.return_mapping is None) == (rb.return_mapping is None))

            state_rel_err: dict[str, float] = {}
            if self.check_state:
                for key in ra.state:
                    va = np.asarray(ra.state[key], dtype=float)
                    vb = np.asarray(rb.state.get(key, np.zeros_like(va)), dtype=float)
                    state_rel_err[key] = float(np.max(np.abs(vb - va) / (np.abs(va) + _EPS)))

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
            )

    def run(
        self,
        model,
        test_cases: list,
    ) -> SolverComparisonResult:
        """Compare solvers across all test cases and return aggregate result.

        Parameters
        ----------
        model : MaterialModel
        test_cases : list[dict]

        Returns
        -------
        SolverComparisonResult
        """
        cases: list[SolverCaseResult] = []
        max_s_err = 0.0
        max_t_err = 0.0
        max_st_err: dict[str, float] = {}
        max_trial_err = 0.0
        n_passed = 0
        details = []

        for cr in self.iter_run(model, test_cases):
            cases.append(cr)

            rec: dict = {
                "case_index":           cr.index,
                "stress_rel_err":       cr.stress_rel_err,
                "tangent_rel_err":      cr.tangent_rel_err,
                "state_rel_err":        cr.state_rel_err,
                "trial_rel_err":        cr.trial_rel_err,
                "is_plastic_match":     cr.is_plastic_match,
                "elastic_branch_match": cr.elastic_branch_match,
                "passed":               cr.passed,
            }
            if self.check_n_iterations:
                rec["n_iterations_a"] = cr.result_a.n_iterations
                rec["n_iterations_b"] = cr.result_b.n_iterations
            if self.check_residual_history:
                rec["residual_history_a"] = cr.result_a.residual_history
                rec["residual_history_b"] = cr.result_b.residual_history
            details.append(rec)

            max_s_err    = max(max_s_err,    cr.stress_rel_err or 0.0)
            max_t_err    = max(max_t_err,    cr.tangent_rel_err or 0.0)
            max_trial_err = max(max_trial_err, cr.trial_rel_err)
            for key, val in cr.state_rel_err.items():
                max_st_err[key] = max(max_st_err.get(key, 0.0), val)
            if cr.passed:
                n_passed += 1

        return SolverComparisonResult(
            passed=n_passed == len(test_cases),
            n_cases=len(test_cases),
            n_passed=n_passed,
            max_stress_rel_err=max_s_err,
            max_tangent_rel_err=max_t_err,
            max_state_rel_err=max_st_err,
            max_trial_rel_err=max_trial_err,
            cases=cases,
            details=details,
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
        block_errs[name] = float(np.max(np.abs(vb - va) / (np.abs(va) + _EPS)))

    for name in _DICT_BLOCKS:
        da = getattr(jac_a, name)
        db = getattr(jac_b, name)
        if da is None or db is None:
            continue
        for key in da:
            va = np.asarray(da[key], dtype=float)
            vb = np.asarray(db.get(key, np.zeros_like(va)), dtype=float)
            block_errs[f"{name}::{key}"] = float(np.max(np.abs(vb - va) / (np.abs(va) + _EPS)))

    max_err = max(block_errs.values()) if block_errs else 0.0

    return JacobianComparisonResult(
        passed=max_err <= rtol,
        blocks=block_errs,
        max_rel_err=max_err,
    )
