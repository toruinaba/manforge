"""Generic solver-vs-solver and Jacobian comparison utilities.

Provides :func:`compare_solvers` and :func:`compare_jacobians` for
verifying that two constitutive-model implementations agree.

A *solver* is any callable with the signature::

    solver(model, strain_inc, stress_n, state_n) -> StressUpdateResult
"""

from dataclasses import dataclass, field

import numpy as np

_EPS = 1e-300  # zero-division guard — no physical meaning


# ---------------------------------------------------------------------------
# compare_solvers
# ---------------------------------------------------------------------------

@dataclass
class SolverComparisonResult:
    """Result of comparing two solvers across a set of test cases.

    Attributes
    ----------
    passed : bool
        ``True`` if every test case is within tolerance.
    n_cases : int
        Number of test cases evaluated.
    n_passed : int
        Number of test cases that passed.
    max_stress_rel_err : float
        Maximum relative stress error across all cases.
    max_tangent_rel_err : float
        Maximum relative tangent error across all cases.
    max_state_rel_err : dict[str, float]
        Per state-variable maximum relative error across all cases.
    max_trial_rel_err : float
        Maximum relative stress_trial error across all cases.
    details : list[dict]
        Per-case records.  Each dict contains:

        ``"case_index"``, ``"stress_rel_err"``, ``"tangent_rel_err"``,
        ``"state_rel_err"`` (dict), ``"trial_rel_err"``,
        ``"is_plastic_match"``, ``"elastic_branch_match"``, ``"passed"``.

        Optional keys (when requested): ``"n_iterations_a"``,
        ``"n_iterations_b"``, ``"residual_history_a"``,
        ``"residual_history_b"``.
    """

    passed: bool
    n_cases: int
    n_passed: int
    max_stress_rel_err: float
    max_tangent_rel_err: float
    max_state_rel_err: dict = field(default_factory=dict)
    max_trial_rel_err: float = 0.0
    details: list = field(default_factory=list)


def compare_solvers(
    model,
    solver_a,
    solver_b,
    test_cases: list,
    *,
    stress_tol: float = 1e-6,
    tangent_tol: float = 1e-5,
    state_tol: float = 1e-6,
    check_state: bool = True,
    check_residual_history: bool = False,
    check_n_iterations: bool = False,
    check_is_plastic: bool = True,
) -> SolverComparisonResult:
    """Compare two solvers across a set of test cases.

    Parameters
    ----------
    model : MaterialModel
        Constitutive model instance — passed as first argument to each solver.
    solver_a : callable
        Reference solver: ``(model, strain_inc, stress_n, state_n)``
        → ``StressUpdateResult``.
    solver_b : callable
        Candidate solver with the same signature.
    test_cases : list[dict]
        Each dict must contain keys:

        - ``"strain_inc"`` : array-like, shape (ntens,)
        - ``"stress_n"``   : array-like, shape (ntens,)
        - ``"state_n"``    : dict
    stress_tol : float, optional
        Per-case pass threshold for relative stress error (default 1e-6).
    tangent_tol : float, optional
        Per-case pass threshold for relative tangent error (default 1e-5).
    state_tol : float, optional
        Per-case pass threshold for state relative error (default 1e-6).
    check_state : bool, optional
        Whether to include state error in pass/fail (default True).
    check_residual_history : bool, optional
        Include ``residual_history_a/b`` in detail records (default False).
    check_n_iterations : bool, optional
        Include ``n_iterations_a/b`` in detail records (default False).
    check_is_plastic : bool, optional
        Fail the case when ``is_plastic`` flags disagree (default True).

    Returns
    -------
    SolverComparisonResult
    """
    details = []
    max_stress_err = 0.0
    max_tangent_err = 0.0
    max_state_err: dict[str, float] = {}
    max_trial_err = 0.0
    n_passed = 0

    for idx, case in enumerate(test_cases):
        strain_inc = case["strain_inc"]
        stress_n   = case["stress_n"]
        state_n    = case["state_n"]

        ra = solver_a(model, strain_inc, stress_n, state_n)
        rb = solver_b(model, strain_inc, stress_n, state_n)

        stress_a  = np.asarray(ra.stress,  dtype=float)
        stress_b  = np.asarray(rb.stress,  dtype=float)
        ddsdde_a  = np.asarray(ra.ddsdde,  dtype=float)
        ddsdde_b  = np.asarray(rb.ddsdde,  dtype=float)
        trial_a   = np.asarray(ra.stress_trial, dtype=float)
        trial_b   = np.asarray(rb.stress_trial, dtype=float)

        stress_rel_err  = float(np.max(np.abs(stress_b  - stress_a)  / (np.abs(stress_a)  + _EPS)))
        tangent_rel_err = float(np.max(np.abs(ddsdde_b  - ddsdde_a)  / (np.abs(ddsdde_a)  + _EPS)))
        trial_rel_err   = float(np.max(np.abs(trial_b   - trial_a)   / (np.abs(trial_a)   + _EPS)))

        is_plastic_match    = bool(ra.is_plastic == rb.is_plastic)
        elastic_branch_match = bool((ra.return_mapping is None) == (rb.return_mapping is None))

        # state comparison
        state_rel_err: dict[str, float] = {}
        if check_state:
            for key in ra.state:
                va = np.asarray(ra.state[key], dtype=float)
                vb = np.asarray(rb.state.get(key, np.zeros_like(va)), dtype=float)
                state_rel_err[key] = float(np.max(np.abs(vb - va) / (np.abs(va) + _EPS)))

        case_passed = (
            stress_rel_err  <= stress_tol
            and tangent_rel_err <= tangent_tol
        )
        if check_state and state_rel_err:
            case_passed = case_passed and all(v <= state_tol for v in state_rel_err.values())
        if check_is_plastic:
            case_passed = case_passed and is_plastic_match

        rec = {
            "case_index":          idx,
            "stress_rel_err":      stress_rel_err,
            "tangent_rel_err":     tangent_rel_err,
            "state_rel_err":       state_rel_err,
            "trial_rel_err":       trial_rel_err,
            "is_plastic_match":    is_plastic_match,
            "elastic_branch_match": elastic_branch_match,
            "passed":              case_passed,
        }
        if check_n_iterations:
            rec["n_iterations_a"] = ra.n_iterations
            rec["n_iterations_b"] = rb.n_iterations
        if check_residual_history:
            rec["residual_history_a"] = ra.residual_history
            rec["residual_history_b"] = rb.residual_history

        details.append(rec)

        max_stress_err  = max(max_stress_err,  stress_rel_err)
        max_tangent_err = max(max_tangent_err, tangent_rel_err)
        max_trial_err   = max(max_trial_err,   trial_rel_err)
        for key, val in state_rel_err.items():
            max_state_err[key] = max(max_state_err.get(key, 0.0), val)
        if case_passed:
            n_passed += 1

    return SolverComparisonResult(
        passed=n_passed == len(test_cases),
        n_cases=len(test_cases),
        n_passed=n_passed,
        max_stress_rel_err=max_stress_err,
        max_tangent_rel_err=max_tangent_err,
        max_state_rel_err=max_state_err,
        max_trial_rel_err=max_trial_err,
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
        Constitutive model instance.
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
