"""Generic solver-vs-solver comparison utility.

Provides :func:`compare_solvers`, which compares two callable solvers
across a set of strain-increment test cases and reports element-wise
relative errors in stress and tangent.

A *solver* is any callable with the signature::

    solver(strain_inc, stress_n, state_n, params)
        -> (stress_new, state_new, ddsdde)

This covers the autodiff path (``return_mapping(model, ...)``, partially
applied), the analytical path (``return_mapping(model, ..., method='analytical')``,
partially applied), and the Fortran UMAT bridge (:class:`~manforge.verification.fortran_bridge.FortranUMAT`).
"""

from dataclasses import dataclass, field

import numpy as np


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
    details : list[dict]
        Per-case records with keys ``"case_index"``, ``"stress_rel_err"``,
        ``"tangent_rel_err"``, ``"passed"``.
    """

    passed: bool
    n_cases: int
    n_passed: int
    max_stress_rel_err: float
    max_tangent_rel_err: float
    details: list = field(default_factory=list)


def compare_solvers(
    solver_a,
    solver_b,
    test_cases: list,
    stress_tol: float = 1e-6,
    tangent_tol: float = 1e-5,
    denom_offset: float = 1.0,
) -> SolverComparisonResult:
    """Compare two solvers for a set of test cases.

    Calls both solvers with identical inputs for each test case and
    reports element-wise relative errors in stress (σ) and tangent (DDSDDE).
    Solver *a* is treated as the reference.

    Parameters
    ----------
    solver_a : callable
        Reference solver: ``(strain_inc, stress_n, state_n, params)``
        → ``(stress, state, ddsdde)``.
    solver_b : callable
        Candidate solver with the same signature.
    test_cases : list[dict]
        Each dict must contain keys:

        - ``"strain_inc"`` : array-like, shape (ntens,)
        - ``"stress_n"``   : array-like, shape (ntens,)
        - ``"state_n"``    : dict
        - ``"params"``     : dict
    stress_tol : float, optional
        Per-case pass threshold for relative stress error (default 1e-6).
    tangent_tol : float, optional
        Per-case pass threshold for relative tangent error (default 1e-5).
    denom_offset : float, optional
        Small additive offset in the denominator to avoid division by near
        zero on low-magnitude entries.  Should match the stress unit scale
        (default 1.0, suitable for MPa).

    Returns
    -------
    SolverComparisonResult
    """
    details = []
    max_stress_err = 0.0
    max_tangent_err = 0.0
    n_passed = 0

    for idx, case in enumerate(test_cases):
        strain_inc = case["strain_inc"]
        stress_n   = case["stress_n"]
        state_n    = case["state_n"]
        params     = case["params"]

        stress_a, _state_a, ddsdde_a = solver_a(strain_inc, stress_n, state_n, params)
        stress_b, _state_b, ddsdde_b = solver_b(strain_inc, stress_n, state_n, params)

        stress_a  = np.asarray(stress_a,  dtype=float)
        stress_b  = np.asarray(stress_b,  dtype=float)
        ddsdde_a  = np.asarray(ddsdde_a,  dtype=float)
        ddsdde_b  = np.asarray(ddsdde_b,  dtype=float)

        s_denom = np.abs(stress_a)  + denom_offset
        t_denom = np.abs(ddsdde_a)  + denom_offset

        stress_rel_err  = float(np.max(np.abs(stress_b  - stress_a)  / s_denom))
        tangent_rel_err = float(np.max(np.abs(ddsdde_b  - ddsdde_a)  / t_denom))

        case_passed = (stress_rel_err <= stress_tol) and (tangent_rel_err <= tangent_tol)

        details.append({
            "case_index":      idx,
            "stress_rel_err":  stress_rel_err,
            "tangent_rel_err": tangent_rel_err,
            "passed":          case_passed,
        })

        max_stress_err  = max(max_stress_err,  stress_rel_err)
        max_tangent_err = max(max_tangent_err, tangent_rel_err)
        if case_passed:
            n_passed += 1

    return SolverComparisonResult(
        passed=n_passed == len(test_cases),
        n_cases=len(test_cases),
        n_passed=n_passed,
        max_stress_rel_err=max_stress_err,
        max_tangent_rel_err=max_tangent_err,
        details=details,
    )
