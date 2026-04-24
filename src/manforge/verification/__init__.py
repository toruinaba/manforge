"""Verification utilities for constitutive model validation."""

from manforge.verification.compare import (
    compare_solvers,
    SolverComparisonResult,
    iter_compare_solvers,
    SolverCaseResult,
    compare_jacobians,
    JacobianComparisonResult,
)
from manforge.verification.fd_check import check_tangent, TangentCheckResult
from manforge.verification.fortran_bridge import FortranUMAT
from manforge.verification.test_cases import (
    estimate_yield_strain,
    generate_single_step_cases,
    generate_strain_history,
)
from manforge.verification.fortran_registry import (
    FortranBinding,
    verified_against_fortran,
    collect_bindings,
    check_bindings,
)
from manforge.verification.umat_crosscheck import (
    crosscheck_umat,
    UMATCrosscheckResult,
)

__all__ = [
    "compare_solvers",
    "SolverComparisonResult",
    "iter_compare_solvers",
    "SolverCaseResult",
    "compare_jacobians",
    "JacobianComparisonResult",
    "check_tangent",
    "TangentCheckResult",
    "FortranUMAT",
    "estimate_yield_strain",
    "generate_single_step_cases",
    "generate_strain_history",
    "FortranBinding",
    "verified_against_fortran",
    "collect_bindings",
    "check_bindings",
    "crosscheck_umat",
    "UMATCrosscheckResult",
]
