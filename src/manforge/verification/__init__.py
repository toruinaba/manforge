"""Verification utilities for constitutive model validation."""

from manforge.verification.compare import (
    compare_solvers,
    SolverComparisonResult,
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

__all__ = [
    "compare_solvers",
    "SolverComparisonResult",
    "compare_jacobians",
    "JacobianComparisonResult",
    "check_tangent",
    "TangentCheckResult",
    "FortranUMAT",
    "estimate_yield_strain",
    "generate_single_step_cases",
    "generate_strain_history",
]
