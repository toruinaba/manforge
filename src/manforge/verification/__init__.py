"""Verification utilities for constitutive model validation."""

from manforge.verification.comparator_base import (
    Comparator,
    CaseResult,
    ComparisonResult,
)
from manforge.verification.solver_crosscheck import (
    SolverCrosscheck,
    SolverCaseResult,
    SolverCrosscheckResult,
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
from manforge.verification.crosscheck_driver import (
    CrosscheckCaseResult,
    CrosscheckResult,
    CrosscheckStrainDriver,
    CrosscheckStressDriver,
)

__all__ = [
    # Comparator base
    "Comparator",
    "CaseResult",
    "ComparisonResult",
    # Python-vs-Python or integrator-vs-integrator single-step crosscheck
    "SolverCrosscheck",
    "SolverCaseResult",
    "SolverCrosscheckResult",
    "compare_jacobians",
    "JacobianComparisonResult",
    # Finite-difference tangent check
    "check_tangent",
    "TangentCheckResult",
    # Fortran interface
    "FortranUMAT",
    # Test case generation
    "estimate_yield_strain",
    "generate_single_step_cases",
    "generate_strain_history",
    # Fortran binding registry
    "FortranBinding",
    "verified_against_fortran",
    "collect_bindings",
    "check_bindings",
    # Multi-step crosscheck harness
    "CrosscheckCaseResult",
    "CrosscheckResult",
    "CrosscheckStrainDriver",
    "CrosscheckStressDriver",
]
