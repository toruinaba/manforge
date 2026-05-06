"""Verification utilities for constitutive model validation."""

from manforge.verification.comparator_base import (
    Comparator,
    CaseResult,
    ComparisonResult,
)
from manforge.verification.jacobian import JacobianBlocks, ad_jacobian_blocks
from manforge.verification.jacobian_compare import (
    compare_jacobians,
    JacobianComparisonResult,
)
from manforge.verification.fd_check import check_tangent, TangentCheckResult
from manforge.verification.fortran_bridge import FortranModule
from manforge.verification.test_cases import (
    estimate_yield_strain,
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
    # Jacobian block decomposition and comparison
    "JacobianBlocks",
    "ad_jacobian_blocks",
    "compare_jacobians",
    "JacobianComparisonResult",
    # Finite-difference tangent check
    "check_tangent",
    "TangentCheckResult",
    # Fortran interface
    "FortranModule",
    # Test case generation
    "estimate_yield_strain",
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
