"""Verification utilities for constitutive model validation."""

from manforge.verification.comparator_base import (
    Comparator,
    CaseResult,
    ComparisonResult,
)
from manforge.verification.jacobian import (
    JacobianBlocks,
    JacobianComparisonResult,
    JacobianChecker,
    HandBlockCheck,
    HandBlockReport,
)
from manforge.verification.tangent import TangentChecker, TangentCheckResult
from manforge.simulation.integrator.fortran_module import FortranModule
from manforge.verification.test_cases import (
    estimate_yield_strain,
    generate_strain_history,
    make_increment,
)
from manforge.core.material.fortran_binding import (
    FortranBinding,
    verified_against_fortran,
    collect_bindings,
)
from manforge.verification.fortran_check import check_bindings
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
    "JacobianComparisonResult",
    "JacobianChecker",
    "HandBlockCheck",
    "HandBlockReport",
    # Finite-difference tangent check
    "TangentChecker",
    "TangentCheckResult",
    # Fortran interface
    "FortranModule",
    # Test case generation
    "estimate_yield_strain",
    "generate_strain_history",
    "make_increment",
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
