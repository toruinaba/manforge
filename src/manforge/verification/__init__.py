"""Verification utilities for constitutive model validation.

Checks only.  What they check is declared elsewhere: ``hand_block`` and
``verified_against_fortran`` on the model (:mod:`manforge.core`),
``FortranModule`` and ``JacobianBlocks`` with the solver
(:mod:`manforge.simulation`).
"""

from manforge.verification.comparator_base import (
    Comparator,
    CaseResult,
    ComparisonResult,
)
from manforge.verification.jacobian import (
    JacobianComparisonResult,
    JacobianChecker,
    HandBlockCheck,
    HandBlockReport,
)
from manforge.verification.tangent import TangentChecker, TangentCheckResult
from manforge.verification.test_cases import (
    estimate_yield_strain,
    generate_strain_history,
    make_increment,
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
    # Jacobian comparison and hand-block checks
    "JacobianComparisonResult",
    "JacobianChecker",
    "HandBlockCheck",
    "HandBlockReport",
    # Finite-difference tangent check
    "TangentChecker",
    "TangentCheckResult",
    # Test case generation
    "estimate_yield_strain",
    "generate_strain_history",
    "make_increment",
    # Fortran binding check
    "check_bindings",
    # Multi-step crosscheck harness
    "CrosscheckCaseResult",
    "CrosscheckResult",
    "CrosscheckStrainDriver",
    "CrosscheckStressDriver",
]
