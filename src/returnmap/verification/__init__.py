"""Checks for validating a constitutive model and its Fortran UMAT.

Every check is a function that returns a report: ``report.passed`` is the
verdict and ``report.summary()`` the table behind it, so a test reads
``assert report.passed, report.summary()`` whichever check it uses.

==========================  ====================================================
:func:`check_model`         every check below that applies to a model, in one
                            call
:func:`check_hand_blocks`   hand-derived Jacobian blocks against autodiff
:func:`check_solution`      a hand-written return mapping's solutions against
                            the framework residual
:func:`check_tangent`       a consistent tangent against central differences
:func:`crosscheck`          two integrators over one loading history
:func:`compare_jacobians`   the Jacobian at two solutions of one step
:func:`check_bindings`      bound model methods against Fortran subroutines
==========================  ====================================================

All of them measure an error the same way — the largest difference relative
to the largest magnitude of the reference, ``max|candidate − reference| /
max|reference|``, over whole arrays — and pass when it is ``<= tol``.  No
constant with a unit enters, so a tolerance does not depend on the units a
model is written in.  (The residual rows of :func:`check_solution` have no
reference array; they are measured against the size of the terms the row is
made of.)

Checks only.  What they check is declared elsewhere: ``hand_block`` and
``verified_against_fortran`` on the model (:mod:`returnmap.core`),
``FortranModule``, ``JacobianBlocks`` and ``hand_jacobian`` with the solver
(:mod:`returnmap.simulation`).
"""

from returnmap.verification.jacobian import (
    HandBlockCheck,
    HandBlockReport,
    JacobianComparisonReport,
    check_hand_blocks,
    compare_jacobians,
)
from returnmap.verification.solution import (
    SolutionRowCheck,
    SolutionReport,
    check_solution,
)
from returnmap.verification.tangent import TangentReport, check_tangent
from returnmap.verification.test_cases import (
    estimate_yield_strain,
    generate_strain_history,
    make_increment,
)
from returnmap.verification.fortran_check import (
    BindingCheck,
    BindingReport,
    check_bindings,
)
from returnmap.verification.crosscheck import (
    crosscheck,
    CrosscheckReport,
    CrosscheckStep,
)
from returnmap.verification.model import (
    ModelCheck,
    ModelReport,
    check_model,
    standard_histories,
)
from returnmap.verification.umat import Umat

__all__ = [
    # Every check that applies to a model
    "check_model",
    "ModelReport",
    "ModelCheck",
    "standard_histories",
    # The Fortran UMAT that belongs to a model
    "Umat",
    # Hand-derived Jacobian blocks against autodiff
    "check_hand_blocks",
    "HandBlockReport",
    "HandBlockCheck",
    # A hand-written return mapping's solutions against the framework residual
    "check_solution",
    "SolutionReport",
    "SolutionRowCheck",
    # Consistent tangent against central differences
    "check_tangent",
    "TangentReport",
    # Two integrators over one loading history
    "crosscheck",
    "CrosscheckReport",
    "CrosscheckStep",
    # The Jacobian at two solutions of one step
    "compare_jacobians",
    "JacobianComparisonReport",
    # Bound model methods against Fortran subroutines
    "check_bindings",
    "BindingReport",
    "BindingCheck",
    # Test case generation
    "estimate_yield_strain",
    "generate_strain_history",
    "make_increment",
]
