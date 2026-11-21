"""What the voce family is verified on.

``MODELS`` are the models the family's tests run the common checks on
(``returnmap.verification.check_model``).  The standard loading histories
are enough for this family, so no ``histories`` function is defined.
``UMATS`` are the Fortran ports: ``returnmap build --family mymodels``
compiles them, and ``check_model`` compares a built one with the Python
model step by step.
"""

from returnmap.verification import Umat

from mymodels.voce.model import J2Voce1D, J2Voce3D

MODELS = [
    J2Voce3D.at_defaults(),
    J2Voce1D.at_defaults(),
]



def _umat_failure(ret):
    """``voce_3d`` returns ``(stress, ep, ddsdde, converged)``."""
    return None if int(ret[3]) == 1 else "the UMAT's Newton iteration did not converge"


UMATS = [
    Umat(
        J2Voce3D, module="voce_3d", sources=["fortran/voce_3d.f90"],
        # Without this a step the UMAT did not solve would be compared as if
        # it were a solution.
        integrator_options={"parse_umat_failure": _umat_failure},
    ),
]
