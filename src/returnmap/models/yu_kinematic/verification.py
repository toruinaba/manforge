"""What the yu_kinematic family is verified on.

``MODELS`` are the models the family's tests run the common checks on
(:func:`returnmap.verification.check_model`), each at the typical parameters
it documents.  ``histories`` adds a loading history to the standard ones.
``UMATS`` are the Fortran ports: ``returnmap build --family
returnmap.models.yu_kinematic`` compiles them, and ``check_model`` compares a
built one with the Python model step by step.
"""

from returnmap.models.yu_kinematic.model import YUKinematic1D, YUKinematic3D, YUKinematicPS
from returnmap.simulation import FieldHistory
from returnmap.verification import Umat

MODELS = [
    YUKinematic3D.at_defaults(),
    YUKinematicPS.at_defaults(),
    YUKinematic1D.at_defaults(),
]


def histories(model):
    """Reversed cycles at 2 % strain, on top of the standard histories.

    The standard histories stay within five yield strains.  That is not
    enough for this model: the yield surface's backstress has to travel
    ``B - Y`` before the hardening rate switches from ``C_1`` to ``C_2``, and
    the stagnation surface only starts to matter after a reversal from a
    strain well beyond that.
    """
    return {
        "reversed 2 %": FieldHistory.cyclic_strain(
            [0.02, -0.02, 0.02], n_per_segment=12, ntens=model.ntens,
        ),
    }


def _umat_failure(ret):
    """Both cores return ``(stress, 7 state variables, ddsdde, n_iter, converged, ...)``."""
    return None if int(ret[10]) == 1 else "the UMAT's return mapping did not converge"


UMATS = [
    Umat(YUKinematic3D, module="yu_kinematic_3d",
         sources=["fortran/yu_kinematic_3d.f90"], libraries=["lapack", "blas"],
         integrator_options={"parse_umat_failure": _umat_failure}),
    Umat(YUKinematicPS, module="yu_kinematic_ps",
         sources=["fortran/yu_kinematic_ps.f90"], libraries=["lapack", "blas"],
         integrator_options={"parse_umat_failure": _umat_failure}),
]
