"""What the j2_isotropic family is verified on.

``MODELS`` are the models the family's tests run the common checks on
(:func:`returnmap.verification.check_model`), each at the typical parameters
it documents.  The standard loading histories are enough for this family, so
no ``histories`` function is defined.  ``UMATS`` are the Fortran ports:
``returnmap build --family returnmap.models.j2_isotropic`` compiles them, and
``check_model`` compares a built one with the Python model step by step.
"""

from returnmap.models.j2_isotropic.model import J2Isotropic3D, J2IsotropicPS, J2Isotropic1D
from returnmap.verification import Umat

MODELS = [
    J2Isotropic3D.at_defaults(),
    J2IsotropicPS.at_defaults(),
    J2Isotropic1D.at_defaults(),
]

UMATS = [
    Umat(J2Isotropic3D, module="j2_isotropic_3d", sources=["fortran/j2_isotropic_3d.f90"]),
]
