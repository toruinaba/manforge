"""What the af_kinematic family is verified on.

``MODELS`` are the models the family's tests run the common checks on
(:func:`returnmap.verification.check_model`), each at the typical parameters
it documents.  The standard loading histories are enough for this family, so
no ``histories`` function is defined.
"""

from returnmap.models.af_kinematic.model import AFKinematic3D, AFKinematicPS, AFKinematic1D

MODELS = [
    AFKinematic3D.at_defaults(),
    AFKinematicPS.at_defaults(),
    AFKinematic1D.at_defaults(),
]
