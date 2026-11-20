"""What the ow_kinematic family is verified on.

``MODELS`` are the models the family's tests run the common checks on
(:func:`returnmap.verification.check_model`), each at the typical parameters
it documents.  The standard loading histories are enough for this family, so
no ``histories`` function is defined.
"""

from returnmap.models.ow_kinematic.model import OWKinematic3D, OWKinematicPS, OWKinematic1D

MODELS = [
    OWKinematic3D.at_defaults(),
    OWKinematicPS.at_defaults(),
    OWKinematic1D.at_defaults(),
]
