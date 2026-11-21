"""What the chaboche_single family is verified on.

``MODELS`` are the models the family's tests run the common checks on
(:func:`returnmap.verification.check_model`), each at the typical parameters
it documents.  The standard loading histories are enough for this family, so
no ``histories`` function is defined.  These are Python models only — no
hand-written return mapping, no UMAT.
"""

from returnmap.models.chaboche_single.model import (
    ChabocheKinematic1D,
    ChabocheKinematic3D,
    ChabocheKinematicPS,
)

MODELS = [
    ChabocheKinematic3D.at_defaults(),
    ChabocheKinematicPS.at_defaults(),
    ChabocheKinematic1D.at_defaults(),
]
