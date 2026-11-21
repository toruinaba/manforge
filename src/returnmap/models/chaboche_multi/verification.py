"""What the chaboche_multi family is verified on.

``MODELS`` are the models the family's tests run the common checks on
(:func:`returnmap.verification.check_model`), each at the typical parameters
it documents.  The standard loading histories are enough for this family, so
no ``histories`` function is defined.  These are Python models only — no
hand-written return mapping, no UMAT.
"""

from returnmap.models.chaboche_multi.model import (
    ChabocheKinematicMulti1D,
    ChabocheKinematicMulti3D,
    ChabocheKinematicMultiPS,
)

MODELS = [
    ChabocheKinematicMulti3D.at_defaults(),
    ChabocheKinematicMultiPS.at_defaults(),
    ChabocheKinematicMulti1D.at_defaults(),
]
