"""Reference models, one family per subpackage.

A family directory holds everything that belongs to one model: the model
itself, its formulation, its Fortran UMAT if it has one, and its tests.  The
names are imported from here — ``from returnmap.models import J2Isotropic3D``.
"""

from returnmap.models.j2_isotropic import (
    J2Isotropic,
    J2Isotropic3D,
    J2IsotropicPS,
    J2Isotropic1D,
)
from returnmap.models.j2_damage import (
    J2IsotropicDamage,
    J2IsotropicDamage3D,
    J2IsotropicDamagePS,
    J2IsotropicDamage1D,
    J2IsotropicDamageCrackClosure,
    J2IsotropicDamageCrackClosure3D,
    J2IsotropicDamageCrackClosurePS,
    J2IsotropicDamageCrackClosure1D,
    J2IsotropicDamageUnilateral,
    J2IsotropicDamageUnilateral3D,
    J2IsotropicDamageUnilateralPS,
    J2IsotropicDamageUnilateral1D,
)
from returnmap.models.af_kinematic import (
    AFKinematic,
    AFKinematic3D,
    AFKinematicPS,
    AFKinematic1D,
)
from returnmap.models.ow_kinematic import (
    OWKinematic,
    OWKinematic3D,
    OWKinematicPS,
    OWKinematic1D,
)
from returnmap.models.yu_kinematic import (
    YUKinematic,
    YUKinematic3D,
    YUKinematicPS,
    YUKinematic1D,
)

__all__ = [
    "J2Isotropic",
    "J2Isotropic3D",
    "J2IsotropicPS",
    "J2Isotropic1D",
    "J2IsotropicDamage",
    "J2IsotropicDamage3D",
    "J2IsotropicDamagePS",
    "J2IsotropicDamage1D",
    "J2IsotropicDamageCrackClosure",
    "J2IsotropicDamageCrackClosure3D",
    "J2IsotropicDamageCrackClosurePS",
    "J2IsotropicDamageCrackClosure1D",
    "J2IsotropicDamageUnilateral",
    "J2IsotropicDamageUnilateral3D",
    "J2IsotropicDamageUnilateralPS",
    "J2IsotropicDamageUnilateral1D",
    "AFKinematic",
    "AFKinematic3D",
    "AFKinematicPS",
    "AFKinematic1D",
    "OWKinematic",
    "OWKinematic3D",
    "OWKinematicPS",
    "OWKinematic1D",
    "YUKinematic",
    "YUKinematic3D",
    "YUKinematicPS",
    "YUKinematic1D",
]
