"""What the j2_damage family is verified on.

``MODELS`` are the models the family's tests run the common checks on
(:func:`returnmap.verification.check_model`), each at the typical parameters
it documents.  The standard loading histories are enough for this family, so
no ``histories`` function is defined.
"""

from returnmap.models.j2_damage.damage import (
    J2IsotropicDamage1D, J2IsotropicDamage3D, J2IsotropicDamageCrackClosure1D,
    J2IsotropicDamageCrackClosure3D, J2IsotropicDamageCrackClosurePS,
    J2IsotropicDamagePS,
)
from returnmap.models.j2_damage.unilateral import (
    J2IsotropicDamageUnilateral1D, J2IsotropicDamageUnilateral3D,
    J2IsotropicDamageUnilateralPS,
)

MODELS = [
    J2IsotropicDamage3D.at_defaults(),
    J2IsotropicDamagePS.at_defaults(),
    J2IsotropicDamage1D.at_defaults(),
    J2IsotropicDamageCrackClosure3D.at_defaults(),
    J2IsotropicDamageCrackClosurePS.at_defaults(),
    J2IsotropicDamageCrackClosure1D.at_defaults(),
    J2IsotropicDamageUnilateral3D.at_defaults(),
    J2IsotropicDamageUnilateralPS.at_defaults(),
    J2IsotropicDamageUnilateral1D.at_defaults(),
]
