"""J2 plasticity with weakly coupled Lemaitre damage, with and without crack closure."""

from returnmap.models.j2_damage.damage import (
    J2IsotropicDamage,
    J2IsotropicDamage3D,
    J2IsotropicDamagePS,
    J2IsotropicDamage1D,
    J2IsotropicDamageCrackClosure,
    J2IsotropicDamageCrackClosure3D,
    J2IsotropicDamageCrackClosurePS,
    J2IsotropicDamageCrackClosure1D,
)
from returnmap.models.j2_damage.unilateral import (
    J2IsotropicDamageUnilateral,
    J2IsotropicDamageUnilateral3D,
    J2IsotropicDamageUnilateralPS,
    J2IsotropicDamageUnilateral1D,
)

__all__ = [
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
]
