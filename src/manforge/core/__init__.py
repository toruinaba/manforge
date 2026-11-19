"""Everything a model definition needs.

The model base class and its declarations (state fields, parameters'
metadata, hand-derived blocks, Fortran bindings), the dimensions, and the
result types a hook returns or receives.
"""

from manforge.core.state import (
    Implicit, Explicit, State, Increment, NTENS, SCALAR,
    StateResidual, StateUpdate, DlambdaResidual, FlowVector,
)
# dimension before material: material.base needs it, and dimension pulls in
# manforge.utils, which must not find this package half-initialised.
from manforge.core.dimension import (
    StressDimension,
    SOLID_3D,
    PLANE_STRAIN,
    PLANE_STRESS,
    PLANE_STRESS_P,
    UNIAXIAL_1D,
    P_PLANE_STRESS,
)
from manforge.core.result import ReturnMappingResult, StressUpdateResult
from manforge.core.material import (
    MaterialModel,
    ParamMeta,
    HandBlock,
    hand_block,
    FortranBinding,
    verified_against_fortran,
)

__all__ = [
    # model
    "MaterialModel",
    "ParamMeta",
    # state declarations and the objects hooks exchange
    "Implicit",
    "Explicit",
    "NTENS",
    "SCALAR",
    "State",
    "Increment",
    "StateResidual",
    "StateUpdate",
    "DlambdaResidual",
    "FlowVector",
    # dimensions
    "StressDimension",
    "SOLID_3D",
    "PLANE_STRAIN",
    "PLANE_STRESS",
    "PLANE_STRESS_P",
    "UNIAXIAL_1D",
    "P_PLANE_STRESS",
    # results
    "ReturnMappingResult",
    "StressUpdateResult",
    # declarations checked by manforge.verification
    "HandBlock",
    "hand_block",
    "FortranBinding",
    "verified_against_fortran",
]
