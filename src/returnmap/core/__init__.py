"""Everything a model definition needs.

The model base class and its declarations (state fields, parameters'
metadata, hand-derived blocks, Fortran bindings), the dimensions, and the
result types a hook returns or receives.
"""

from returnmap.core.state import (
    Implicit, Explicit, StateField, State, Increment, NTENS, SCALAR,
    StateResidual, StateUpdate, DlambdaResidual, FlowVector,
)
# dimension before material: material.base needs it, and dimension pulls in
# returnmap.utils, which must not find this package half-initialised.
from returnmap.core.dimension import (
    StressDimension,
    SOLID_3D,
    PLANE_STRAIN,
    PLANE_STRESS,
    PLANE_STRESS_P,
    UNIAXIAL_1D,
    P_PLANE_STRESS,
)
from returnmap.core.result import ReturnMappingResult, StressUpdateResult
from returnmap.core.material import (
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
    "StateField",
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
    # declarations checked by returnmap.verification
    "HandBlock",
    "hand_block",
    "FortranBinding",
    "verified_against_fortran",
]
