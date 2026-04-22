"""Canonical strain increment vectors reused across integration tests."""

import autograd.numpy as anp

# 4 canonical 3D (ntens=6) strain increments
DEPS_UNIAXIAL_3D    = [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0]
DEPS_EQUIBIAXIAL_3D = [2e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0]
DEPS_PURE_SHEAR_3D  = [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0]
DEPS_COMBINED_3D    = [2e-3, 0.0, 0.0, 2e-3, 0.0, 0.0]

DEPS_VEC_LIST_3D = [
    DEPS_UNIAXIAL_3D,
    DEPS_EQUIBIAXIAL_3D,
    DEPS_PURE_SHEAR_3D,
    DEPS_COMBINED_3D,
]
