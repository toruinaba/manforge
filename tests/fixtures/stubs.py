"""Minimal concrete subclasses of MaterialModel for unit tests.

These stubs let tests instantiate MaterialModel without providing
a full material model implementation.
"""

from manforge.core import MaterialModel, PLANE_STRESS, UNIAXIAL_1D


class _Stub3D(MaterialModel):
    """Concrete stub for full-rank stress state operator tests (any dimension)."""
    param_names = []

    def elastic_stiffness(self, state):
        raise NotImplementedError

    def yield_function(self, state):
        raise NotImplementedError

    def update_state(self, new, dlambda, inc):
        raise NotImplementedError


class _StubPS(MaterialModel):
    """Concrete stub for plane-stress operator tests."""
    param_names = []
    dimension = PLANE_STRESS

    def elastic_stiffness(self, state):
        raise NotImplementedError

    def yield_function(self, state):
        raise NotImplementedError

    def update_state(self, new, dlambda, inc):
        raise NotImplementedError


class _Stub1D(MaterialModel):
    """Concrete stub for uniaxial (1D) operator tests."""
    param_names = []
    dimension = UNIAXIAL_1D

    def elastic_stiffness(self, state):
        raise NotImplementedError

    def yield_function(self, state):
        raise NotImplementedError

    def update_state(self, new, dlambda, inc):
        raise NotImplementedError


class _StubWithParams(MaterialModel):
    """Stub with param_names to test the params property."""
    param_names = ["a", "b"]

    def elastic_stiffness(self, state):
        raise NotImplementedError

    def yield_function(self, state):
        raise NotImplementedError

    def update_state(self, new, dlambda, inc):
        raise NotImplementedError
