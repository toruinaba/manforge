"""Minimal concrete subclasses of base material models for unit tests.

These stubs let tests instantiate the abstract base classes without providing
a full material model implementation.
"""

from manforge.core.material import MaterialModel3D, MaterialModelPS, MaterialModel1D
from manforge.core.stress_state import SOLID_3D


class _Stub3D(MaterialModel3D):
    """Concrete stub — lets us instantiate MaterialModel3D for operator tests."""
    param_names = []
    state_names = []

    def elastic_stiffness(self, state=None):
        raise NotImplementedError

    def yield_function(self, stress, state):
        raise NotImplementedError

    def update_state(self, dlambda, stress, state):
        raise NotImplementedError


class _StubPS(MaterialModelPS):
    """Concrete stub for MaterialModelPS operator tests."""
    param_names = []
    state_names = []

    def elastic_stiffness(self, state=None):
        raise NotImplementedError

    def yield_function(self, stress, state):
        raise NotImplementedError

    def update_state(self, dlambda, stress, state):
        raise NotImplementedError


class _Stub1D(MaterialModel1D):
    """Concrete stub for MaterialModel1D operator tests."""
    param_names = []
    state_names = []

    def elastic_stiffness(self, state=None):
        raise NotImplementedError

    def yield_function(self, stress, state):
        raise NotImplementedError

    def update_state(self, dlambda, stress, state):
        raise NotImplementedError


class _StubWithParams(MaterialModel3D):
    """Stub with param_names to test the params property."""
    param_names = ["a", "b"]
    state_names = []

    def __init__(self, a=1.0, b=2.0, **kwargs):
        super().__init__(SOLID_3D)
        self.a = a
        self.b = b

    def elastic_stiffness(self, state=None):
        raise NotImplementedError

    def yield_function(self, stress, state):
        raise NotImplementedError

    def update_state(self, dlambda, stress, state):
        raise NotImplementedError
