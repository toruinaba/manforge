"""Minimal concrete subclasses of MaterialModel for unit tests.

These stubs let tests instantiate MaterialModel without providing
a full material model implementation.
"""

from manforge.core.material import MaterialModel
from manforge.core.dimension import SOLID_3D, PLANE_STRESS, UNIAXIAL_1D


class _Stub3D(MaterialModel):
    """Concrete stub for full-rank stress state operator tests."""
    param_names = []

    def __init__(self, dimension=SOLID_3D):
        super().__init__(dimension=dimension)

    def elastic_stiffness(self, state=None):
        raise NotImplementedError

    def yield_function(self, state):
        raise NotImplementedError

    def update_state(self, dlambda, state_new, state_n, *, stress_trial=None, strain_inc=None):
        raise NotImplementedError


class _StubPS(MaterialModel):
    """Concrete stub for plane-stress operator tests."""
    param_names = []

    def __init__(self, dimension=PLANE_STRESS):
        super().__init__(dimension=dimension)

    def elastic_stiffness(self, state=None):
        raise NotImplementedError

    def yield_function(self, state):
        raise NotImplementedError

    def update_state(self, dlambda, state_new, state_n, *, stress_trial=None, strain_inc=None):
        raise NotImplementedError


class _Stub1D(MaterialModel):
    """Concrete stub for uniaxial (1D) operator tests."""
    param_names = []

    def __init__(self, dimension=UNIAXIAL_1D):
        super().__init__(dimension=dimension)

    def elastic_stiffness(self, state=None):
        raise NotImplementedError

    def yield_function(self, state):
        raise NotImplementedError

    def update_state(self, dlambda, state_new, state_n, *, stress_trial=None, strain_inc=None):
        raise NotImplementedError


class _StubWithParams(MaterialModel):
    """Stub with param_names to test the params property."""
    param_names = ["a", "b"]

    def __init__(self, a=1.0, b=2.0, **kwargs):
        super().__init__(dimension=SOLID_3D)
        self.a = a
        self.b = b

    def elastic_stiffness(self, state=None):
        raise NotImplementedError

    def yield_function(self, state):
        raise NotImplementedError

    def update_state(self, dlambda, state_new, state_n, *, stress_trial=None, strain_inc=None):
        raise NotImplementedError
