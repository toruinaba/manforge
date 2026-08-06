"""YUKinematicProj: the projected stagnation-surface variant of YUKinematic.

These tests pin the inheritance contract, not the physics -- the projected
update itself is not implemented yet.  What matters here is that the variant
differs from the published formulation in exactly one method: if a dimension
class stopped inheriting the shared residual / Jacobian / NR machinery, a
comparison between the two formulations would silently measure that difference
instead of the stagnation update.
"""

import numpy as np
import pytest

from manforge.core.dimension import PLANE_STRESS_P, SOLID_3D, UNIAXIAL_1D
from manforge.models import (
    YUKinematic1D,
    YUKinematic3D,
    YUKinematicPS,
    YUKinematicProj1D,
    YUKinematicProj3D,
    YUKinematicProjPS,
)

PARAMS = dict(
    E=206_000, nu=0.3, Y=360.0, C_1=2000.0, C_2=200.0,
    B=435.0, Rsat=255.0, k=26.0, b=66.0, h=0.4, Ea=159_000, xi=61.0,
)

# (variant, published counterpart, expected dimension)
PAIRS = [
    (YUKinematicProj3D, YUKinematic3D, SOLID_3D),
    (YUKinematicProjPS, YUKinematicPS, PLANE_STRESS_P),
    (YUKinematicProj1D, YUKinematic1D, UNIAXIAL_1D),
]


@pytest.mark.parametrize("variant,published,dim", PAIRS)
def test_dimension_and_state_match_published(variant, published, dim):
    """State layout and stress state must be identical to the published class.

    The two formulations are compared step by step, so a differing state layout
    would make the comparison meaningless rather than merely wrong.
    """
    v = variant(**PARAMS)
    p = published(**PARAMS)
    assert v.dimension is dim
    assert list(v.state_names) == list(p.state_names)
    assert list(v.implicit_state_names) == list(p.implicit_state_names)


@pytest.mark.parametrize("variant,published,_dim", PAIRS)
def test_k_eff_matches_published(variant, published, _dim):
    """k_eff carries the norm-form vs quadratic-form dlambda scaling."""
    assert variant(**PARAMS).k_eff == published(**PARAMS).k_eff


@pytest.mark.parametrize("variant,published,_dim", PAIRS)
def test_only_stagnation_update_is_overridden(variant, published, _dim):
    """The variant must override _stagnation_update and nothing else.

    Any second override is a place where the two formulations could drift
    apart for reasons unrelated to the stagnation surface.
    """
    own = {
        name for name, attr in vars(variant).items()
        if not name.startswith("__") and callable(attr)
    }
    assert own == {"_stagnation_update"}


@pytest.mark.parametrize("variant,published,_dim", PAIRS)
def test_shared_machinery_is_inherited(variant, published, _dim):
    """Residual, Jacobian and tangent code must be the published objects."""
    v = variant(**PARAMS)
    for name in ("update_state", "state_residual", "yield_function",
                 "elastic_stiffness", "_calc_E_factor"):
        assert getattr(type(v), name) is getattr(published, name), (
            f"{name} is not the inherited implementation"
        )


def test_stagnation_update_delegates_to_single_implementation():
    """All three dimensions must route to the same module-level function.

    The override is a two-line delegation precisely so the projected update
    exists once; three copies could diverge.
    """
    from manforge.models import yu_kinematic_proj as mod

    calls = []
    original = mod._stagnation_update_proj
    mod._stagnation_update_proj = lambda model, *a: calls.append(type(model).__name__)
    try:
        for variant, _published, _dim in PAIRS:
            m = variant(**PARAMS)
            ntens = m.dimension.ntens
            m._stagnation_update(1.0, 0.0, np.zeros(ntens), 1.0, np.zeros(ntens), 1e-3)
    finally:
        mod._stagnation_update_proj = original

    assert calls == ["YUKinematicProj3D", "YUKinematicProjPS", "YUKinematicProj1D"]


@pytest.mark.parametrize("variant,_published,_dim", PAIRS)
def test_stagnation_update_not_implemented_yet(variant, _published, _dim):
    m = variant(**PARAMS)
    ntens = m.dimension.ntens
    with pytest.raises(NotImplementedError):
        m._stagnation_update(1.0, 0.0, np.zeros(ntens), 1.0, np.zeros(ntens), 1e-3)
