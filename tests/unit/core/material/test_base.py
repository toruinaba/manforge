"""Tests for MaterialModel operator methods across stress states.

Covers:
- Constructor dimension acceptance
- _hydrostatic, _dev, _vonmises operators — values and shapes
- isotropic_C — shape, symmetry, known values
- _I_vol, _I_dev — projection identity: P_vol + P_dev == I
- inner_product, deviatoric_inner_product, strain_norm — known values per dimension
"""

import math

import autograd.numpy as anp
import numpy as np
import pytest

from manforge.core.material import MaterialModel
from manforge.core.state import NTENS
from manforge.core.dimension import SOLID_3D, PLANE_STRAIN, PLANE_STRESS, UNIAXIAL_1D
from manforge.models.j2_isotropic import J2Isotropic3D, J2IsotropicPS, J2Isotropic1D
from manforge.models import (
    J2IsotropicDamage3D, J2IsotropicDamageCrackClosure3D, J2IsotropicDamageUnilateral3D,
)
from tests.fixtures.stubs import _Stub3D, _StubPS, _Stub1D, _StubWithParams


# ---------------------------------------------------------------------------
# Constructor dimension
# ---------------------------------------------------------------------------

def test_stub3d_accepts_solid_3d():
    m = _Stub3D(dimension=SOLID_3D)
    assert m.dimension is SOLID_3D
    assert m.ntens == 6


def test_stub3d_accepts_plane_strain():
    m = _Stub3D(dimension=PLANE_STRAIN)
    assert m.dimension is PLANE_STRAIN
    assert m.ntens == 4


# ---------------------------------------------------------------------------
# _hydrostatic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ss", [SOLID_3D, PLANE_STRAIN])
def test_hydrostatic_isotropic(ss):
    """Hydrostatic stress of isotropic state p*delta equals p."""
    m = _Stub3D(dimension=ss)
    p = 150.0
    stress = anp.array([p, p, p] + [0.0] * (ss.ntens - 3))
    np.testing.assert_allclose(float(m.hydrostatic(stress)), p)


@pytest.mark.parametrize("ss", [SOLID_3D, PLANE_STRAIN])
def test_hydrostatic_uniaxial(ss):
    """Uniaxial stress σ11=σ gives p = σ/3."""
    m = _Stub3D(dimension=ss)
    sigma = 300.0
    stress = (lambda _a: (_a.__setitem__(0, sigma), _a)[1])(np.zeros(ss.ntens))
    np.testing.assert_allclose(float(m.hydrostatic(stress)), sigma / 3.0)


def test_hydrostatic_shear_only():
    """Pure shear contributes nothing to hydrostatic pressure."""
    m = _Stub3D(dimension=SOLID_3D)
    stress = anp.array([0.0, 0.0, 0.0, 100.0, 50.0, 25.0])
    np.testing.assert_allclose(float(m.hydrostatic(stress)), 0.0)


# ---------------------------------------------------------------------------
# _dev
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ss", [SOLID_3D, PLANE_STRAIN])
def test_dev_trace_zero(ss):
    """Trace of deviatoric stress must be zero."""
    m = _Stub3D(dimension=ss)
    stress = anp.array([100.0, -50.0, 30.0] + [20.0] * (ss.ntens - 3))
    s = m.dev(stress)
    # Trace = sum of direct components
    np.testing.assert_allclose(float(anp.sum(s[:3])), 0.0, atol=1e-12)


@pytest.mark.parametrize("ss", [SOLID_3D, PLANE_STRAIN])
def test_dev_shape(ss):
    """_dev returns a vector of the same shape as the input."""
    m = _Stub3D(dimension=ss)
    stress = anp.ones(ss.ntens)
    assert m.dev(stress).shape == (ss.ntens,)


def test_dev_isotropic_stress_is_zero():
    """Deviatoric of an isotropic stress p*delta is zero."""
    m = _Stub3D(dimension=SOLID_3D)
    p = 200.0
    stress = anp.array([p, p, p, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(np.asarray(m.dev(stress)), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# _vonmises
# ---------------------------------------------------------------------------

def test_vonmises_uniaxial_solid_3d():
    """Uniaxial σ11: von Mises = σ11."""
    m = _Stub3D(dimension=SOLID_3D)
    sigma = 300.0
    stress = anp.array([sigma, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(float(m.vonmises(stress)), sigma, rtol=1e-6)


def test_vonmises_pure_shear_solid_3d():
    """Pure shear σ12: von Mises = √3 · σ12."""
    m = _Stub3D(dimension=SOLID_3D)
    tau = 100.0
    stress = anp.array([0.0, 0.0, 0.0, tau, 0.0, 0.0])
    expected = float(anp.sqrt(3.0) * tau)
    np.testing.assert_allclose(float(m.vonmises(stress)), expected, rtol=1e-6)


def test_vonmises_plane_strain_uniaxial():
    """Uniaxial loading in PLANE_STRAIN: von Mises = σ11."""
    m = _Stub3D(dimension=PLANE_STRAIN)
    sigma = 300.0
    stress = anp.array([sigma, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(float(m.vonmises(stress)), sigma, rtol=1e-6)


@pytest.mark.parametrize("ss", [SOLID_3D, PLANE_STRAIN])
def test_vonmises_nonnegative(ss):
    """Von Mises stress is always non-negative."""
    m = _Stub3D(dimension=ss)
    stress = anp.array([-200.0, 100.0, 50.0] + [-30.0] * (ss.ntens - 3))
    assert float(m.vonmises(stress)) >= 0.0


# ---------------------------------------------------------------------------
# isotropic_C
# ---------------------------------------------------------------------------

def test_isotropic_C_shape_solid_3d():
    m = _Stub3D(dimension=SOLID_3D)
    C = m.isotropic_C(lam=121153.8, mu=80769.2)
    assert C.shape == (6, 6)


def test_isotropic_C_shape_plane_strain():
    m = _Stub3D(dimension=PLANE_STRAIN)
    C = m.isotropic_C(lam=121153.8, mu=80769.2)
    assert C.shape == (4, 4)


@pytest.mark.parametrize("ss", [SOLID_3D, PLANE_STRAIN])
def test_isotropic_C_symmetry(ss):
    """Stiffness tensor must be symmetric."""
    m = _Stub3D(dimension=ss)
    C = m.isotropic_C(lam=121153.8, mu=80769.2)
    np.testing.assert_allclose(np.asarray(C), np.asarray(C.T), atol=1e-10)


def test_isotropic_C_solid_3d_diagonal():
    """C[0,0] = λ + 2μ and C[0,1] = λ for SOLID_3D."""
    E, nu = 210000.0, 0.3
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    m = _Stub3D(dimension=SOLID_3D)
    C = m.isotropic_C(lam, mu)
    np.testing.assert_allclose(float(C[0, 0]), lam + 2 * mu, rtol=1e-8)
    np.testing.assert_allclose(float(C[0, 1]), lam, rtol=1e-8)
    np.testing.assert_allclose(float(C[3, 3]), mu, rtol=1e-8)


def test_isotropic_C_plane_strain_is_submatrix():
    """PLANE_STRAIN C must equal the top-left 4×4 of SOLID_3D C."""
    E, nu = 210000.0, 0.3
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    C3d = _Stub3D(dimension=SOLID_3D).isotropic_C(lam, mu)
    Cpe = _Stub3D(dimension=PLANE_STRAIN).isotropic_C(lam, mu)
    np.testing.assert_allclose(np.asarray(Cpe), np.asarray(C3d[:4, :4]), atol=1e-10)


# ---------------------------------------------------------------------------
# _I_vol and _I_dev
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ss", [SOLID_3D, PLANE_STRAIN])
def test_I_vol_plus_I_dev_equals_identity(ss):
    """P_vol + P_dev must equal the ntens×ntens identity."""
    m = _Stub3D(dimension=ss)
    np.testing.assert_allclose(np.asarray(m.I_vol() + m.I_dev()), np.eye(ss.ntens), atol=1e-12)


@pytest.mark.parametrize("ss", [SOLID_3D, PLANE_STRAIN])
def test_I_vol_projects_to_hydrostatic(ss):
    """P_vol @ σ must equal p δ (volumetric part only)."""
    m = _Stub3D(dimension=ss)
    stress = anp.array([100.0, -50.0, 30.0] + [20.0] * (ss.ntens - 3))
    p = m.hydrostatic(stress)
    vol_part = m.I_vol() @ stress
    expected = p * ss.identity_np
    np.testing.assert_allclose(np.asarray(vol_part), np.asarray(expected), atol=1e-10)


@pytest.mark.parametrize("ss", [SOLID_3D, PLANE_STRAIN])
def test_I_dev_projects_to_deviatoric(ss):
    """P_dev @ σ must equal _dev(σ)."""
    m = _Stub3D(dimension=ss)
    stress = anp.array([100.0, -50.0, 30.0] + [20.0] * (ss.ntens - 3))
    np.testing.assert_allclose(np.asarray(m.I_dev() @ stress), np.asarray(m.dev(stress)), atol=1e-10)


# ===========================================================================
# MaterialModelPS
# ===========================================================================

# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_stubps_accepts_plane_stress():
    m = _StubPS()
    assert m.dimension is PLANE_STRESS
    assert m.ntens == 3


# ---------------------------------------------------------------------------
# _hydrostatic
# ---------------------------------------------------------------------------

def test_ps_hydrostatic_uniaxial():
    """σ11 = σ, σ22 = σ12 = 0 → p = σ/3."""
    m = _StubPS()
    sigma = 300.0
    stress = anp.array([sigma, 0.0, 0.0])
    np.testing.assert_allclose(float(m.hydrostatic(stress)), sigma / 3.0)


def test_ps_hydrostatic_equal_biaxial():
    """σ11 = σ22 = σ → p = 2σ/3 (σ33 = 0 contributes nothing)."""
    m = _StubPS()
    sigma = 200.0
    stress = anp.array([sigma, sigma, 0.0])
    np.testing.assert_allclose(float(m.hydrostatic(stress)), 2.0 * sigma / 3.0)


def test_ps_hydrostatic_shear_only():
    """Pure shear contributes nothing to hydrostatic pressure."""
    m = _StubPS()
    stress = anp.array([0.0, 0.0, 100.0])
    np.testing.assert_allclose(float(m.hydrostatic(stress)), 0.0)


# ---------------------------------------------------------------------------
# _dev
# ---------------------------------------------------------------------------

def test_ps_dev_trace_zero():
    """Trace of stored deviatoric components must be zero (σ11+σ22 part)."""
    m = _StubPS()
    stress = anp.array([100.0, -50.0, 20.0])
    s = m.dev(stress)
    # sum of direct stored components (indices 0,1) minus correction
    # σ11 + σ22 - 2p = σ11 + σ22 - 2(σ11+σ22)/3 = (σ11+σ22)/3
    # The full 3D trace s11+s22+s33 = 0; here we verify s11+s22 = -s33 = p
    p = m.hydrostatic(stress)
    np.testing.assert_allclose(float(s[0] + s[1]), float(p), atol=1e-12)


def test_ps_dev_shape():
    m = _StubPS()
    stress = anp.ones(3)
    assert m.dev(stress).shape == (3,)


# ---------------------------------------------------------------------------
# _vonmises
# ---------------------------------------------------------------------------

def test_ps_vonmises_uniaxial():
    """Uniaxial σ11: von Mises = σ11 (same as full 3D)."""
    m = _StubPS()
    sigma = 300.0
    stress = anp.array([sigma, 0.0, 0.0])
    np.testing.assert_allclose(float(m.vonmises(stress)), sigma, rtol=1e-6)


def test_ps_vonmises_equal_biaxial():
    """Equal biaxial σ11 = σ22 = σ: von Mises = σ (same as full 3D)."""
    m = _StubPS()
    sigma = 200.0
    stress = anp.array([sigma, sigma, 0.0])
    np.testing.assert_allclose(float(m.vonmises(stress)), sigma, rtol=1e-6)


def test_ps_vonmises_pure_shear():
    """Pure shear σ12: von Mises = √3 · σ12."""
    m = _StubPS()
    tau = 100.0
    stress = anp.array([0.0, 0.0, tau])
    np.testing.assert_allclose(float(m.vonmises(stress)), float(anp.sqrt(3.0) * tau), rtol=1e-6)


def test_ps_vonmises_matches_3d_reference():
    """PLANE_STRESS _vonmises must match the equivalent full 3D computation."""
    m_ps = _StubPS()
    m_3d = _Stub3D(dimension=SOLID_3D)
    # A stress that is valid in both: [σ11, σ22, σ12] plane-stress = [σ11, σ22, 0, σ12, 0, 0]
    sigma = anp.array([200.0, -100.0, 50.0])
    sigma_3d = anp.array([200.0, -100.0, 0.0, 50.0, 0.0, 0.0])
    np.testing.assert_allclose(float(m_ps.vonmises(sigma)), float(m_3d.vonmises(sigma_3d)), rtol=1e-6)


# ---------------------------------------------------------------------------
# isotropic_C
# ---------------------------------------------------------------------------

def test_ps_isotropic_C_shape():
    m = _StubPS()
    C = m.isotropic_C(lam=121153.8, mu=80769.2)
    assert C.shape == (3, 3)


def test_ps_isotropic_C_symmetry():
    m = _StubPS()
    C = m.isotropic_C(lam=121153.8, mu=80769.2)
    np.testing.assert_allclose(np.asarray(C), np.asarray(C.T), atol=1e-10)


def test_ps_isotropic_C_known_values():
    """C[0,0] = E/(1-nu²) and C[0,1] = E*nu/(1-nu²) for plane stress."""
    E, nu = 210000.0, 0.3
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    m = _StubPS()
    C = m.isotropic_C(lam, mu)
    expected_00 = E / (1 - nu ** 2)
    expected_01 = E * nu / (1 - nu ** 2)
    np.testing.assert_allclose(float(C[0, 0]), expected_00, rtol=1e-6)
    np.testing.assert_allclose(float(C[0, 1]), expected_01, rtol=1e-6)


# ---------------------------------------------------------------------------
# _I_vol and _I_dev
# ---------------------------------------------------------------------------

def test_ps_I_vol_plus_I_dev_equals_identity():
    m = _StubPS()
    np.testing.assert_allclose(np.asarray(m.I_vol() + m.I_dev()), np.eye(3), atol=1e-12)


def test_ps_I_dev_projects_to_deviatoric():
    """P_dev @ σ must equal _dev(σ) for plane stress."""
    m = _StubPS()
    stress = anp.array([100.0, -50.0, 20.0])
    np.testing.assert_allclose(np.asarray(m.I_dev() @ stress), np.asarray(m.dev(stress)), atol=1e-10)


# ===========================================================================
# MaterialModel1D
# ===========================================================================

# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_stub1d_accepts_uniaxial_1d():
    m = _Stub1D()
    assert m.dimension is UNIAXIAL_1D
    assert m.ntens == 1


# ---------------------------------------------------------------------------
# _hydrostatic
# ---------------------------------------------------------------------------

def test_1d_hydrostatic_uniaxial():
    """σ11 = σ → p = σ/3 (σ22 = σ33 = 0)."""
    m = _Stub1D()
    sigma = 300.0
    stress = anp.array([sigma])
    np.testing.assert_allclose(float(m.hydrostatic(stress)), sigma / 3.0)


def test_1d_hydrostatic_compressive():
    """Compressive uniaxial: p = σ/3 (negative)."""
    m = _Stub1D()
    sigma = -200.0
    stress = anp.array([sigma])
    np.testing.assert_allclose(float(m.hydrostatic(stress)), sigma / 3.0)


# ---------------------------------------------------------------------------
# _dev
# ---------------------------------------------------------------------------

def test_1d_dev_value():
    """s11 = σ11 − σ11/3 = 2σ11/3 for uniaxial stress."""
    m = _Stub1D()
    sigma = 300.0
    stress = anp.array([sigma])
    s = m.dev(stress)
    np.testing.assert_allclose(float(s[0]), 2.0 * sigma / 3.0, rtol=1e-8)


def test_1d_dev_shape():
    m = _Stub1D()
    assert m.dev(anp.array([100.0])).shape == (1,)


# ---------------------------------------------------------------------------
# _vonmises
# ---------------------------------------------------------------------------

def test_1d_vonmises_equals_abs_stress():
    """For uniaxial loading, σ_vm = |σ11| exactly."""
    m = _Stub1D()
    for sigma in [300.0, -200.0, 0.0]:
        stress = anp.array([sigma])
        np.testing.assert_allclose(float(m.vonmises(stress)), abs(sigma), atol=1e-8)


def test_1d_vonmises_matches_3d_reference():
    """UNIAXIAL_1D _vonmises must match the equivalent full 3D computation."""
    m_1d = _Stub1D()
    m_3d = _Stub3D(dimension=SOLID_3D)
    sigma = 350.0
    stress_1d = anp.array([sigma])
    stress_3d = anp.array([sigma, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(float(m_1d.vonmises(stress_1d)), float(m_3d.vonmises(stress_3d)), rtol=1e-6)


# ---------------------------------------------------------------------------
# isotropic_C
# ---------------------------------------------------------------------------

def test_1d_isotropic_C_shape():
    m = _Stub1D()
    C = m.isotropic_C(lam=121153.8, mu=80769.2)
    assert C.shape == (1, 1)


def test_1d_isotropic_C_equals_E():
    """C[0,0] must equal Young's modulus E."""
    E, nu = 210000.0, 0.3
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    m = _Stub1D()
    C = m.isotropic_C(lam, mu)
    np.testing.assert_allclose(float(C[0, 0]), E, rtol=1e-6)


# ---------------------------------------------------------------------------
# _I_vol and _I_dev
# ---------------------------------------------------------------------------

def test_1d_I_vol_plus_I_dev_equals_identity():
    m = _Stub1D()
    np.testing.assert_allclose(np.asarray(m.I_vol() + m.I_dev()), np.eye(1), atol=1e-12)


def test_1d_I_vol_value():
    """P_vol must equal [[1/3]] for 1D."""
    m = _Stub1D()
    np.testing.assert_allclose(np.asarray(m.I_vol()), np.array([[1.0 / 3.0]]), atol=1e-12)


def test_1d_I_dev_projects_to_deviatoric():
    """P_dev @ σ must equal _dev(σ) for 1D."""
    m = _Stub1D()
    stress = anp.array([300.0])
    np.testing.assert_allclose(np.asarray(m.I_dev() @ stress), np.asarray(m.dev(stress)), atol=1e-10)


# ---------------------------------------------------------------------------
# __init_subclass__ validation
# ---------------------------------------------------------------------------

def test_explicit_state_without_update_state_raises():
    """Model with an Explicit field but no update_state must raise TypeError."""
    from manforge.core.state import Explicit
    with pytest.raises(TypeError, match="update_state"):
        class Bad(MaterialModel):
            param_names = []
            ep = Explicit(shape=())

            def elastic_stiffness(self, state):
                pass

            def yield_function(self, state):
                pass


def test_implicit_state_without_state_residual_raises():
    """Model with an Implicit field but no state_residual must raise TypeError."""
    from manforge.core.state import Implicit
    with pytest.raises(TypeError, match="state_residual"):
        class Bad(MaterialModel):
            param_names = []
            alpha = Implicit(shape=NTENS)

            def elastic_stiffness(self, state):
                pass

            def yield_function(self, state):
                pass



def test_mixed_implicit_explicit_requires_both_methods():
    """Partial implicit (some explicit, some implicit) requires both methods."""
    from manforge.core.state import Implicit, Explicit
    class OK(MaterialModel):
        param_names = []
        alpha = Implicit(shape=NTENS)
        ep = Explicit(shape=())

        def elastic_stiffness(self, state):
            pass

        def yield_function(self, state):
            pass

        def update_state(self, new, dlambda, inc):
            return [self.ep(inc.state_n["ep"] + dlambda)]

        def state_residual(self, new, dlambda, inc):
            return [self.alpha(new["alpha"] - inc.state_n["alpha"])]

    assert OK.implicit_state_names == ["alpha"]


def test_all_implicit_states_no_update_state_needed():
    """Model with all states implicit does not need update_state."""
    from manforge.core.state import Implicit
    class OKImplicit(MaterialModel):
        param_names = []
        alpha = Implicit(shape=NTENS)
        ep = Implicit(shape=())

        def elastic_stiffness(self, state):
            pass

        def yield_function(self, state):
            pass

        def state_residual(self, new, dlambda, inc):
            return [self.alpha(anp.array(0.0)), self.ep(anp.array(0.0))]

    assert set(OKImplicit.implicit_state_names) == {"alpha", "ep"}


def test_no_state_fields_no_methods_needed():
    """Model with no StateField declarations needs neither update_state nor state_residual."""
    class OKStateless(MaterialModel):
        param_names = []

        def elastic_stiffness(self, state):
            pass

        def yield_function(self, state):
            pass

    # stress is always auto-attached even when no user fields are declared
    assert OKStateless.state_names == ["stress"]
    assert OKStateless.implicit_state_names == []



# ---------------------------------------------------------------------------
# params property
# ---------------------------------------------------------------------------

def test_params_returns_dict_keyed_by_param_names():
    model = _StubWithParams(a=1.0, b=2.0)
    assert model.params == {"a": 1.0, "b": 2.0}


def test_params_reflects_current_attribute_values():
    model = _StubWithParams(a=1.0, b=2.0)
    model.a = 99.0
    assert model.params["a"] == 99.0


def test_params_empty_when_param_names_empty():
    model = _Stub3D()
    assert model.params == {}


def test_j2_isotropic3d_params_contains_all_param_names(model):
    assert set(model.params.keys()) == set(model.param_names)
    assert model.params["E"] == 210000.0
    assert model.params["sigma_y0"] == 250.0


# ---------------------------------------------------------------------------
# Constructor: parameters are bound from param_names, all of them required
# ---------------------------------------------------------------------------

_DMG = dict(E=210000.0, nu=0.3, sigma_y0=250.0, H=1500.0, S=0.5, s=1.0)


def test_constructor_binds_every_param_name():
    m = _StubWithParams(a=1.0, b=2.0)
    assert (m.a, m.b) == (1.0, 2.0)


def test_missing_parameter_is_named():
    with pytest.raises(TypeError, match=r"_StubWithParams: missing parameters \['b'\]"):
        _StubWithParams(a=1.0)


def test_unexpected_parameter_is_named():
    with pytest.raises(TypeError, match=r"_StubWithParams: unexpected \['c'\]"):
        _StubWithParams(a=1.0, b=2.0, c=3.0)


def test_misspelt_parameter_reports_both_sides():
    with pytest.raises(
        TypeError, match=r"missing parameters \['b'\]; unexpected \['B'\]"
    ):
        _StubWithParams(a=1.0, B=2.0)


def test_parameters_are_keyword_only():
    with pytest.raises(TypeError):
        _StubWithParams(1.0, 2.0)


def test_reference_models_have_no_default_parameters():
    """p_D and h used to default; a default is a value PROPS never sees."""
    with pytest.raises(TypeError, match=r"missing parameters \['p_D'\]"):
        J2IsotropicDamage3D(**_DMG)
    with pytest.raises(TypeError, match=r"missing parameters \['h'\]"):
        J2IsotropicDamageCrackClosure3D(**_DMG, p_D=0.0)


def test_non_parameter_setting_goes_through_an_init_override():
    """The unilateral model's Heaviside value is a setting, not a parameter."""
    m = J2IsotropicDamageUnilateral3D(c=0.25, **_DMG, p_D=0.0, h=0.2)
    assert m.c == 0.25
    assert "c" not in m.params
    assert J2IsotropicDamageUnilateral3D(**_DMG, p_D=0.0, h=0.2).c == 0.5
    with pytest.raises(TypeError, match=r"missing parameters \['h'\]"):
        J2IsotropicDamageUnilateral3D(c=0.25, **_DMG, p_D=0.0)


@pytest.mark.parametrize("names, match", [
    (["E", "E"], "more than once"),
    ("E", "must be a list of str"),
    (["E", 1], "must be a list of str"),
    (["E", "sigma-y0"], "not valid attribute names"),
    (["E", "ep"], "collide"),            # a state field of the model
    (["E", "dimension"], "collide"),     # MaterialModel attributes
    (["E", "flow"], "collide"),
], ids=["duplicate", "bare-str", "non-str", "non-identifier",
        "state-field", "dimension", "method"])
def test_param_names_are_validated_at_class_creation(names, match):
    from manforge.core.state import Explicit, SCALAR

    with pytest.raises(TypeError, match=match):
        class _Bad(MaterialModel):
            param_names = names
            ep = Explicit(shape=SCALAR)

            def yield_function(self, state):
                return 0.0

            def update_state(self, new, dlambda, inc):
                return []


def test_param_names_must_be_declared():
    with pytest.raises(TypeError, match="param_names is not declared"):
        class _Bad(MaterialModel):
            def yield_function(self, state):
                return 0.0


# ---------------------------------------------------------------------------
# with_params: the same model with other parameter values
# ---------------------------------------------------------------------------

def test_with_params_replaces_only_the_named_parameters():
    m = _StubWithParams(a=1.0, b=2.0)
    n = m.with_params(b=5.0)
    assert n.params == {"a": 1.0, "b": 5.0}
    assert type(n) is _StubWithParams
    assert m.params == {"a": 1.0, "b": 2.0}      # the original is untouched


def test_with_params_keeps_the_dimension_of_an_open_class():
    m = _Stub3D(dimension=PLANE_STRAIN)
    assert m.with_params().dimension is PLANE_STRAIN


def test_with_params_keeps_non_parameter_settings():
    """What rebuilding through the constructor would lose."""
    m = J2IsotropicDamageUnilateral3D(c=0.25, **_DMG, p_D=0.0, h=0.2)
    n = m.with_params(S=0.8)
    assert n.c == 0.25
    assert n.S == 0.8
    assert n.E == m.E


def test_with_params_rejects_names_that_are_not_parameters():
    m = J2IsotropicDamageUnilateral3D(**_DMG, p_D=0.0, h=0.2)
    with pytest.raises(TypeError, match=r"with_params: unexpected \['c'\]"):
        m.with_params(c=0.25)          # a setting, not a parameter
    with pytest.raises(TypeError, match=r"unexpected \['sigma_y'\]"):
        m.with_params(sigma_y=300.0)   # misspelt


def test_with_params_stiffness_follows_the_new_parameters(model):
    C0 = model.isotropic_stiffness()            # fills the original's cache
    stiffer = model.with_params(E=2.0 * model.E)
    np.testing.assert_allclose(stiffer.isotropic_stiffness(), 2.0 * C0)
    np.testing.assert_allclose(model.isotropic_stiffness(), C0)


# ---------------------------------------------------------------------------
# Constructor: dimension — open unless a class body fixes it
# ---------------------------------------------------------------------------

def test_open_class_defaults_to_3d():
    assert _Stub3D().dimension is SOLID_3D
    assert _Stub3D._fixed_dimension is None


def test_class_body_dimension_fixes_the_class():
    assert _StubPS().dimension is PLANE_STRESS
    # Its own dimension is accepted, so a model can always be rebuilt with
    # type(m)(dimension=m.dimension, **m.params).
    assert _StubPS(dimension=PLANE_STRESS).dimension is PLANE_STRESS
    with pytest.raises(
        TypeError, match="_StubPS is fixed to PLANE_STRESS; got dimension=3D_SOLID"
    ):
        _StubPS(dimension=SOLID_3D)


def test_fixed_dimension_is_inherited():
    class _Child(_StubPS):
        pass

    assert _Child().dimension is PLANE_STRESS
    with pytest.raises(TypeError, match="_Child is fixed to PLANE_STRESS"):
        _Child(dimension=UNIAXIAL_1D)


def test_subclass_fixes_an_open_family():
    class _PE(_Stub3D):
        dimension = PLANE_STRAIN

    assert _PE().ntens == 4
    with pytest.raises(TypeError, match="_PE is fixed to PLANE_STRAIN"):
        _PE(dimension=SOLID_3D)
    # The parent stays open.
    assert _Stub3D(dimension=UNIAXIAL_1D).ntens == 1


def test_dimension_must_be_a_stress_dimension():
    with pytest.raises(TypeError, match="dimension must be a StressDimension"):
        _Stub3D(dimension="3d")
    with pytest.raises(TypeError, match="dimension must be a StressDimension"):
        class _Bad(_Stub3D):
            dimension = "3d"


def test_dimension_is_keyword_only():
    with pytest.raises(TypeError):
        _Stub3D(PLANE_STRAIN)


def test_reference_models_fix_the_dimension_their_name_says():
    """Family parents are open; every ``*3D`` / ``*PS`` / ``*1D`` class is fixed."""
    import manforge.models as models

    for name in models.__all__:
        cls = getattr(models, name)
        suffix = next((s for s in ("3D", "PS", "1D") if name.endswith(s)), None)
        if suffix is None:
            assert cls._fixed_dimension is None, name
        else:
            assert cls._fixed_dimension is not None, name
            assert cls._fixed_dimension.ntens == {"3D": 6, "PS": 3, "1D": 1}[suffix], name


def test_closed_form_model_rejects_a_dimension_it_does_not_hold_in():
    with pytest.raises(TypeError, match="J2Isotropic3D is fixed to 3D_SOLID"):
        J2Isotropic3D(dimension=PLANE_STRESS, E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)


# ---------------------------------------------------------------------------
# inner_product (MaterialModel method)
# ---------------------------------------------------------------------------

def _model3d():
    return J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)

def _model_ps():
    return J2IsotropicPS(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)

def _model1d():
    return J2Isotropic1D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)


def test_inner_product_uniaxial_3d():
    m = _model3d()
    a = anp.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    b = anp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(float(m.inner_product(a, b)), 100.0, atol=1e-12)


def test_inner_product_pure_shear_doubled():
    # Physical shear σ12=1: Mandel factor √2 → inner_product = (√2)² = 2
    m = _model3d()
    v = anp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(float(m.inner_product(v, v)), 2.0, atol=1e-12)


def test_inner_product_agrees_with_tensor_contraction():
    """inner_product equals A:B on the full symmetric 3×3 tensors."""
    def tensor(v):  # physical-shear Voigt [11, 22, 33, 12, 13, 23]
        return np.array([[v[0], v[3], v[4]],
                         [v[3], v[1], v[5]],
                         [v[4], v[5], v[2]]])

    m = _model3d()
    a = anp.array([100.0, 50.0, -30.0, 10.0, 5.0, 2.0])
    b = anp.array([20.0, -10.0, 15.0, 3.0, 1.0, 4.0])
    expected = float(np.sum(tensor(a) * tensor(b)))
    np.testing.assert_allclose(float(m.inner_product(a, b)), expected, atol=1e-12)


def test_inner_product_ps():
    m = _model_ps()
    a = anp.array([100.0, 50.0, 10.0])
    b = anp.array([20.0, -10.0, 3.0])
    # Mandel factors for PS: [1, 1, √2]
    sqrt2 = math.sqrt(2.0)
    expected = 100.0*20.0 + 50.0*(-10.0) + (sqrt2*10.0)*(sqrt2*3.0)
    np.testing.assert_allclose(float(m.inner_product(a, b)), expected, atol=1e-12)


def test_inner_product_1d():
    m = _model1d()
    a = anp.array([100.0])
    b = anp.array([3.0])
    np.testing.assert_allclose(float(m.inner_product(a, b)), 300.0, atol=1e-12)


# ---------------------------------------------------------------------------
# deviatoric_inner_product
# ---------------------------------------------------------------------------

def test_deviatoric_inner_product_3d_same_as_inner():
    m = _model3d()
    stress = anp.array([300.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s = anp.array(m.dev(stress))
    np.testing.assert_allclose(
        float(m.deviatoric_inner_product(s, s)),
        float(m.inner_product(s, s)),
        atol=1e-12,
    )


def test_deviatoric_inner_product_ps_explicit():
    m = _model_ps()
    # Deviatoric stress in PS: s = [s11, s22, s12], s33 = -(s11+s22)
    s11, s22, s12 = 80.0, -30.0, 15.0
    s = anp.array([s11, s22, s12])
    s33 = -(s11 + s22)
    expected = s11*s11 + s22*s22 + s33*s33 + 2.0*s12*s12
    np.testing.assert_allclose(
        float(m.deviatoric_inner_product(s, s)), expected, atol=1e-12
    )


def test_deviatoric_inner_product_1d():
    m = _model1d()
    sigma = 300.0
    # dev([σ11]) = [2/3 σ11], s22 = s33 = -s11/2
    s = anp.array(m.dev(anp.array([sigma])))
    s11_dev = float(s[0])
    expected = s11_dev**2 + 2.0 * (s11_dev / 2.0)**2
    np.testing.assert_allclose(
        float(m.deviatoric_inner_product(s, s)), expected, atol=1e-12
    )


def test_deviatoric_inner_product_consistent_with_vonmises_norm():
    """√(3/2 dev_inner(s, s)) == vonmises_norm(s) for all stress states."""
    for m, stress in [
        (_model3d(), anp.array([300.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        (_model_ps(), anp.array([300.0, 0.0, 0.0])),
        (_model1d(), anp.array([300.0])),
    ]:
        s = anp.array(m.dev(stress))
        via_dev_inner = float(anp.sqrt(1.5 * m.deviatoric_inner_product(s, s)))
        via_vonmises_norm = float(m.vonmises_norm(s))
        np.testing.assert_allclose(via_dev_inner, via_vonmises_norm, rtol=1e-10)


# ---------------------------------------------------------------------------
# strain_norm
# ---------------------------------------------------------------------------

def test_strain_norm_uniaxial_3d():
    # Isochoric uniaxial plastic strain: ε=[ε11, -ε11/2, -ε11/2, 0, 0, 0] → ε11
    m = _model3d()
    e11 = 0.01
    eps = anp.array([e11, -e11/2, -e11/2, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(float(m.strain_norm(eps)), e11, rtol=1e-10)


def test_strain_norm_uniaxial_ps():
    m = _model_ps()
    e11 = 0.01
    eps = anp.array([e11, -e11/2, 0.0])
    np.testing.assert_allclose(float(m.strain_norm(eps)), e11, rtol=1e-10)


def test_strain_norm_uniaxial_1d():
    m = _model1d()
    e11 = 0.01
    eps = anp.array([e11])
    np.testing.assert_allclose(float(m.strain_norm(eps)), e11, rtol=1e-10)


def test_strain_norm_pure_shear_3d():
    # Engineering shear γ12 → tensor ε12 = γ/2
    # ε_eq = √(2/3 × 2ε12²×2) = √(2/3 × γ²/2) = γ/√3
    m = _model3d()
    g = 0.005  # engineering shear γ12 (driver convention)
    eps_eng = anp.array([0.0, 0.0, 0.0, g, 0.0, 0.0])
    expected = g / math.sqrt(3.0)
    np.testing.assert_allclose(float(m.strain_norm(eps_eng)), expected, rtol=1e-10)


def test_strain_norm_conjugate_to_vonmises():
    """For purely deviatoric isochoric strain, strain_norm(ε) * vonmises(σ) equals
    the plastic dissipation rate when σ is uniaxial at yield."""
    m = _model3d()
    sigma_y = 250.0
    stress = anp.array([sigma_y, 0.0, 0.0, 0.0, 0.0, 0.0])
    e11 = 0.01
    eps_p = anp.array([e11, -e11/2, -e11/2, 0.0, 0.0, 0.0])
    # σ:ε = σ11 * ε11 = σ_y * e11 (uniaxial)
    # vonmises(σ) * strain_norm(ε) should equal same
    plastic_dissipation = sigma_y * e11
    vm_times_enorm = float(m.vonmises(stress)) * float(m.strain_norm(eps_p))
    np.testing.assert_allclose(vm_times_enorm, plastic_dissipation, rtol=1e-10)


# ---------------------------------------------------------------------------
# Hook signatures are checked when the class is created
# ---------------------------------------------------------------------------

class _GuardBase(MaterialModel):
    param_names = ["E", "nu"]

    def __init__(self):
        super().__init__(E=210000.0, nu=0.3)

    def yield_function(self, state):
        return self.vonmises(state["stress"]) - 250.0


@pytest.mark.parametrize("hook, old_params, expected", [
    ("update_state", "self, dlambda, state_new, state_n, *, stress_trial=None, strain_inc=None",
     "new, dlambda, inc"),
    ("state_residual", "self, state_new, dlambda, state_n, *, stress_trial, strain_inc=None",
     "new, dlambda, inc"),
    ("elastic_check", "self, state_n, *, stress_trial, strain_inc=None", "inc"),
    ("elastic_update_state", "self, state_n, *, stress_trial, strain_inc=None", "inc"),
    ("user_defined_return_mapping", "self, stress_trial, C, state_n", "inc"),
    ("user_defined_tangent", "self, stress, state, dlambda, C, state_n", "new, dlambda, inc"),
])
def test_hook_with_another_argument_list_is_rejected(hook, old_params, expected):
    """Called positionally, a stale hook would get its arguments in the wrong slots."""
    namespace = {}
    exec(f"def {hook}({old_params}):\n    return None\n", namespace)
    with pytest.raises(TypeError, match=rf"{hook} must take \({expected}\)"):
        type("_Stale", (_GuardBase,), {hook: namespace[hook]})


def test_hook_check_covers_abstract_intermediates():
    """A mixin without yield_function is checked too."""
    def update_state(self, dlambda, state_new, state_n):
        return []

    with pytest.raises(TypeError, match="update_state must take"):
        type("_Mixin", (MaterialModel,), {"update_state": update_state})


def test_current_signatures_are_accepted():
    class _Fine(_GuardBase):
        def elastic_check(self, inc):
            return True

        def user_defined_tangent(self, new, dlambda, inc):
            return None

    assert _Fine().ntens == 6


# ---------------------------------------------------------------------------
# isotropic_stiffness / elastic_stiffness
# ---------------------------------------------------------------------------

def test_isotropic_stiffness_is_built_from_E_and_nu(model):
    E, nu = model.E, model.nu
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    np.testing.assert_array_equal(
        np.asarray(model.isotropic_stiffness()), np.asarray(model.isotropic_C(lam, mu)))


def test_default_elastic_stiffness_is_the_isotropic_stiffness(model):
    state = model.initial_state()
    assert model.elastic_stiffness(state) is model.isotropic_stiffness()


def test_elastic_stiffness_requires_a_state(model):
    """No state-less call: there is no default that would stand for 'undamaged'."""
    with pytest.raises(TypeError):
        model.elastic_stiffness()


def test_isotropic_stiffness_follows_reassigned_parameters(model):
    C_before = np.asarray(model.isotropic_stiffness()).copy()
    model.E = 2.0 * model.E
    np.testing.assert_allclose(np.asarray(model.isotropic_stiffness()), 2.0 * C_before)


def test_isotropic_stiffness_needs_E_and_nu():
    m = _Stub3D()
    with pytest.raises(NotImplementedError, match="E and"):
        m.isotropic_stiffness()


def test_default_elastic_tangent_hook_is_none(model):
    from manforge.verification import make_increment
    inc = make_increment(model, model.initial_state(), stress_trial=np.zeros(6))
    assert model.elastic_tangent(inc) is None
