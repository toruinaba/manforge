"""Tests for MaterialModel operator methods across stress states.

Covers:
- Constructor dimension acceptance
- _hydrostatic, _dev, _vonmises operators — values and shapes
- isotropic_C — shape, symmetry, known values
- _I_vol, _I_dev — projection identity: P_vol + P_dev == I
"""

import autograd.numpy as anp
import numpy as np
import pytest

from manforge.core.material import MaterialModel
from manforge.core.state import NTENS
from manforge.core.dimension import SOLID_3D, PLANE_STRAIN, PLANE_STRESS, UNIAXIAL_1D
from tests.fixtures.stubs import _Stub3D, _StubPS, _Stub1D, _StubWithParams


# ---------------------------------------------------------------------------
# Constructor dimension
# ---------------------------------------------------------------------------

def test_stub3d_accepts_solid_3d():
    m = _Stub3D(SOLID_3D)
    assert m.dimension is SOLID_3D
    assert m.ntens == 6


def test_stub3d_accepts_plane_strain():
    m = _Stub3D(PLANE_STRAIN)
    assert m.dimension is PLANE_STRAIN
    assert m.ntens == 4


# ---------------------------------------------------------------------------
# _hydrostatic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ss", [SOLID_3D, PLANE_STRAIN])
def test_hydrostatic_isotropic(ss):
    """Hydrostatic stress of isotropic state p*delta equals p."""
    m = _Stub3D(ss)
    p = 150.0
    stress = anp.array([p, p, p] + [0.0] * (ss.ntens - 3))
    np.testing.assert_allclose(float(m.hydrostatic(stress)), p)


@pytest.mark.parametrize("ss", [SOLID_3D, PLANE_STRAIN])
def test_hydrostatic_uniaxial(ss):
    """Uniaxial stress σ11=σ gives p = σ/3."""
    m = _Stub3D(ss)
    sigma = 300.0
    stress = (lambda _a: (_a.__setitem__(0, sigma), _a)[1])(np.zeros(ss.ntens))
    np.testing.assert_allclose(float(m.hydrostatic(stress)), sigma / 3.0)


def test_hydrostatic_shear_only():
    """Pure shear contributes nothing to hydrostatic pressure."""
    m = _Stub3D(SOLID_3D)
    stress = anp.array([0.0, 0.0, 0.0, 100.0, 50.0, 25.0])
    np.testing.assert_allclose(float(m.hydrostatic(stress)), 0.0)


# ---------------------------------------------------------------------------
# _dev
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ss", [SOLID_3D, PLANE_STRAIN])
def test_dev_trace_zero(ss):
    """Trace of deviatoric stress must be zero."""
    m = _Stub3D(ss)
    stress = anp.array([100.0, -50.0, 30.0] + [20.0] * (ss.ntens - 3))
    s = m.dev(stress)
    # Trace = sum of direct components
    np.testing.assert_allclose(float(anp.sum(s[:3])), 0.0, atol=1e-12)


@pytest.mark.parametrize("ss", [SOLID_3D, PLANE_STRAIN])
def test_dev_shape(ss):
    """_dev returns a vector of the same shape as the input."""
    m = _Stub3D(ss)
    stress = anp.ones(ss.ntens)
    assert m.dev(stress).shape == (ss.ntens,)


def test_dev_isotropic_stress_is_zero():
    """Deviatoric of an isotropic stress p*delta is zero."""
    m = _Stub3D(SOLID_3D)
    p = 200.0
    stress = anp.array([p, p, p, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(np.asarray(m.dev(stress)), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# _vonmises
# ---------------------------------------------------------------------------

def test_vonmises_uniaxial_solid_3d():
    """Uniaxial σ11: von Mises = σ11."""
    m = _Stub3D(SOLID_3D)
    sigma = 300.0
    stress = anp.array([sigma, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(float(m.vonmises(stress)), sigma, rtol=1e-6)


def test_vonmises_pure_shear_solid_3d():
    """Pure shear σ12: von Mises = √3 · σ12."""
    m = _Stub3D(SOLID_3D)
    tau = 100.0
    stress = anp.array([0.0, 0.0, 0.0, tau, 0.0, 0.0])
    expected = float(anp.sqrt(3.0) * tau)
    np.testing.assert_allclose(float(m.vonmises(stress)), expected, rtol=1e-6)


def test_vonmises_plane_strain_uniaxial():
    """Uniaxial loading in PLANE_STRAIN: von Mises = σ11."""
    m = _Stub3D(PLANE_STRAIN)
    sigma = 300.0
    stress = anp.array([sigma, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(float(m.vonmises(stress)), sigma, rtol=1e-6)


@pytest.mark.parametrize("ss", [SOLID_3D, PLANE_STRAIN])
def test_vonmises_nonnegative(ss):
    """Von Mises stress is always non-negative."""
    m = _Stub3D(ss)
    stress = anp.array([-200.0, 100.0, 50.0] + [-30.0] * (ss.ntens - 3))
    assert float(m.vonmises(stress)) >= 0.0


# ---------------------------------------------------------------------------
# isotropic_C
# ---------------------------------------------------------------------------

def test_isotropic_C_shape_solid_3d():
    m = _Stub3D(SOLID_3D)
    C = m.isotropic_C(lam=121153.8, mu=80769.2)
    assert C.shape == (6, 6)


def test_isotropic_C_shape_plane_strain():
    m = _Stub3D(PLANE_STRAIN)
    C = m.isotropic_C(lam=121153.8, mu=80769.2)
    assert C.shape == (4, 4)


@pytest.mark.parametrize("ss", [SOLID_3D, PLANE_STRAIN])
def test_isotropic_C_symmetry(ss):
    """Stiffness tensor must be symmetric."""
    m = _Stub3D(ss)
    C = m.isotropic_C(lam=121153.8, mu=80769.2)
    np.testing.assert_allclose(np.asarray(C), np.asarray(C.T), atol=1e-10)


def test_isotropic_C_solid_3d_diagonal():
    """C[0,0] = λ + 2μ and C[0,1] = λ for SOLID_3D."""
    E, nu = 210000.0, 0.3
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    m = _Stub3D(SOLID_3D)
    C = m.isotropic_C(lam, mu)
    np.testing.assert_allclose(float(C[0, 0]), lam + 2 * mu, rtol=1e-8)
    np.testing.assert_allclose(float(C[0, 1]), lam, rtol=1e-8)
    np.testing.assert_allclose(float(C[3, 3]), mu, rtol=1e-8)


def test_isotropic_C_plane_strain_is_submatrix():
    """PLANE_STRAIN C must equal the top-left 4×4 of SOLID_3D C."""
    E, nu = 210000.0, 0.3
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    C3d = _Stub3D(SOLID_3D).isotropic_C(lam, mu)
    Cpe = _Stub3D(PLANE_STRAIN).isotropic_C(lam, mu)
    np.testing.assert_allclose(np.asarray(Cpe), np.asarray(C3d[:4, :4]), atol=1e-10)


# ---------------------------------------------------------------------------
# _I_vol and _I_dev
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ss", [SOLID_3D, PLANE_STRAIN])
def test_I_vol_plus_I_dev_equals_identity(ss):
    """P_vol + P_dev must equal the ntens×ntens identity."""
    m = _Stub3D(ss)
    np.testing.assert_allclose(np.asarray(m.I_vol() + m.I_dev()), np.eye(ss.ntens), atol=1e-12)


@pytest.mark.parametrize("ss", [SOLID_3D, PLANE_STRAIN])
def test_I_vol_projects_to_hydrostatic(ss):
    """P_vol @ σ must equal p δ (volumetric part only)."""
    m = _Stub3D(ss)
    stress = anp.array([100.0, -50.0, 30.0] + [20.0] * (ss.ntens - 3))
    p = m.hydrostatic(stress)
    vol_part = m.I_vol() @ stress
    expected = p * ss.identity_jnp
    np.testing.assert_allclose(np.asarray(vol_part), np.asarray(expected), atol=1e-10)


@pytest.mark.parametrize("ss", [SOLID_3D, PLANE_STRAIN])
def test_I_dev_projects_to_deviatoric(ss):
    """P_dev @ σ must equal _dev(σ)."""
    m = _Stub3D(ss)
    stress = anp.array([100.0, -50.0, 30.0] + [20.0] * (ss.ntens - 3))
    np.testing.assert_allclose(np.asarray(m.I_dev() @ stress), np.asarray(m.dev(stress)), atol=1e-10)


# ===========================================================================
# MaterialModelPS
# ===========================================================================

# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_stubps_accepts_plane_stress():
    m = _StubPS(PLANE_STRESS)
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
    m_3d = _Stub3D(SOLID_3D)
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
    m = _Stub1D(UNIAXIAL_1D)
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
    m_3d = _Stub3D(SOLID_3D)
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

            def elastic_stiffness(self, state=None):
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

            def elastic_stiffness(self, state=None):
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

        def elastic_stiffness(self, state=None):
            pass

        def yield_function(self, state):
            pass

        def update_state(self, dlambda, state_new, state_n, *, stress_trial=None, strain_inc=None):
            return [self.ep(state_n["ep"] + dlambda)]

        def state_residual(self, state_new, dlambda, state_n, *, stress_trial, strain_inc=None):
            return [self.alpha(state_new["alpha"] - state_n["alpha"])]

    assert OK.implicit_state_names == ["alpha"]


def test_all_implicit_states_no_update_state_needed():
    """Model with all states implicit does not need update_state."""
    from manforge.core.state import Implicit
    class OKImplicit(MaterialModel):
        param_names = []
        alpha = Implicit(shape=NTENS)
        ep = Implicit(shape=())

        def elastic_stiffness(self, state=None):
            pass

        def yield_function(self, state):
            pass

        def state_residual(self, state_new, dlambda, state_n, *, stress_trial, strain_inc=None):
            return [self.alpha(anp.array(0.0)), self.ep(anp.array(0.0))]

    assert set(OKImplicit.implicit_state_names) == {"alpha", "ep"}


def test_no_state_fields_no_methods_needed():
    """Model with no StateField declarations needs neither update_state nor state_residual."""
    class OKStateless(MaterialModel):
        param_names = []

        def elastic_stiffness(self, state=None):
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
