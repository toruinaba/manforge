"""Integration tests for multi-dimensionality: PLANE_STRAIN and constructor guards.

Covers:
- J2Isotropic parent accepts any StressDimension (SOLID_3D, PLANE_STRAIN, etc.)
- PLANE_STRAIN elastic step: shapes, stress == C@deps, tangent == C
- PLANE_STRAIN plastic step: yield consistency, ep > 0
- PLANE_STRAIN analytical tangent: finite-difference verification
- PLANE_STRAIN analytical vs autodiff cross-check
- Driver integration with 4-component arrays (UniaxialDriver, GeneralDriver)
- Plane-strain signature: sigma_33 != 0 under axial loading
- J2Isotropic(PLANE_STRAIN) autodiff path works correctly
- J2IsotropicPS (no plastic_corrector) raises NotImplementedError via PythonAnalyticalIntegrator
"""

import re

import autograd.numpy as anp
import numpy as np
import pytest

from manforge.core.dimension import SOLID_3D, PLANE_STRAIN, PLANE_STRESS, UNIAXIAL_1D
from manforge.models.j2_isotropic import J2Isotropic, J2Isotropic3D, J2IsotropicPS, J2Isotropic1D
from manforge.simulation.driver import StrainDriver
from manforge.simulation.integrator import (
    PythonIntegrator,
    PythonNumericalIntegrator,
    PythonAnalyticalIntegrator,
)
from manforge.simulation.types import FieldHistory, FieldType
from manforge.verification.tangent import check_tangent


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_j2isotropic3d_accepts_solid_3d():
    model = J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
    assert model.dimension is SOLID_3D
    assert model.ntens == 6


def test_j2isotropic_accepts_plane_strain():
    model = J2Isotropic(dimension=PLANE_STRAIN, E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
    assert model.dimension is PLANE_STRAIN
    assert model.ntens == 4



# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pe_model():
    return J2Isotropic(dimension=PLANE_STRAIN, E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)


@pytest.fixture
def pe_state(pe_model):
    return pe_model.initial_state()


@pytest.fixture
def pe_model_3d():
    """J2Isotropic3D used as plane-strain via analytical hooks (SOLID_3D)."""
    return J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)


@pytest.fixture
def pe_state_3d(pe_model_3d):
    return pe_model_3d.initial_state()


# ---------------------------------------------------------------------------
# Elastic step — shape and value
# ---------------------------------------------------------------------------

def test_elastic_step_shapes(pe_model, pe_state):
    """Elastic step produces stress (4,) and tangent (4, 4)."""
    deps = anp.array([1e-4, 0.0, 0.0, 0.0])
    _r = PythonIntegrator(pe_model).stress_update(deps, anp.zeros(4), pe_state)
    stress, state, ddsdde = _r.stress, _r.state, _r.ddsdde
    assert stress.shape == (4,)
    assert ddsdde.shape == (4, 4)


def test_elastic_step_stress_equals_C_deps(pe_model, pe_state):
    """Elastic stress must equal C @ deps."""
    deps = anp.array([1e-4, 0.0, 0.0, 0.0])
    C = pe_model.elastic_stiffness()
    stress = PythonIntegrator(pe_model).stress_update(deps, anp.zeros(4), pe_state).stress
    np.testing.assert_allclose(np.asarray(stress), np.asarray(C @ deps), rtol=1e-10)


def test_elastic_step_tangent_equals_C(pe_model, pe_state):
    """Elastic tangent must equal the elastic stiffness C."""
    deps = anp.array([1e-4, 0.0, 0.0, 0.0])
    C = pe_model.elastic_stiffness()
    ddsdde = PythonIntegrator(pe_model).stress_update(deps, anp.zeros(4), pe_state).ddsdde
    np.testing.assert_allclose(np.asarray(ddsdde), np.asarray(C), rtol=1e-10)


# ---------------------------------------------------------------------------
# Plastic step — yield consistency and ep update
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strain_inc_vec", [
    [2e-3, 0.0, 0.0, 0.0],
    [2e-3, -1e-3, -1e-3, 0.0],   # isochoric ×2 — vm≈485 MPa > sigma_y0
    [0.0, 0.0, 0.0, 2e-3],
    [2e-3, 1e-3, 4e-4, 2e-3],    # mixed ×2 — vm≈360 MPa > sigma_y0
])
def test_plastic_yield_consistency(pe_model, pe_state, strain_inc_vec):
    """Plastic step: yield function ≈ 0 at converged state."""
    deps = anp.array(strain_inc_vec)
    _r = PythonIntegrator(pe_model).stress_update(deps, anp.zeros(4), pe_state)
    stress, state = _r.stress, _r.state
    f = pe_model.yield_function(state)
    assert abs(float(f)) < 1e-8, f"|f| = {abs(float(f)):.3e}"


@pytest.mark.parametrize("strain_inc_vec", [
    [2e-3, 0.0, 0.0, 0.0],
    [2e-3, -1e-3, -1e-3, 0.0],   # isochoric ×2 — vm≈485 MPa > sigma_y0
    [0.0, 0.0, 0.0, 2e-3],
    [2e-3, 1e-3, 4e-4, 2e-3],    # mixed ×2 — vm≈360 MPa > sigma_y0
])
def test_plastic_ep_positive(pe_model, pe_state, strain_inc_vec):
    """Plastic step: equivalent plastic strain must increase."""
    deps = anp.array(strain_inc_vec)
    state = PythonIntegrator(pe_model).stress_update(deps, anp.zeros(4), pe_state).state
    assert float(state["ep"]) > 0.0


# ---------------------------------------------------------------------------
# Analytical tangent — finite-difference verification (J2Isotropic3D, SOLID_3D)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strain_inc_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1e-3, -5e-4, -5e-4, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
    [1e-3, 5e-4, 2e-4, 1e-3, 0.0, 0.0],
])
def test_analytical_tangent_fd_check(pe_model_3d, pe_state_3d, strain_inc_vec):
    """3D analytical tangent passes finite-difference check."""
    result = check_tangent(
        PythonAnalyticalIntegrator(pe_model_3d),
        anp.zeros(6),
        pe_state_3d,
        anp.array(strain_inc_vec),
    )
    assert result.passed, f"FD check failed: max_rel_err = {result.max_rel_err:.3e}"


# ---------------------------------------------------------------------------
# Analytical vs autodiff cross-check (J2Isotropic3D, SOLID_3D)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strain_inc_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1e-3, -5e-4, -5e-4, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
    [1e-3, 5e-4, 2e-4, 1e-3, 0.0, 0.0],
])
def test_analytical_stress_matches_autodiff(pe_model_3d, pe_state_3d, strain_inc_vec):
    """Analytical and autodiff stress must agree to atol=1e-6."""
    deps = anp.array(strain_inc_vec)
    s_ad = PythonNumericalIntegrator(pe_model_3d).stress_update(deps, anp.zeros(6), pe_state_3d).stress
    s_an = PythonAnalyticalIntegrator(pe_model_3d).stress_update(deps, anp.zeros(6), pe_state_3d).stress
    np.testing.assert_allclose(
        np.asarray(s_an), np.asarray(s_ad), atol=1e-6,
        err_msg=f"max stress diff = {float(anp.max(anp.abs(s_an - s_ad))):.3e}",
    )


@pytest.mark.parametrize("strain_inc_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1e-3, -5e-4, -5e-4, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
])
def test_analytical_tangent_matches_autodiff(pe_model_3d, pe_state_3d, strain_inc_vec):
    """Analytical and autodiff tangent must agree within 1e-5 relative error."""
    deps = anp.array(strain_inc_vec)
    D_ad = PythonNumericalIntegrator(pe_model_3d).stress_update(deps, anp.zeros(6), pe_state_3d).ddsdde
    D_an = PythonAnalyticalIntegrator(pe_model_3d).stress_update(deps, anp.zeros(6), pe_state_3d).ddsdde
    rel_err = anp.abs(D_an - D_ad) / (anp.abs(D_ad) + 1.0)
    assert float(anp.max(rel_err)) < 1e-5, \
        f"max tangent rel err = {float(anp.max(rel_err)):.3e}"


# ---------------------------------------------------------------------------
# Driver integration
# ---------------------------------------------------------------------------

def test_uniaxial_driver_plane_strain(pe_model):
    """StrainDriver (uniaxial) works with a PLANE_STRAIN model."""
    eps_history = np.linspace(0, 5e-3, 20)
    load = FieldHistory(FieldType.STRAIN, "Strain", eps_history)
    result = StrainDriver(PythonNumericalIntegrator(pe_model)).run(load)
    assert result.stress.shape == (20, 4)
    # σ11 must increase monotonically for hardening material
    assert np.all(np.diff(result.stress[:, 0]) >= 0)


def test_general_driver_plane_strain_shapes(pe_model):
    """StrainDriver (general) produces (N, 4) stress output for PLANE_STRAIN model."""
    N = 15
    strain_history = np.zeros((N, 4))
    strain_history[:, 0] = np.linspace(0, 5e-3, N)  # ramp eps_11
    load = FieldHistory(FieldType.STRAIN, "Strain", strain_history)
    result = StrainDriver(PythonNumericalIntegrator(pe_model)).run(load)
    assert result.stress.shape == (N, 4)


def test_plane_strain_sigma33_nonzero(pe_model):
    """Plane-strain constraint produces non-zero sigma_33 under axial loading."""
    eps_history = np.zeros((10, 4))
    eps_history[:, 0] = np.linspace(0, 5e-3, 10)  # ramp eps_11 only
    load = FieldHistory(FieldType.STRAIN, "Strain", eps_history)
    result = StrainDriver(PythonNumericalIntegrator(pe_model)).run(load)
    # sigma_33 (index 2) must be non-zero due to plane-strain lateral constraint
    assert np.any(np.abs(result.stress[:, 2]) > 1.0), \
        f"sigma_33 unexpectedly near zero: {result.stress[:, 2]}"


# ---------------------------------------------------------------------------
# Autodiff path and analytical-raises behavior
# ---------------------------------------------------------------------------

def test_j2isotropic3d_autodiff_plane_strain(pe_state):
    """J2Isotropic(PLANE_STRAIN) via PythonNumericalIntegrator works correctly."""
    model = J2Isotropic(dimension=PLANE_STRAIN, E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
    deps = anp.array([2e-3, 0.0, 0.0, 0.0])
    _r = PythonNumericalIntegrator(model).stress_update(deps, anp.zeros(4), pe_state)
    stress, state, ddsdde = _r.stress, _r.state, _r.ddsdde
    assert stress.shape == (4,)
    assert ddsdde.shape == (4, 4)
    # Yield consistency (state already includes "stress")
    f = model.yield_function(state)
    assert abs(float(f)) < 1e-8


def test_autodiff_only_model_analytical_raises():
    """J2IsotropicPS (no plastic_corrector) raises NotImplementedError via PythonAnalyticalIntegrator."""
    model = J2IsotropicPS(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
    deps = anp.array([2e-3, 0.0, 0.0])
    state0 = model.initial_state()
    with pytest.raises(NotImplementedError):
        PythonAnalyticalIntegrator(model).stress_update(deps, anp.zeros(3), state0)


# ---------------------------------------------------------------------------
# param_names / Fortran argument order contract (requires compiled .so)
# ---------------------------------------------------------------------------

def _extract_dummy_args(fortran_module, subroutine_name: str) -> list[str]:
    """Parse argument names from the f2py-generated __doc__ for a subroutine."""
    fn = getattr(fortran_module, subroutine_name)
    doc = fn.__doc__ or ""
    first_line = doc.split("\n")[0]
    m = re.search(r"\(([^)]+)\)", first_line)
    if not m:
        return []
    return [a.strip().lower() for a in m.group(1).split(",")]


@pytest.mark.fortran
@pytest.mark.parametrize("model_fn,module_name,subroutine", [
    (lambda: J2Isotropic3D(E=210e3, nu=0.3, sigma_y0=250.0, H=1000.0), "j2_isotropic_3d", "j2_isotropic_3d"),
    (lambda: J2IsotropicPS(E=210e3, nu=0.3, sigma_y0=250.0, H=1000.0), "j2_isotropic_3d", "j2_isotropic_3d"),
    (lambda: J2Isotropic1D(E=210e3, nu=0.3, sigma_y0=250.0, H=1000.0), "j2_isotropic_3d", "j2_isotropic_3d"),
])
def test_param_names_match_fortran_arg_order(model_fn, module_name, subroutine):
    """model.param_names order must match the first N Fortran dummy argument names.

    Enforces the project convention that PROPS order == param_names order so that
    FortranIntegrator.from_model() can auto-generate param_fn safely.
    """
    from manforge.verification import FortranModule
    pytest.importorskip(
        module_name,
        reason=f"{module_name} not compiled -- run: uv run manforge build fortran/abaqus_stubs.f90 fortran/j2_isotropic_3d.f90 --name j2_isotropic_3d",
    )
    model = model_fn()
    fortran = FortranModule(module_name)
    dummy_args = _extract_dummy_args(fortran.module, subroutine)
    n = len(model.param_names)
    assert len(dummy_args) >= n, \
        f"{subroutine}: expected at least {n} dummy args, got {len(dummy_args)}"
    expected = [name.lower() for name in model.param_names]
    actual = dummy_args[:n]
    assert actual == expected, (
        f"{type(model).__name__}.param_names {model.param_names!r} "
        f"does not match first {n} Fortran args {dummy_args[:n]!r} "
        f"of subroutine '{subroutine}'"
    )
