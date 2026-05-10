"""Tests for MixedDriver (mixed strain/stress boundary-condition driver)."""

import numpy as np
import pytest

from manforge.core.dimension import SOLID_3D, UNIAXIAL_1D
from manforge.models.j2_isotropic import J2Isotropic1D, J2Isotropic3D
from manforge.simulation.driver import MixedDriver, StrainDriver, StressDriver
from manforge.simulation.integrator import PythonNumericalIntegrator
from manforge.simulation.types import FieldHistory, FieldType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strain_load(data, name="Strain"):
    return FieldHistory(FieldType.STRAIN, name, data)


def stress_load(data, name="Stress"):
    return FieldHistory(FieldType.STRESS, name, data)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model_3d():
    return J2Isotropic3D(SOLID_3D, E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)


@pytest.fixture
def model_1d():
    return J2Isotropic1D(UNIAXIAL_1D, E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)


@pytest.fixture
def integ_3d(model_3d):
    return PythonNumericalIntegrator(model_3d)


@pytest.fixture
def integ_1d(model_1d):
    return PythonNumericalIntegrator(model_1d)


# ---------------------------------------------------------------------------
# __init__ validation
# ---------------------------------------------------------------------------

def test_validation_duplicate_strain_idx(integ_3d):
    with pytest.raises(ValueError, match="duplicate"):
        MixedDriver(integ_3d, prescribed_strain_idx=[0, 0])


def test_validation_out_of_range(integ_3d):
    with pytest.raises(ValueError, match="out of range"):
        MixedDriver(integ_3d, prescribed_strain_idx=[7])


def test_validation_overlap(integ_3d):
    with pytest.raises(ValueError, match="disjoint"):
        MixedDriver(integ_3d, prescribed_strain_idx=[0, 1],
                    prescribed_stress_idx=[1, 2, 3, 4, 5])


def test_validation_incomplete_union(integ_3d):
    with pytest.raises(ValueError, match="cover all"):
        MixedDriver(integ_3d, prescribed_strain_idx=[0],
                    prescribed_stress_idx=[1, 2, 3])  # missing 4, 5


def test_validation_empty_strain_idx(integ_3d):
    with pytest.raises(ValueError, match="must not be empty"):
        MixedDriver(integ_3d, prescribed_strain_idx=[])


def test_validation_load_type_must_be_strain(integ_3d):
    driver = MixedDriver(integ_3d, prescribed_strain_idx=[0])
    N = 5
    bad_load = stress_load(np.zeros((N, 6)))
    with pytest.raises(ValueError, match="FieldType.STRAIN"):
        list(driver.iter_run(bad_load))


def test_validation_load_shape_must_match_nP(integ_3d):
    driver = MixedDriver(integ_3d, prescribed_strain_idx=[0])
    N = 5
    bad_load = strain_load(np.zeros((N, 6)))  # expects (N, 1)
    with pytest.raises(ValueError, match=r"\(N, 1\)"):
        list(driver.iter_run(bad_load))


def test_validation_stress_history_shape_mismatch(integ_3d):
    driver = MixedDriver(integ_3d, prescribed_strain_idx=[0])
    N = 5
    load = strain_load(np.zeros((N, 1)))
    bad_sig = np.zeros((N, 3))  # expects (N, 5)
    with pytest.raises(ValueError, match=r"\(5, 5\)"):
        list(driver.iter_run(load, prescribed_stress_history=bad_sig))


def test_complement_inferred_from_strain_idx(integ_3d):
    driver = MixedDriver(integ_3d, prescribed_strain_idx=[0])
    np.testing.assert_array_equal(driver._P, [0])
    np.testing.assert_array_equal(driver._F, [1, 2, 3, 4, 5])


def test_explicit_stress_idx_matches_complement(integ_3d):
    driver = MixedDriver(integ_3d, prescribed_strain_idx=[0],
                         prescribed_stress_idx=[1, 2, 3, 4, 5])
    np.testing.assert_array_equal(driver._P, [0])
    np.testing.assert_array_equal(driver._F, [1, 2, 3, 4, 5])


def test_validation_stress_history_provided_when_nF_zero(integ_3d):
    driver = MixedDriver(integ_3d, prescribed_strain_idx=[0, 1, 2, 3, 4, 5])
    N = 3
    load = strain_load(np.zeros((N, 6)))
    with pytest.raises(ValueError, match="no stress-prescribed"):
        list(driver.iter_run(load, prescribed_stress_history=np.zeros((N, 1))))


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

def test_output_shapes(integ_3d):
    N = 10
    driver = MixedDriver(integ_3d, prescribed_strain_idx=[0])
    load = strain_load(np.linspace(0, 0.005, N).reshape(-1, 1))
    result = driver.run(load)
    assert result.stress.shape == (N, 6)
    assert result.strain.shape == (N, 6)


# ---------------------------------------------------------------------------
# Elastic uniaxial: 3D model, ε11 controlled, σ_other = 0
# ---------------------------------------------------------------------------

def test_uniaxial_elastic_stress_11(model_3d, integ_3d):
    E, nu = model_3d.E, model_3d.nu
    sigma_y0 = model_3d.sigma_y0
    N = 20
    eps_max = 0.5 * sigma_y0 / E  # stay elastic

    load = strain_load(np.linspace(0, eps_max, N).reshape(-1, 1))
    driver = MixedDriver(integ_3d, prescribed_strain_idx=[0])
    result = driver.run(load)

    eps_hist = np.linspace(0, eps_max, N)
    np.testing.assert_allclose(result.stress[:, 0], E * eps_hist, rtol=1e-5,
                               err_msg="σ11 must equal E * ε11 in elastic regime")


def test_uniaxial_elastic_lateral_stress_zero(model_3d, integ_3d):
    E, sigma_y0 = model_3d.E, model_3d.sigma_y0
    N = 20
    eps_max = 0.5 * sigma_y0 / E

    load = strain_load(np.linspace(0, eps_max, N).reshape(-1, 1))
    result = MixedDriver(integ_3d, prescribed_strain_idx=[0]).run(load)

    np.testing.assert_allclose(result.stress[:, 1:], 0.0, atol=1e-6,
                               err_msg="σ_other must be ~0 under uniaxial stress control (σ=0 prescribed)")


def test_uniaxial_elastic_lateral_strains(model_3d, integ_3d):
    E, nu, sigma_y0 = model_3d.E, model_3d.nu, model_3d.sigma_y0
    N = 20
    eps_max = 0.5 * sigma_y0 / E
    eps_hist = np.linspace(0, eps_max, N)

    load = strain_load(eps_hist.reshape(-1, 1))
    result = MixedDriver(integ_3d, prescribed_strain_idx=[0]).run(load)

    np.testing.assert_allclose(result.strain[:, 1], -nu * eps_hist, rtol=1e-5,
                               err_msg="ε22 must equal -ν·ε11 in elastic regime")
    np.testing.assert_allclose(result.strain[:, 2], -nu * eps_hist, rtol=1e-5,
                               err_msg="ε33 must equal -ν·ε11 in elastic regime")
    np.testing.assert_allclose(result.strain[:, 3:], 0.0, atol=1e-10,
                               err_msg="shear strains must be zero under uniaxial loading")


# ---------------------------------------------------------------------------
# Plastic: MixedDriver(ε11 control) matches StressDriver(σ11 target, others=0)
# ---------------------------------------------------------------------------

def test_uniaxial_plastic_matches_stressdriver(model_3d, integ_3d):
    """MixedDriver and StressDriver must agree on stress and strain under uniaxial loading."""
    sigma_y0 = model_3d.sigma_y0
    E = model_3d.E
    N = 30
    eps_max = 3.0 * sigma_y0 / E  # well into plastic zone

    load_mixed = strain_load(np.linspace(0, eps_max, N).reshape(-1, 1))
    res_mixed = MixedDriver(integ_3d, prescribed_strain_idx=[0]).run(load_mixed)

    # Build σ11 target history from MixedDriver output, feed into StressDriver
    sig_history = np.zeros((N, 6))
    sig_history[:, 0] = res_mixed.stress[:, 0]
    res_stress = StressDriver(integ_3d).run(stress_load(sig_history))

    np.testing.assert_allclose(res_mixed.stress, res_stress.stress, atol=1e-5,
                               err_msg="Stress mismatch between MixedDriver and StressDriver")
    np.testing.assert_allclose(res_mixed.strain, res_stress.strain, atol=1e-5,
                               err_msg="Strain mismatch between MixedDriver and StressDriver")


# ---------------------------------------------------------------------------
# Consistency with 1D StrainDriver
# ---------------------------------------------------------------------------

def test_consistency_with_1d_strain_driver(model_3d, integ_3d, model_1d, integ_1d):
    """MixedDriver(3D, ε11 control) must produce the same σ11 and ep as StrainDriver(1D)."""
    sigma_y0 = model_3d.sigma_y0
    E = model_3d.E
    N = 30
    eps_max = 3.0 * sigma_y0 / E

    eps_hist = np.linspace(0, eps_max, N)
    load_mixed = strain_load(eps_hist.reshape(-1, 1))
    res_3d = MixedDriver(integ_3d, prescribed_strain_idx=[0]).run(
        load_mixed, collect_state={"ep": FieldType.STRAIN}
    )

    # 1D StrainDriver with the same ε11 history (shape (N,) uniaxial)
    res_1d = StrainDriver(integ_1d).run(
        FieldHistory(FieldType.STRAIN, "Strain", eps_hist),
        collect_state={"ep": FieldType.STRAIN},
    )

    np.testing.assert_allclose(res_3d.stress[:, 0], res_1d.stress[:, 0], rtol=1e-4,
                               err_msg="σ11 mismatch between 3D MixedDriver and 1D StrainDriver")
    np.testing.assert_allclose(res_3d.fields["ep"].data, res_1d.fields["ep"].data, rtol=1e-4,
                               err_msg="ep mismatch between 3D MixedDriver and 1D StrainDriver")


# ---------------------------------------------------------------------------
# Biaxial: ε11 controlled + σ22 = constant lateral pressure
# ---------------------------------------------------------------------------

def test_biaxial_constant_lateral_stress(model_3d, integ_3d):
    """σ22 target must be satisfied at every step while ε11 is ramped."""
    sigma_y0 = model_3d.sigma_y0
    E = model_3d.E
    N = 30
    eps_max = 2.0 * sigma_y0 / E
    sigma22_target = -100.0  # compressive lateral pressure

    load = strain_load(np.linspace(0, eps_max, N).reshape(-1, 1))

    # prescribed_stress_history: (N, 5) with σ22 in first column
    sig_F = np.zeros((N, 5))
    sig_F[:, 0] = sigma22_target  # F=[1,2,3,4,5] so index 0 → Voigt component 1 = σ22

    driver = MixedDriver(integ_3d, prescribed_strain_idx=[0])
    result = driver.run(load, prescribed_stress_history=sig_F)

    np.testing.assert_allclose(result.stress[:, 1], sigma22_target, atol=1e-5,
                               err_msg="σ22 must match the prescribed target")
    np.testing.assert_allclose(result.stress[:, 2:], 0.0, atol=1e-5,
                               err_msg="σ33, σ12, σ13, σ23 must be ~0 (not prescribed)")


# ---------------------------------------------------------------------------
# Degenerate case: full strain control (nF = 0)
# ---------------------------------------------------------------------------

def test_full_strain_control_matches_strain_driver(model_3d, integ_3d):
    """MixedDriver with all 6 components strain-controlled must agree with StrainDriver."""
    sigma_y0 = model_3d.sigma_y0
    E = model_3d.E
    N = 20

    strain6 = np.zeros((N, 6))
    strain6[:, 0] = np.linspace(0, 3.0 * sigma_y0 / E, N)

    load_mixed = strain_load(strain6)
    load_strain = FieldHistory(FieldType.STRAIN, "Strain", strain6)

    res_mixed = MixedDriver(integ_3d, prescribed_strain_idx=[0, 1, 2, 3, 4, 5]).run(load_mixed)
    res_strain = StrainDriver(integ_3d).run(load_strain)

    np.testing.assert_allclose(res_mixed.stress, res_strain.stress, atol=1e-10)
    np.testing.assert_allclose(res_mixed.strain, res_strain.strain, atol=1e-10)


# ---------------------------------------------------------------------------
# Convergence failure
# ---------------------------------------------------------------------------

def test_max_iter_one_raises(model_3d, integ_3d):
    """max_iter=1 must fail on a plastic step and raise RuntimeError."""
    sigma_y0 = model_3d.sigma_y0
    E = model_3d.E
    # single large step into plastic zone
    load = strain_load(np.array([[3.0 * sigma_y0 / E]]))
    driver = MixedDriver(integ_3d, prescribed_strain_idx=[0], max_iter=1)
    with pytest.raises(RuntimeError, match="NR did not converge"):
        driver.run(load)


def test_raise_on_nonconverged_false_yields_and_stops(model_3d, integ_3d):
    """raise_on_nonconverged=False must yield a non-converged step and stop."""
    sigma_y0 = model_3d.sigma_y0
    E = model_3d.E
    N = 3
    eps_hist = np.linspace(0, 3.0 * sigma_y0 / E, N).reshape(-1, 1)
    load = strain_load(eps_hist)

    driver = MixedDriver(integ_3d, prescribed_strain_idx=[0], max_iter=1)
    steps = list(driver.iter_run(load, raise_on_nonconverged=False))

    assert len(steps) >= 1
    # At least one step must be non-converged (the driver stops after the first failure)
    non_converged = [s for s in steps if not s.converged]
    assert len(non_converged) == 1
    assert non_converged[0].i < N  # stopped before end


# ---------------------------------------------------------------------------
# initial_stress / initial_state
# ---------------------------------------------------------------------------

def test_initial_stress_zero_increment_preserves_prestress(model_3d, integ_3d):
    """A zero strain increment from a prestressed state must leave stress unchanged."""
    prestress = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    load = strain_load(np.zeros((1, 1)))  # ε11 increment = 0
    driver = MixedDriver(integ_3d, prescribed_strain_idx=[0])
    steps = list(driver.iter_run(load, initial_stress=prestress))
    np.testing.assert_allclose(
        np.array(steps[0].result.stress_trial), prestress, atol=1e-12,
        err_msg="Zero increment from prestressed state must keep trial stress == prestress",
    )


def test_initial_state_shifts_yield_surface(model_3d, integ_3d):
    """Non-zero initial ep must raise the yield surface (step stays elastic when hardened)."""
    import autograd.numpy as anp
    sigma_y0, E, H = model_3d.sigma_y0, model_3d.E, model_3d.H
    eps_y = sigma_y0 / E
    load = strain_load(np.array([[eps_y * 1.5]]))  # 1.5× yield strain — plastic without hardening

    driver = MixedDriver(integ_3d, prescribed_strain_idx=[0])
    step_fresh = list(driver.iter_run(load))[0]
    assert step_fresh.result.is_plastic, "Should be plastic without prior hardening"

    # ep large enough that sigma_y0 + H*ep > sigma_VM(trial)
    # sigma_VM(trial) for uniaxial eps_y*1.5 in 3D ~ E*eps_y*1.5 ≈ 375 MPa.
    # Need: 250 + 1000*ep > 375 → ep > 0.125; use ep=0.2 for margin.
    large_ep = anp.array(0.2)
    step_hardened = list(driver.iter_run(load, initial_state={"ep": large_ep}))[0]
    assert not step_hardened.result.is_plastic, "Should be elastic with large initial ep"


def test_initial_stress_state_run_forwards(model_3d, integ_3d):
    """run() must accept and forward initial_stress / initial_state without error,
    and initial_stress must be reflected in the first step's trial stress."""
    sigma_y0 = model_3d.sigma_y0
    prestress = np.array([sigma_y0 * 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Zero strain increment: trial stress must equal prestress
    load = strain_load(np.zeros((1, 1)))
    result = MixedDriver(integ_3d, prescribed_strain_idx=[0]).run(
        load, initial_stress=prestress
    )
    assert result.stress.shape == (1, 6)
    np.testing.assert_allclose(
        np.array(result.step_results[0].stress_trial), prestress, atol=1e-12,
        err_msg="run() must forward initial_stress: trial stress must equal prestress for zero increment",
    )


# ---------------------------------------------------------------------------
# collect_state
# ---------------------------------------------------------------------------

def test_collect_state_ep(model_3d, integ_3d):
    """collect_state must include ep in the result fields."""
    sigma_y0 = model_3d.sigma_y0
    E = model_3d.E
    N = 20
    load = strain_load(np.linspace(0, 3.0 * sigma_y0 / E, N).reshape(-1, 1))
    result = MixedDriver(integ_3d, prescribed_strain_idx=[0]).run(
        load, collect_state={"ep": FieldType.STRAIN}
    )
    assert "ep" in result.fields
    ep = result.fields["ep"].data
    assert ep.shape == (N,)
    assert result.fields["ep"].type == FieldType.STRAIN
    assert ep[-1] > 0.0, "ep should be positive after plastic loading"


# ---------------------------------------------------------------------------
# ε_P fidelity: prescribed strain components must equal the input history
# ---------------------------------------------------------------------------

def test_prescribed_strain_equals_input(model_3d, integ_3d):
    """eps_total[P] at each step must equal the prescribed strain history."""
    sigma_y0 = model_3d.sigma_y0
    E = model_3d.E
    N = 30
    eps_hist = np.linspace(0, 3.0 * sigma_y0 / E, N)

    load = strain_load(eps_hist.reshape(-1, 1))
    result = MixedDriver(integ_3d, prescribed_strain_idx=[0]).run(load)

    np.testing.assert_allclose(result.strain[:, 0], eps_hist, atol=1e-12,
                               err_msg="ε11 (P component) must equal the prescribed input history")
