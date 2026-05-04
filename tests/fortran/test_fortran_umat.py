"""Cross-validation: Fortran J2 UMAT vs. Python return_mapping.

Uses FortranModule.call to invoke compiled subroutines and compares results
explicitly against the Python reference implementation.

The compiled module must be built before running:

    make fortran-build-umat

If the module is not available, all tests in this file are skipped.
"""

import numpy as np
import autograd.numpy as anp
import pytest

pytest.importorskip(
    "j2_isotropic_3d",
    reason="j2_isotropic_3d not compiled -- run: make fortran-build-umat",
)

pytestmark = pytest.mark.fortran

from manforge.simulation.integrator import PythonIntegrator
from manforge.verification import FortranModule


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fortran():
    return FortranModule("j2_isotropic_3d")


# ---------------------------------------------------------------------------
# Shape / smoke tests
# ---------------------------------------------------------------------------

def test_call_j2_isotropic_3d(fortran, model):
    """j2_isotropic_3d subroutine returns stress (6,) and ddsdde (6,6)."""
    dstran = np.array([1e-4, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    stress_f, ep_f, ddsdde_f = fortran.call(
        "j2_isotropic_3d",
        model.E, model.nu, model.sigma_y0, model.H,
        np.zeros(6), 0.0, dstran,
    )
    assert np.asarray(stress_f).shape == (6,)
    assert np.asarray(ddsdde_f).shape == (6, 6)


def test_call_elastic_stiffness(fortran):
    """j2_isotropic_3d_elastic_stiffness returns a (6,6) matrix."""
    C = fortran.call("j2_isotropic_3d_elastic_stiffness", 210_000.0, 0.3)
    assert np.asarray(C).shape == (6, 6)


# ---------------------------------------------------------------------------
# Fortran vs Python comparison
# ---------------------------------------------------------------------------

def test_fortran_vs_python_elastic(fortran, model):
    """Elastic step: Fortran stress and tangent match Python reference."""
    dstran = np.array([1e-4, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    stress_f, ep_f, ddsdde_f = fortran.call(
        "j2_isotropic_3d",
        model.E, model.nu, model.sigma_y0, model.H,
        np.zeros(6), 0.0, dstran,
    )
    _r = PythonIntegrator(model).stress_update(anp.array(dstran), anp.zeros(6), model.initial_state())
    stress_py, ddsdde_py = _r.stress, _r.ddsdde

    np.testing.assert_allclose(np.array(stress_py), stress_f, rtol=1e-6)
    np.testing.assert_allclose(np.array(ddsdde_py), np.array(ddsdde_f), rtol=1e-5)


def test_fortran_vs_python_plastic_uniaxial(fortran, model):
    """Plastic uniaxial step: Fortran stress and tangent match Python reference."""
    dstran = np.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    stress_f, ep_f, ddsdde_f = fortran.call(
        "j2_isotropic_3d",
        model.E, model.nu, model.sigma_y0, model.H,
        np.zeros(6), 0.0, dstran,
    )
    _r = PythonIntegrator(model).stress_update(anp.array(dstran), anp.zeros(6), model.initial_state())
    stress_py, state_py, ddsdde_py = _r.stress, _r.state, _r.ddsdde

    np.testing.assert_allclose(np.array(stress_py), stress_f, rtol=1e-6)
    np.testing.assert_allclose(np.array(ddsdde_py), np.array(ddsdde_f), rtol=1e-5)
    assert abs(float(state_py["ep"]) - float(ep_f)) / (abs(float(state_py["ep"])) + 1e-14) < 1e-6


def test_fortran_vs_python_plastic_multiaxial(fortran, model):
    """Plastic multiaxial step: Fortran matches Python."""
    dstran = np.array([1.5e-3, -0.5e-3, -0.5e-3, 0.5e-3, 0.0, 0.0], dtype=np.float64)

    stress_f, ep_f, ddsdde_f = fortran.call(
        "j2_isotropic_3d",
        model.E, model.nu, model.sigma_y0, model.H,
        np.zeros(6), 0.0, dstran,
    )
    _r = PythonIntegrator(model).stress_update(anp.array(dstran), anp.zeros(6), model.initial_state())
    stress_py, ddsdde_py = _r.stress, _r.ddsdde

    np.testing.assert_allclose(np.array(stress_py), stress_f, rtol=1e-6)
    np.testing.assert_allclose(np.array(ddsdde_py), np.array(ddsdde_f), rtol=1e-5)


def test_elastic_stiffness_vs_python(fortran, model):
    """Elastic stiffness sub-component: Fortran matches Python to near machine precision."""
    C_f = fortran.call("j2_isotropic_3d_elastic_stiffness", model.E, model.nu)
    C_py = model.elastic_stiffness(model.initial_state())

    np.testing.assert_allclose(np.array(C_py), np.array(C_f), rtol=1e-12)


# ---------------------------------------------------------------------------
# Multi-step: independent state propagation
# ---------------------------------------------------------------------------

def test_multi_step_tension_unload_compression(fortran, model):
    """Multi-step tension-unload-compression: accumulated error stays small."""
    eps_y = model.sigma_y0 / model.E
    n = 35
    tension = np.linspace(0.0, 3.0 * eps_y, n // 2 + 1)
    compression = np.linspace(tension[-1], -3.0 * eps_y, n - len(tension) + 2)[1:]
    strain_vals = np.concatenate([tension, compression])

    # Build full strain history (uniaxial, component 0)
    ntens = model.ntens
    history = np.zeros((len(strain_vals), ntens))
    history[:, 0] = strain_vals

    stress_py = anp.zeros(ntens)
    state_py = model.initial_state()
    stress_f = np.zeros(ntens)
    ep_f = 0.0
    eps_prev = np.zeros(ntens)

    max_stress_err = 0.0

    for i in range(len(history)):
        dstran = history[i] - eps_prev
        eps_prev = history[i].copy()

        _r = PythonIntegrator(model).stress_update(anp.array(dstran), stress_py, state_py)
        stress_py, state_py = _r.stress, _r.state
        stress_f, ep_f, _ = fortran.call(
            "j2_isotropic_3d",
            model.E, model.nu, model.sigma_y0, model.H,
            stress_f, ep_f, dstran,
        )

        s_py = np.asarray(stress_py, dtype=float)
        s_f = np.asarray(stress_f, dtype=float)
        err = float(np.max(np.abs(s_f - s_py) / (np.abs(s_py) + 1.0)))
        max_stress_err = max(max_stress_err, err)

    assert max_stress_err < 1e-6, f"Max stress error {max_stress_err:.2e} exceeds 1e-6"
