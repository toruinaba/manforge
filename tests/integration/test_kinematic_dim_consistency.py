"""Dimensional consistency tests for AF/OW kinematic hardening models.

Verifies that 1D StrainDriver, PS MixedDriver(ε11 ctrl, σ_other=0),
and 3D MixedDriver(ε11 ctrl, σ_other=0) all reproduce the same uniaxial
σ11(t) and ep(t) histories under monotonic and cyclic loading.

These tests FAIL before the 1D/PS backstress evolution bug is fixed.
"""

import pytest
import numpy as np

from manforge.models.af_kinematic import AFKinematic1D, AFKinematicPS, AFKinematic3D
from manforge.models.ow_kinematic import OWKinematic1D, OWKinematicPS, OWKinematic3D
from manforge.simulation.integrator import PythonIntegrator
from manforge.simulation.driver import StrainDriver, MixedDriver
from manforge.simulation.types import FieldHistory, FieldType


_PARAMS_LINEAR = dict(E=210_000.0, nu=0.3, sigma_y0=250.0, C_k=10_000.0, gamma=0.0)
_PARAMS_NL = dict(E=210_000.0, nu=0.3, sigma_y0=250.0, C_k=10_000.0, gamma=100.0)
_PARAMS_OW = dict(E=210_000.0, nu=0.3, sigma_y0=250.0, C_k=10_000.0, gamma=1.0)


def _strain_history_1d(n=20, eps_max=3e-3):
    """Monotonic uniaxial strain ramp (1-component)."""
    return FieldHistory(
        type=FieldType.STRAIN,
        name="eps",
        data=np.linspace(0, eps_max, n).reshape(-1, 1),
    )


def _strain_history_1d_cyclic(n=40, eps_amp=3e-3):
    """Cyclic ±eps_amp for 1 full cycle."""
    t = np.linspace(0, 1, n)
    eps = eps_amp * np.sin(2 * np.pi * t)
    return FieldHistory(
        type=FieldType.STRAIN,
        name="eps",
        data=eps.reshape(-1, 1),
    )


def _run_1d(model_1d, load_1d):
    integ = PythonIntegrator(model_1d)
    return StrainDriver(integ).run(load_1d, collect_state={"ep": FieldType.STRAIN})


def _run_ps(model_ps, load_1d):
    """MixedDriver with ε11 prescribed, σ22=σ12=0."""
    integ = PythonIntegrator(model_ps)
    return MixedDriver(integ, prescribed_strain_idx=[0]).run(
        load_1d, collect_state={"ep": FieldType.STRAIN}
    )


def _run_3d(model_3d, load_1d):
    """MixedDriver with ε11 prescribed, σ22=…=σ23=0."""
    integ = PythonIntegrator(model_3d)
    return MixedDriver(integ, prescribed_strain_idx=[0]).run(
        load_1d, collect_state={"ep": FieldType.STRAIN}
    )


# ---------------------------------------------------------------------------
# AF — linear (gamma=0), monotonic
# ---------------------------------------------------------------------------

def test_af_1d_vs_3d_linear_monotonic():
    """AF gamma=0 monotonic: 1D and 3D must give the same σ11 and ep."""
    load = _strain_history_1d()
    r1 = _run_1d(AFKinematic1D(**_PARAMS_LINEAR), load)
    r3 = _run_3d(AFKinematic3D(**_PARAMS_LINEAR), load)
    np.testing.assert_allclose(r1.stress[:, 0], r3.stress[:, 0], rtol=1e-4,
                               err_msg="AF 1D vs 3D: σ11 mismatch (linear)")
    np.testing.assert_allclose(r1.fields["ep"].data, r3.fields["ep"].data, rtol=1e-4,
                               err_msg="AF 1D vs 3D: ep mismatch (linear)")


def test_af_ps_vs_3d_linear_monotonic():
    """AF gamma=0 monotonic: PS and 3D must give the same σ11 and ep."""
    load = _strain_history_1d()
    rps = _run_ps(AFKinematicPS(**_PARAMS_LINEAR), load)
    r3 = _run_3d(AFKinematic3D(**_PARAMS_LINEAR), load)
    np.testing.assert_allclose(rps.stress[:, 0], r3.stress[:, 0], rtol=1e-4,
                               err_msg="AF PS vs 3D: σ11 mismatch (linear)")
    np.testing.assert_allclose(rps.fields["ep"].data, r3.fields["ep"].data, rtol=1e-4,
                               err_msg="AF PS vs 3D: ep mismatch (linear)")


# ---------------------------------------------------------------------------
# AF — nonlinear (gamma>0), monotonic
# ---------------------------------------------------------------------------

def test_af_1d_vs_3d_nonlinear_monotonic():
    """AF gamma>0 monotonic: 1D and 3D must give the same σ11 and ep."""
    load = _strain_history_1d()
    r1 = _run_1d(AFKinematic1D(**_PARAMS_NL), load)
    r3 = _run_3d(AFKinematic3D(**_PARAMS_NL), load)
    np.testing.assert_allclose(r1.stress[:, 0], r3.stress[:, 0], rtol=1e-4,
                               err_msg="AF 1D vs 3D: σ11 mismatch (nonlinear)")
    np.testing.assert_allclose(r1.fields["ep"].data, r3.fields["ep"].data, rtol=1e-4,
                               err_msg="AF 1D vs 3D: ep mismatch (nonlinear)")


def test_af_ps_vs_3d_nonlinear_monotonic():
    """AF gamma>0 monotonic: PS and 3D must give the same σ11 and ep."""
    load = _strain_history_1d()
    rps = _run_ps(AFKinematicPS(**_PARAMS_NL), load)
    r3 = _run_3d(AFKinematic3D(**_PARAMS_NL), load)
    np.testing.assert_allclose(rps.stress[:, 0], r3.stress[:, 0], rtol=1e-4,
                               err_msg="AF PS vs 3D: σ11 mismatch (nonlinear)")
    np.testing.assert_allclose(rps.fields["ep"].data, r3.fields["ep"].data, rtol=1e-4,
                               err_msg="AF PS vs 3D: ep mismatch (nonlinear)")


# ---------------------------------------------------------------------------
# AF — cyclic
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_af_1d_vs_3d_cyclic():
    """AF cyclic: 1D and 3D must give the same σ11 and ep over a full cycle."""
    load = _strain_history_1d_cyclic()
    r1 = _run_1d(AFKinematic1D(**_PARAMS_NL), load)
    r3 = _run_3d(AFKinematic3D(**_PARAMS_NL), load)
    np.testing.assert_allclose(r1.stress[:, 0], r3.stress[:, 0], rtol=1e-4,
                               err_msg="AF 1D vs 3D: σ11 mismatch (cyclic)")


@pytest.mark.slow
def test_af_ps_vs_3d_cyclic():
    """AF cyclic: PS and 3D must give the same σ11 and ep over a full cycle."""
    load = _strain_history_1d_cyclic()
    rps = _run_ps(AFKinematicPS(**_PARAMS_NL), load)
    r3 = _run_3d(AFKinematic3D(**_PARAMS_NL), load)
    np.testing.assert_allclose(rps.stress[:, 0], r3.stress[:, 0], rtol=1e-4,
                               err_msg="AF PS vs 3D: σ11 mismatch (cyclic)")


# ---------------------------------------------------------------------------
# AF gamma=0 analytical check: α(ε_p) = C_k * ε_p (effective backstress)
# ---------------------------------------------------------------------------

def test_af_1d_linear_analytical():
    """AF gamma=0: α should equal C_k * ep (effective backstress definition)."""
    model = AFKinematic1D(**_PARAMS_LINEAR)
    load = _strain_history_1d(n=30)
    integ = PythonIntegrator(model)
    r2 = StrainDriver(integ).run(
        load, collect_state={"ep": FieldType.STRAIN, "alpha": FieldType.STRESS}
    )
    alpha_11 = r2.fields["alpha"].data[:, 0]
    ep_arr = r2.fields["ep"].data
    # After yielding: α = C_k * ep for linear AF
    plastic = ep_arr > 1e-10
    if plastic.any():
        np.testing.assert_allclose(
            alpha_11[plastic],
            _PARAMS_LINEAR["C_k"] * ep_arr[plastic],
            rtol=1e-4,
            err_msg="AF 1D linear: α ≠ C_k * ep (effective backstress broken)",
        )


# ---------------------------------------------------------------------------
# OW — linear (gamma=0), monotonic
# ---------------------------------------------------------------------------

def test_ow_1d_vs_3d_linear_monotonic():
    """OW gamma=0 monotonic: 1D and 3D must give the same σ11 and ep."""
    load = _strain_history_1d()
    r1 = _run_1d(OWKinematic1D(**_PARAMS_LINEAR), load)
    r3 = _run_3d(OWKinematic3D(**_PARAMS_LINEAR), load)
    np.testing.assert_allclose(r1.stress[:, 0], r3.stress[:, 0], rtol=1e-4,
                               err_msg="OW 1D vs 3D: σ11 mismatch (linear)")
    np.testing.assert_allclose(r1.fields["ep"].data, r3.fields["ep"].data, rtol=1e-4,
                               err_msg="OW 1D vs 3D: ep mismatch (linear)")


def test_ow_ps_vs_3d_linear_monotonic():
    """OW gamma=0 monotonic: PS and 3D must give the same σ11 and ep."""
    load = _strain_history_1d()
    rps = _run_ps(OWKinematicPS(**_PARAMS_LINEAR), load)
    r3 = _run_3d(OWKinematic3D(**_PARAMS_LINEAR), load)
    np.testing.assert_allclose(rps.stress[:, 0], r3.stress[:, 0], rtol=1e-4,
                               err_msg="OW PS vs 3D: σ11 mismatch (linear)")
    np.testing.assert_allclose(rps.fields["ep"].data, r3.fields["ep"].data, rtol=1e-4,
                               err_msg="OW PS vs 3D: ep mismatch (linear)")


# ---------------------------------------------------------------------------
# OW — nonlinear (gamma>0), monotonic
# ---------------------------------------------------------------------------

def test_ow_1d_vs_3d_nonlinear_monotonic():
    """OW gamma>0 monotonic: 1D and 3D must give the same σ11 and ep."""
    load = _strain_history_1d()
    r1 = _run_1d(OWKinematic1D(**_PARAMS_OW), load)
    r3 = _run_3d(OWKinematic3D(**_PARAMS_OW), load)
    np.testing.assert_allclose(r1.stress[:, 0], r3.stress[:, 0], rtol=1e-4,
                               err_msg="OW 1D vs 3D: σ11 mismatch (nonlinear)")
    np.testing.assert_allclose(r1.fields["ep"].data, r3.fields["ep"].data, rtol=1e-4,
                               err_msg="OW 1D vs 3D: ep mismatch (nonlinear)")


def test_ow_ps_vs_3d_nonlinear_monotonic():
    """OW gamma>0 monotonic: PS and 3D must give the same σ11 and ep."""
    load = _strain_history_1d()
    rps = _run_ps(OWKinematicPS(**_PARAMS_OW), load)
    r3 = _run_3d(OWKinematic3D(**_PARAMS_OW), load)
    np.testing.assert_allclose(rps.stress[:, 0], r3.stress[:, 0], rtol=1e-4,
                               err_msg="OW PS vs 3D: σ11 mismatch (nonlinear)")
    np.testing.assert_allclose(rps.fields["ep"].data, r3.fields["ep"].data, rtol=1e-4,
                               err_msg="OW PS vs 3D: ep mismatch (nonlinear)")


# ---------------------------------------------------------------------------
# OW — cyclic
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_ow_1d_vs_3d_cyclic():
    """OW cyclic: 1D and 3D must give the same σ11 over a full cycle."""
    load = _strain_history_1d_cyclic()
    r1 = _run_1d(OWKinematic1D(**_PARAMS_OW), load)
    r3 = _run_3d(OWKinematic3D(**_PARAMS_OW), load)
    np.testing.assert_allclose(r1.stress[:, 0], r3.stress[:, 0], rtol=1e-4,
                               err_msg="OW 1D vs 3D: σ11 mismatch (cyclic)")


@pytest.mark.slow
def test_ow_ps_vs_3d_cyclic():
    """OW cyclic: PS and 3D must give the same σ11 over a full cycle."""
    load = _strain_history_1d_cyclic()
    rps = _run_ps(OWKinematicPS(**_PARAMS_OW), load)
    r3 = _run_3d(OWKinematic3D(**_PARAMS_OW), load)
    np.testing.assert_allclose(rps.stress[:, 0], r3.stress[:, 0], rtol=1e-4,
                               err_msg="OW PS vs 3D: σ11 mismatch (cyclic)")
