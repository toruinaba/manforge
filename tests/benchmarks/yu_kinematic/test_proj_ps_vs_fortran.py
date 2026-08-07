"""YUKinematicProjPS: Python vs the Fortran port of the projected stagnation update.

Requires the compiled extension:
    make fortran-build-yu-proj-ps
or, since the host has no gfortran:
    make docker-build && make docker-test-yu

The port is a copy of yu_kinematic_ps.f90 with only the stagnation block
replaced, so what needs checking is that the replacement is faithful and that
the copied machinery still agrees with Python -- a rename that missed a call
site, or a leftover reference to the deleted mu routine, shows up here.
"""

import itertools

import numpy as np
import pytest

pytest.importorskip(
    "yu_kinematic_proj_ps",
    reason="yu_kinematic_proj_ps not compiled -- run: make docker-test-yu",
)

pytestmark = pytest.mark.fortran

from manforge.models import YUKinematicProjPS
from manforge.simulation.integrator import FortranModule, PythonAnalyticalIntegrator

from .conftest import PARAMS

STATE_KEYS = ("theta", "beta", "R", "q", "r", "eps_eq", "theta_max")


@pytest.fixture
def fortran_mod():
    return FortranModule("yu_kinematic_proj_ps")


@pytest.fixture
def model():
    return YUKinematicProjPS(**PARAMS)


def _initial_state():
    return dict(
        stress=np.zeros(3), theta=np.zeros(3), beta=np.zeros(3), R=0.0,
        q=np.zeros(3), r=0.0, eps_eq=0.0, theta_max=0.0,
    )


def _props(m):
    return (m.E, m.nu, m.Y, m.B, m.C_1, m.C_2, m.Rsat, m.k, m.b, m.h, m.Ea, m.xi)


def _unpack(out):
    return dict(
        theta=np.asarray(out[1]), beta=np.asarray(out[2]), R=float(out[3]),
        q=np.asarray(out[4]), r=float(out[5]), eps_eq=float(out[6]),
        theta_max=float(out[7]),
    )


def _call(fortran_mod, model, stress, state, strain_inc):
    return fortran_mod.call(
        "yu_kinematic_proj_ps", *_props(model),
        stress,
        np.asarray(state["theta"]), np.asarray(state["beta"]), float(state["R"]),
        np.asarray(state["q"]), float(state["r"]), float(state["eps_eq"]),
        float(state["theta_max"]),
        strain_inc,
    )


@pytest.mark.parametrize("strain_inc", [
    np.array([1.0e-3, -0.3e-3, 0.0]),
    np.array([0.0, 0.0, 1.5e-3]),
    np.array([8.0e-4, 4.0e-4, -6.0e-4]),
])
def test_trajectory_matches_python(model, fortran_mod, strain_inc):
    """March both forward together; a faithful port stays with Python."""
    integrator = PythonAnalyticalIntegrator(model)
    stress_py, state_py = np.zeros(3), _initial_state()
    stress_f, state_f = np.zeros(3), _initial_state()
    n_plastic = 0

    for step in range(25):
        result = integrator.stress_update(strain_inc, stress_py, state_py)
        stress_py, state_py = np.asarray(result.stress), dict(result.state)

        out = _call(fortran_mod, model, stress_f, state_f, strain_inc)
        assert int(out[10]) == 1, (
            f"Fortran did not converge at step {step}, fail_code={int(out[12])}"
        )
        stress_f, state_f = np.asarray(out[0]), _unpack(out)
        if result.is_plastic:
            n_plastic += 1

        np.testing.assert_allclose(stress_f, stress_py, rtol=1e-8, atol=1e-8,
                                   err_msg=f"stress diverged at step {step}")
        np.testing.assert_allclose(np.asarray(result.ddsdde), np.asarray(out[8]),
                                   rtol=1e-7, atol=1e-7,
                                   err_msg=f"ddsdde diverged at step {step}")
        for key in STATE_KEYS:
            np.testing.assert_allclose(
                np.asarray(state_f[key]), np.asarray(state_py[key]),
                rtol=1e-8, atol=1e-8,
                err_msg=f"state[{key!r}] diverged at step {step}")

    assert n_plastic > 0, "trajectory never yielded"


def test_reversals_match_python(model, fortran_mod):
    """Reversals are where the stagnation surface actually does something."""
    integrator = PythonAnalyticalIntegrator(model)
    sequence = list(itertools.chain(*[
        [np.array([a, -a * 0.35, 0.0])] * 25 for a in (2e-3, -2e-3, 2e-3)
    ]))
    stress_py, state_py = np.zeros(3), _initial_state()
    stress_f, state_f = np.zeros(3), _initial_state()
    r_seen = []

    for step, strain_inc in enumerate(sequence):
        result = integrator.stress_update(strain_inc, stress_py, state_py)
        stress_py, state_py = np.asarray(result.stress), dict(result.state)
        out = _call(fortran_mod, model, stress_f, state_f, strain_inc)
        assert int(out[10]) == 1, f"Fortran failed at step {step}"
        stress_f, state_f = np.asarray(out[0]), _unpack(out)
        r_seen.append(float(state_f["r"]))

        np.testing.assert_allclose(stress_f, stress_py, rtol=1e-8, atol=1e-8,
                                   err_msg=f"stress diverged at step {step}")
        for key in STATE_KEYS:
            np.testing.assert_allclose(
                np.asarray(state_f[key]), np.asarray(state_py[key]),
                rtol=1e-8, atol=1e-8,
                err_msg=f"state[{key!r}] diverged at step {step}")

    assert max(r_seen) > 1.0, "stagnation surface never grew -- test proves nothing"


def test_beta_lands_on_the_stagnation_surface(model, fortran_mod):
    """The port must reproduce the projection property, not just the numbers.

    ``‖beta - q‖ - r == 0`` is what the projected update is for; checking it on
    the Fortran output catches a transcription slip that a stress comparison at
    1e-8 could absorb.
    """
    sequence = list(itertools.chain(*[
        [np.array([a, -a * 0.35, 0.0])] * 25 for a in (2e-3, -2e-3)
    ]))
    stress, state = np.zeros(3), _initial_state()
    n_active = 0

    for step, strain_inc in enumerate(sequence):
        r_before = float(state["r"])
        out = _call(fortran_mod, model, stress, state, strain_inc)
        assert int(out[10]) == 1, f"Fortran failed at step {step}"
        stress, state = np.asarray(out[0]), _unpack(out)
        if abs(float(state["r"]) - r_before) < 1e-14:
            continue
        n_active += 1
        residual = model.vonmises_norm(
            np.asarray(state["beta"]) - np.asarray(state["q"])
        ) - float(state["r"])
        assert abs(float(residual)) < 1e-9, (
            f"step {step}: beta is {residual:.3e} off the stagnation surface"
        )

    assert n_active > 0, "stagnation surface never updated"


def test_no_mu_failures_are_possible(model, fortran_mod):
    """fail_code=2 is unreachable: there is no mu iteration left to fail.

    The strain increment here is large enough to exhaust the outer NR, which
    must report code 1 -- the published port hits code 2 on comparable states.
    """
    out = _call(fortran_mod, model, np.zeros(3), _initial_state(),
                np.array([5.0, -2.0, 1.0]))
    assert int(out[10]) == 0, "increment was not severe enough to fail"
    assert int(out[12]) == 1, f"expected NR exhaustion, got fail_code={int(out[12])}"
    assert int(out[9]) == 50
