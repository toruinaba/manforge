"""Path B benchmarks: YUKinematicPS Python vs Fortran subroutines.

Requires the compiled yu_kinematic_ps extension:
    make fortran-build-yu-ps
or, since the host has no gfortran:
    make docker-build && make docker-test-yu

Test classes:
    TestHelpers              -- P metric, norms, elastic stiffness
    TestJacobianBlocks       -- one test per calc_* block (16 total)
    TestResidualAndJacobian  -- calc_residual + calc_jacobian over a trajectory
    TestReturnMapping        -- yu_kinematic_ps core (return mapping + ddsdde)
    TestCrosscheckTrajectory -- PythonAnalyticalIntegrator vs FortranIntegrator
    TestUmatShim             -- ABAQUS umat entry point (STATEV packing, PNEWDT)
"""

import numpy as np
import pytest

pytest.importorskip(
    "yu_kinematic_ps",
    reason="yu_kinematic_ps not compiled -- run: make docker-test-yu",
)

pytestmark = pytest.mark.fortran

from manforge.models import YUKinematicPS
from manforge.simulation.integrator import (
    FortranModule,
    PythonAnalyticalIntegrator,
    PythonNumericalIntegrator,
)

from .conftest import PARAMS


@pytest.fixture
def fortran_mod():
    return FortranModule("yu_kinematic_ps")


@pytest.fixture
def model():
    return YUKinematicPS(**PARAMS)


def _initial_state(model):
    return dict(
        stress=np.zeros(3), theta=np.zeros(3), beta=np.zeros(3), R=0.0,
        q=np.zeros(3), r=0.0, eps_eq=0.0, theta_max=0.0,
    )


@pytest.fixture
def plastic_state(model):
    """Run several steps into the plastic regime; return (m, state, state_n, dl)."""
    integrator = PythonNumericalIntegrator(model)
    stress = np.zeros(3)
    state = _initial_state(model)
    strain_inc = np.array([1.0e-3, -0.3e-3, 0.0])
    for _ in range(10):
        state_n = state
        result = integrator.stress_update(strain_inc, stress, state)
        stress, state = np.asarray(result.stress), dict(result.state)
    assert result.is_plastic
    return model, state, dict(state_n), float(result.dlambda)


def _props(m):
    """12 material parameters in the Fortran positional order."""
    return (m.E, m.nu, m.Y, m.B, m.C_1, m.C_2, m.Rsat, m.k, m.b, m.h, m.Ea, m.xi)


# ---------------------------------------------------------------------------
# TestHelpers
# ---------------------------------------------------------------------------

class TestHelpers:
    """The P metric is where a plane-stress port most easily goes wrong."""

    def test_pmat(self, model, fortran_mod):
        np.testing.assert_allclose(
            model.P, fortran_mod.call("yu_ps_pmat"), rtol=1e-15, atol=1e-15
        )

    @pytest.mark.parametrize("vec", [
        np.array([120.0, -45.0, 0.0]),
        np.array([0.0, 0.0, 80.0]),
        np.array([200.0, 90.0, -60.0]),
    ])
    def test_pmul(self, model, fortran_mod, vec):
        np.testing.assert_allclose(
            model.P @ vec, fortran_mod.call("yu_ps_pmul", vec), rtol=1e-13
        )

    @pytest.mark.parametrize("vec", [
        np.array([120.0, -45.0, 0.0]),
        np.array([200.0, 90.0, -60.0]),
    ])
    def test_dev_inner_and_norm(self, model, fortran_mod, vec):
        np.testing.assert_allclose(
            model.deviatoric_inner_product(vec, vec),
            fortran_mod.call("yu_ps_dev_inner", vec, vec), rtol=1e-13,
        )
        np.testing.assert_allclose(
            model.vonmises_norm(vec),
            fortran_mod.call("yu_ps_vonmises_norm", vec), rtol=1e-13,
        )

    def test_elastic_stiffness(self, plastic_state, fortran_mod):
        m, state, _, _ = plastic_state
        np.testing.assert_allclose(
            m.elastic_stiffness(state),
            fortran_mod.call("yu_ps_elastic_stiffness",
                             m.E, m.nu, float(state["eps_eq"]), m.Ea, m.xi),
            rtol=1e-12, atol=1e-12,
        )


# ---------------------------------------------------------------------------
# TestJacobianBlocks
#
# Blocks needing theta_max read it from state_n on both sides, matching
# calc_residual.  Getting this wrong is a x(C_1/C_2) error on threshold-crossing
# steps, which is how the Fortran port surfaced it in the Python code.
# ---------------------------------------------------------------------------

class TestJacobianBlocks:
    """One test per calc_* block, Python vs Fortran at a single plastic step."""

    def test_fy_blocks(self, plastic_state, fortran_mod):
        m, state, _, _ = plastic_state
        s, th, be = state["stress"], state["theta"], state["beta"]
        np.testing.assert_allclose(
            m.calc_fy_fs(state), fortran_mod.call("yu_ps_fy_fs", s, th, be), rtol=1e-12)
        np.testing.assert_allclose(
            m.calc_fy_ft(state), fortran_mod.call("yu_ps_fy_ft", s, th, be), rtol=1e-12)
        np.testing.assert_allclose(
            m.calc_fy_fb(state), fortran_mod.call("yu_ps_fy_fb", s, th, be), rtol=1e-12)
        np.testing.assert_allclose(
            m.calc_fy_fl(state), fortran_mod.call("yu_ps_fy_fl"), rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize("name,py_method", [
        ("yu_ps_fe_fs", "calc_fe_fs"),
        ("yu_ps_fe_ft", "calc_fe_ft"),
        ("yu_ps_fe_fb", "calc_fe_fb"),
        ("yu_ps_fe_fl", "calc_fe_fl"),
    ])
    def test_fe_blocks(self, plastic_state, fortran_mod, name, py_method):
        m, state, state_n, dlambda = plastic_state
        py = getattr(m, py_method)(state, dlambda, state_n)
        f = fortran_mod.call(
            name, m.E, m.nu, m.Ea, m.xi, float(state["eps_eq"]),
            state["stress"], state["theta"], state["beta"], dlambda,
        )
        np.testing.assert_allclose(py, f, rtol=1e-11, atol=1e-11)

    def test_ft_fs_and_fb(self, plastic_state, fortran_mod):
        m, state, state_n, dlambda = plastic_state
        args = (m.B, m.Y, m.C_1, m.C_2, float(state["R"]),
                float(state_n["theta_max"]), dlambda)
        np.testing.assert_allclose(
            m.calc_ft_fs(state, dlambda, state_n),
            fortran_mod.call("yu_ps_ft_fs", *args), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            m.calc_ft_fb(state, dlambda, state_n),
            fortran_mod.call("yu_ps_ft_fb", *args), rtol=1e-12, atol=1e-12)

    def test_ft_ft(self, plastic_state, fortran_mod):
        m, state, state_n, dlambda = plastic_state
        np.testing.assert_allclose(
            m.calc_ft_ft(state, dlambda, state_n),
            fortran_mod.call("yu_ps_ft_ft", m.B, m.Y, m.C_1, m.C_2,
                             float(state["R"]), state["theta"],
                             float(state_n["theta_max"]), dlambda),
            rtol=1e-11, atol=1e-11,
        )

    def test_ft_fl(self, plastic_state, fortran_mod):
        """Carries the stagnation gate on da/dlambda -- the block that was wrong."""
        m, state, state_n, dlambda = plastic_state
        np.testing.assert_allclose(
            m.calc_ft_fl(state, dlambda, state_n),
            fortran_mod.call("yu_ps_ft_fl", m.B, m.Y, m.k, m.Rsat, m.C_1, m.C_2,
                             state["stress"], state["theta"], state["beta"],
                             float(state["R"]), float(state_n["R"]),
                             float(state_n["theta_max"]), dlambda),
            rtol=1e-11, atol=1e-11,
        )

    @pytest.mark.parametrize("name,py_method", [
        ("yu_ps_fb_fs", "calc_fb_fs"),
        ("yu_ps_fb_ft", "calc_fb_ft"),
        ("yu_ps_fb_fb", "calc_fb_fb"),
    ])
    def test_fb_matrix_blocks(self, plastic_state, fortran_mod, name, py_method):
        m, state, _, dlambda = plastic_state
        np.testing.assert_allclose(
            getattr(m, py_method)(state, dlambda),
            fortran_mod.call(name, m.Y, m.k, m.b, dlambda), rtol=1e-12, atol=1e-12)

    def test_fb_fl(self, plastic_state, fortran_mod):
        m, state, _, dlambda = plastic_state
        np.testing.assert_allclose(
            m.calc_fb_fl(state, dlambda),
            fortran_mod.call("yu_ps_fb_fl", m.Y, m.k, m.b,
                             state["stress"], state["theta"], state["beta"]),
            rtol=1e-12, atol=1e-12,
        )


# ---------------------------------------------------------------------------
# TestResidualAndJacobian
# ---------------------------------------------------------------------------

class TestResidualAndJacobian:
    """Assembled residual and Jacobian over a whole trajectory, not one point.

    A block can be right in isolation and still be assembled into the wrong
    slot, so these compare the full 10-vector and 10x10 matrix.
    """

    def _steps(self, model):
        integrator = PythonNumericalIntegrator(model)
        stress = np.zeros(3)
        state = _initial_state(model)
        strain_inc = np.array([1.0e-3, -0.3e-3, 0.2e-3])
        for _ in range(15):
            state_n = state
            result = integrator.stress_update(strain_inc, stress, state)
            stress, state = np.asarray(result.stress), dict(result.state)
            if result.is_plastic:
                yield dict(state), dict(state_n), result

    def test_calc_residual(self, model, fortran_mod):
        n = 0
        for state, state_n, result in self._steps(model):
            n += 1
            py = model.calc_residual(state, state_n, result.stress_trial, result.dlambda)
            f = fortran_mod.call(
                "yu_ps_calc_residual", *_props(model),
                state["stress"], state["theta"], state["beta"],
                float(state["R"]), float(state["eps_eq"]),
                state_n["theta"], state_n["beta"], float(state_n["theta_max"]),
                result.stress_trial, result.dlambda,
            )
            np.testing.assert_allclose(py, f, rtol=1e-10, atol=1e-10)
        assert n > 0, "no plastic steps generated"

    def test_calc_jacobian(self, model, fortran_mod):
        n = 0
        for state, state_n, result in self._steps(model):
            n += 1
            py = model.calc_jacobian(state, state_n, result.stress_trial, result.dlambda)
            f = fortran_mod.call(
                "yu_ps_calc_jacobian", *_props(model),
                state["stress"], state["theta"], state["beta"],
                float(state["R"]), float(state["eps_eq"]),
                float(state_n["theta_max"]), float(state_n["R"]), result.dlambda,
            )
            np.testing.assert_allclose(py, f, rtol=1e-10, atol=1e-10)
        assert n > 0, "no plastic steps generated"

    def test_calc_ddsdde(self, model, fortran_mod):
        """The C_n rhs scaling: passing eps_eq_new here is the bug this catches."""
        n = 0
        for state, state_n, result in self._steps(model):
            n += 1
            py = model.calc_ddsdde(state, state_n, result.stress_trial, result.dlambda)
            f = fortran_mod.call(
                "yu_ps_calc_ddsdde", *_props(model),
                state["stress"], state["theta"], state["beta"],
                float(state["R"]), float(state["eps_eq"]),
                float(state_n["theta_max"]), float(state_n["R"]), result.dlambda,
                float(state_n["eps_eq"]),
            )
            np.testing.assert_allclose(py, f, rtol=1e-9, atol=1e-9)
        assert n > 0, "no plastic steps generated"


# ---------------------------------------------------------------------------
# TestReturnMapping
# ---------------------------------------------------------------------------

class TestReturnMapping:
    """The full yu_kinematic_ps entry point against the Python analytical route."""

    def _call_core(self, model, fortran_mod, stress_n, state_n, strain_inc):
        return fortran_mod.call(
            "yu_kinematic_ps", *_props(model),
            stress_n,
            np.asarray(state_n["theta"]), np.asarray(state_n["beta"]),
            float(state_n["R"]), np.asarray(state_n["q"]), float(state_n["r"]),
            float(state_n["eps_eq"]), float(state_n["theta_max"]),
            strain_inc,
        )

    def test_elastic_step(self, model, fortran_mod):
        """Below yield the Fortran must return sigma_trial and the secant C."""
        state_n = _initial_state(model)
        strain_inc = np.array([1.0e-5, 0.0, 0.0])
        integrator = PythonAnalyticalIntegrator(model)
        py = integrator.stress_update(strain_inc, np.zeros(3), state_n)
        assert not py.is_plastic

        out = self._call_core(model, fortran_mod, np.zeros(3), state_n, strain_inc)
        np.testing.assert_allclose(np.asarray(out[0]), np.asarray(py.stress),
                                   rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(np.asarray(out[8]), np.asarray(py.ddsdde),
                                   rtol=1e-11, atol=1e-11)
        assert int(out[10]) == 1

    def test_single_plastic_step(self, model, fortran_mod):
        state_n = _initial_state(model)
        strain_inc = np.array([3.0e-3, -1.0e-3, 0.0])
        integrator = PythonAnalyticalIntegrator(model)
        py = integrator.stress_update(strain_inc, np.zeros(3), state_n)
        assert py.is_plastic

        out = self._call_core(model, fortran_mod, np.zeros(3), state_n, strain_inc)
        assert int(out[10]) == 1, "Fortran return mapping did not converge"
        np.testing.assert_allclose(np.asarray(out[0]), np.asarray(py.stress),
                                   rtol=1e-9, atol=1e-9)
        for idx, key in ((1, "theta"), (2, "beta"), (4, "q")):
            np.testing.assert_allclose(
                np.asarray(out[idx]), np.asarray(py.state[key]),
                rtol=1e-9, atol=1e-9, err_msg=f"state[{key!r}] mismatch")
        for idx, key in ((3, "R"), (5, "r"), (6, "eps_eq"), (7, "theta_max")):
            np.testing.assert_allclose(
                float(out[idx]), float(py.state[key]),
                rtol=1e-9, atol=1e-9, err_msg=f"state[{key!r}] mismatch")
        np.testing.assert_allclose(np.asarray(out[8]), np.asarray(py.ddsdde),
                                   rtol=1e-8, atol=1e-8)

    def test_fail_code_distinguishes_exit_paths(self, model, fortran_mod):
        """fail_code must separate the three non-convergence exits.

        n_iter alone cannot: a mu failure and a singular solve both exit early,
        so a UMAT reading only n_iter reports "internal failure" for two very
        different causes.  sqrt_arg is the discriminator that matters -- when
        negative the mu equation has no real root, which a time-increment
        cutback cannot fix.
        """
        state_n = _initial_state(model)

        # converged: no diagnostics
        out = self._call_core(model, fortran_mod, np.zeros(3), state_n,
                              np.array([1.0e-5, 0.0, 0.0]))
        assert int(out[10]) == 1
        assert int(out[12]) == 0
        np.testing.assert_array_equal(np.asarray(out[13]), np.zeros(6))

        # outer NR exhausted: code 1, and n_iter pinned at the 50-iteration cap
        out = self._call_core(model, fortran_mod, np.zeros(3), state_n,
                              np.array([5.0, -2.0, 1.0]))
        assert int(out[10]) == 0
        assert int(out[12]) == 1
        assert int(out[9]) == 50
        diag = np.asarray(out[13])
        assert np.isfinite(diag).all(), "diagnostics must not be NaN"
        assert diag[5] != 0.0, "dlambda should be recorded"


# ---------------------------------------------------------------------------
# TestCrosscheckTrajectory
# ---------------------------------------------------------------------------

class TestCrosscheckTrajectory:
    """March both implementations forward together and compare every step.

    Errors that are invisible in one step -- a slightly wrong tangent, a
    stagnation flag that turns on one iteration late -- accumulate here.
    """

    @pytest.mark.parametrize("strain_inc", [
        np.array([1.0e-3, -0.3e-3, 0.0]),
        np.array([0.0, 0.0, 1.5e-3]),
        np.array([8.0e-4, 4.0e-4, -6.0e-4]),
    ])
    def test_trajectory(self, model, fortran_mod, strain_inc):
        integrator = PythonAnalyticalIntegrator(model)
        stress_py = np.zeros(3)
        state_py = _initial_state(model)
        stress_f = np.zeros(3)
        state_f = _initial_state(model)
        n_plastic = 0

        for step in range(25):
            res = integrator.stress_update(strain_inc, stress_py, state_py)
            stress_py, state_py = np.asarray(res.stress), dict(res.state)

            out = fortran_mod.call(
                "yu_kinematic_ps", *_props(model),
                stress_f,
                np.asarray(state_f["theta"]), np.asarray(state_f["beta"]),
                float(state_f["R"]), np.asarray(state_f["q"]), float(state_f["r"]),
                float(state_f["eps_eq"]), float(state_f["theta_max"]),
                strain_inc,
            )
            assert int(out[10]) == 1, f"Fortran did not converge at step {step}"
            stress_f = np.asarray(out[0])
            state_f = dict(
                theta=np.asarray(out[1]), beta=np.asarray(out[2]), R=float(out[3]),
                q=np.asarray(out[4]), r=float(out[5]), eps_eq=float(out[6]),
                theta_max=float(out[7]),
            )
            if res.is_plastic:
                n_plastic += 1

            np.testing.assert_allclose(
                stress_f, stress_py, rtol=1e-8, atol=1e-8,
                err_msg=f"stress diverged at step {step}")
            np.testing.assert_allclose(
                np.asarray(res.ddsdde), np.asarray(out[8]), rtol=1e-7, atol=1e-7,
                err_msg=f"ddsdde diverged at step {step}")
            for key in ("theta", "beta", "R", "q", "r", "eps_eq", "theta_max"):
                np.testing.assert_allclose(
                    np.asarray(state_f[key]), np.asarray(state_py[key]),
                    rtol=1e-8, atol=1e-8,
                    err_msg=f"state[{key!r}] diverged at step {step}")

        assert n_plastic > 0, "trajectory never yielded"


# ---------------------------------------------------------------------------
# Helpers for umat calls
# ---------------------------------------------------------------------------

def _pack_statev(state):
    """Pack a state dict into STATEV(13) following the Fortran layout."""
    sv = np.zeros(13)
    sv[0:3]  = np.asarray(state["theta"])
    sv[3:6]  = np.asarray(state["beta"])
    sv[6]    = float(state["R"])
    sv[7:10] = np.asarray(state["q"])
    sv[10]   = float(state["r"])
    sv[11]   = float(state["eps_eq"])
    sv[12]   = float(state["theta_max"])
    return sv


def _unpack_statev(sv):
    return {
        "theta":     sv[0:3].copy(),
        "beta":      sv[3:6].copy(),
        "R":         sv[6],
        "q":         sv[7:10].copy(),
        "r":         sv[10],
        "eps_eq":    sv[11],
        "theta_max": sv[12],
    }


def _call_umat(model, stress_n, state_n, strain_inc, pnewdt_in=1.0):
    """Call the ABAQUS umat entry point; returns (stress, state, ddsdde, pnewdt)."""
    import yu_kinematic_ps as fm
    NTENS = 3
    STRESS = np.asarray(stress_n, dtype=np.float64).copy()
    STATEV = _pack_statev(state_n)
    PROPS = np.array(_props(model), dtype=np.float64)
    DSTRAN = np.asarray(strain_inc, dtype=np.float64)
    STRAN = np.zeros(NTENS)
    TIME = np.zeros(2)
    PREDEF = np.zeros(1)
    DPRED = np.zeros(1)
    pnewdt = np.array(pnewdt_in, dtype=np.float64)

    # STRESS and STATEV are intent(inout): modified in place.
    # NDI=2, NSHR=1 -- the plane-stress / conventional-shell signature.
    ret = fm.umat(STRESS, STATEV, STRAN, DSTRAN, TIME, 1.0, 0.0, 0.0,
                  PREDEF, DPRED, b'YU      ', 2, 1,
                  PROPS, np.zeros(3), np.eye(3), pnewdt, 1.0,
                  np.eye(3), np.eye(3), 1, 1, 1, 1, 1, 1)
    return STRESS.copy(), _unpack_statev(STATEV), ret[0], float(pnewdt)


# ---------------------------------------------------------------------------
# TestUmatShim
# ---------------------------------------------------------------------------

class TestUmatShim:
    """Verify the ABAQUS umat shim: STATEV packing and the PNEWDT contract.

    The core is already pinned against Python above; what is new here is the
    13-slot STATEV layout (R at 7 and r at 11, unlike the 3-D file's 13 and 20)
    and the ROTSIG calls, which are the only places the shim can silently
    scramble state.
    """

    def test_umat_statev_roundtrip(self, model):
        """umat STRESS/STATEV match PythonAnalyticalIntegrator through the pack."""
        state_n = _initial_state(model)
        strain_inc = np.array([3.0e-3, -1.0e-3, 0.0])
        py = PythonAnalyticalIntegrator(model).stress_update(
            strain_inc, np.zeros(3), state_n)
        assert py.is_plastic

        stress_u, state_u, _, pnewdt = _call_umat(
            model, np.zeros(3), state_n, strain_inc)

        assert pnewdt == 1.0, "converged step must not request a cutback"
        np.testing.assert_allclose(stress_u, np.asarray(py.stress),
                                   rtol=1e-9, atol=1e-9,
                                   err_msg="umat STRESS != Python stress")
        for key in ("theta", "beta", "R", "q", "r", "eps_eq", "theta_max"):
            np.testing.assert_allclose(
                np.asarray(state_u[key]), np.asarray(py.state[key]),
                rtol=1e-9, atol=1e-9, err_msg=f"umat STATEV {key} != Python state")

    def test_umat_pnewdt_nonconvergence(self, model):
        """A strain increment too large to converge must halve PNEWDT and
        leave STRESS/STATEV untouched so the retry restarts from step start."""
        state_n = _initial_state(model)
        strain_inc = np.array([5.0, -2.0, 1.0])
        stress_u, state_u, ddsdde, pnewdt = _call_umat(
            model, np.zeros(3), state_n, strain_inc)

        assert pnewdt <= 0.5, "non-convergence must request a time-increment cut"
        np.testing.assert_allclose(stress_u, np.zeros(3), atol=0.0,
                                   err_msg="STRESS must be left unchanged")
        for key in ("theta", "beta", "q"):
            np.testing.assert_allclose(np.asarray(state_u[key]), np.zeros(3),
                                       atol=0.0,
                                       err_msg=f"STATEV {key} must be unchanged")
        # Secant stiffness at eps_eq_n, so the global NR still gets a valid matrix
        np.testing.assert_allclose(
            np.asarray(ddsdde), np.asarray(model.elastic_stiffness(state_n)),
            rtol=1e-11, atol=1e-11)
