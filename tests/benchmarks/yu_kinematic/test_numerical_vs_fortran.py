"""Path B benchmarks: YUKinematic3D Python NR vs Fortran UMAT.

Requires the compiled yu_kinematic_3d Fortran extension:
    make fortran-build-yu
or:
    uv run manforge build fortran/abaqus_stubs.f90 fortran/yu_kinematic_3d.f90 --name yu_kinematic_3d

Test classes added incrementally per implementation step:
    TestJacobianBlocks       -- Step 1: 16 dRxx_dyy blocks + helpers
    TestResidualAndJacobian  -- Step 2: integrated residual + Jacobian
"""

import numpy as np
import pytest

pytest.importorskip(
    "yu_kinematic_3d",
    reason=(
        "yu_kinematic_3d not compiled -- run: "
        "make fortran-build-yu"
    ),
)

pytestmark = pytest.mark.fortran

from manforge.simulation.integrator import FortranModule, PythonNumericalIntegrator
from manforge.models import YUKinematic3D
from manforge.verification import check_bindings

from .conftest import PARAMS


@pytest.fixture
def fortran_mod():
    return FortranModule("yu_kinematic_3d")


@pytest.fixture
def model():
    return YUKinematic3D(**PARAMS)


# ---------------------------------------------------------------------------
# Helper: build check_bindings cases from a converged plastic step state
# ---------------------------------------------------------------------------

def _build_cases(m, state, state_n, dlambda):
    """Return check_bindings cases dict for all 16 dRxx_dyy blocks."""
    C = m.elastic_stiffness(state)
    stress = state["stress"]
    theta = state["theta"]
    beta = state["beta"]
    xi = m.dev(stress) - theta - beta
    eps_eq = float(state["eps_eq"])
    R = float(state["R"])
    R_n = float(state_n["R"])
    theta_max = float(state["theta_max"])
    dl = float(dlambda)

    return {
        "dRstress_dstress": (
            (C, xi, dl),
            (C, xi, dl),
        ),
        "dRstress_dbeta": (
            (C, xi, dl),
            (C, xi, dl),
        ),
        "dRstress_dtheta": (
            (C, xi, dl),
            (C, xi, dl),
        ),
        "dRstress_dlambda": (
            (C, xi, eps_eq, dl),
            (m.E, m.Ea, m.xi, C, xi, eps_eq, dl),
        ),
        "dRbeta_dstress": (
            (dl,),
            (m.k, m.b, m.Y, dl),
        ),
        "dRbeta_dbeta": (
            (dl,),
            (m.k, m.b, m.Y, dl),
        ),
        "dRbeta_dtheta": (
            (dl,),
            (m.k, m.b, m.Y, dl),
        ),
        "dRbeta_dlambda": (
            (xi, beta, dl),
            (m.k, m.b, m.Y, xi, beta, dl),
        ),
        "dRtheta_dstress": (
            (theta, theta_max, R, R_n, dl),
            (m.B, m.Y, m.k, m.Rsat, m.C_1, m.C_2,
             theta, theta_max, R, R_n, dl),
        ),
        "dRtheta_dbeta": (
            (theta, theta_max, R, R_n, dl),
            (m.B, m.Y, m.k, m.Rsat, m.C_1, m.C_2,
             theta, theta_max, R, R_n, dl),
        ),
        "dRtheta_dtheta": (
            (theta, theta_max, R, R_n, dl),
            (m.B, m.Y, m.k, m.Rsat, m.C_1, m.C_2,
             theta, theta_max, R, R_n, dl),
        ),
        "dRtheta_dlambda": (
            (xi, theta, theta_max, R, R_n, dl),
            (m.B, m.Y, m.k, m.Rsat, m.C_1, m.C_2,
             xi, theta, theta_max, R, R_n, dl),
        ),
        "dRyield_dstress": (
            (xi,),
            (xi,),
        ),
        "dRyield_dbeta": (
            (xi,),
            (xi,),
        ),
        "dRyield_dtheta": (
            (xi,),
            (xi,),
        ),
        "dRyield_dlambda": (
            (),
            (),
        ),
    }


# ---------------------------------------------------------------------------
# TestJacobianBlocks: per-step comparison over full trajectories
# ---------------------------------------------------------------------------

class TestJacobianBlocks:
    """Compare all 16 Jacobian blocks Python vs Fortran over trajectory steps."""

    def test_jacobian_blocks_match_fortran(self, yu_3d_scenario, fortran_mod):
        m, history = yu_3d_scenario
        integrator = PythonNumericalIntegrator(m)

        stress_n = np.zeros(6)
        state_n = m.initial_state()
        n_plastic = 0

        for strain_total_n, strain_total_np1 in zip(history[:-1], history[1:]):
            strain_inc = strain_total_np1 - strain_total_n
            result = integrator.stress_update(strain_inc, stress_n, state_n)

            if result.is_plastic:
                n_plastic += 1
                cases = _build_cases(m, result.state, state_n, result.dlambda)
                res = check_bindings(m, fortran_mod, cases, rtol=1e-10, atol=1e-12)
                for name, (ok, err) in res.items():
                    assert ok, (
                        f"step {n_plastic}: {name}: max_rel_err={err:.3e}"
                    )

            stress_n = result.stress
            state_n = result.state

        if n_plastic == 0:
            pytest.skip("No plastic steps in this scenario — Jacobian blocks not exercised")
