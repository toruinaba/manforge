"""Path B benchmark: Python Newton-Raphson vs compiled Fortran UMAT.

Verifies that PythonNumericalIntegrator and the compiled j2_isotropic_3d
Fortran module produce identical results across elastic, plastic, and
multi-step loading scenarios.

Fortran coverage is limited to J2Isotropic3D — the Fortran implementation
only handles 3D stress states.

All tests are skipped when the compiled module is not available.

Tolerance policy (see benchmarks/README.md):
  stress  : atol=1e-6 (absolute) or max_rel_err < 1e-6
  Δλ, ep  : atol=1e-10 (absolute)
  tangent : max_rel_err < 1e-5
"""

import numpy as np
import autograd.numpy as anp
import pytest

pytest.importorskip(
    "j2_isotropic_3d",
    reason="j2_isotropic_3d not compiled -- run: make fortran-build-umat",
)

pytestmark = pytest.mark.fortran

from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.simulation.integrator import (
    PythonIntegrator,
    PythonNumericalIntegrator,
    PythonAnalyticalIntegrator,
    FortranIntegrator,
)
from manforge.simulation.types import FieldHistory, FieldType
from manforge.verification import FortranModule
from manforge.verification import (
    CrosscheckStrainDriver,
    CrosscheckStressDriver,
    generate_strain_history,
)


# ---------------------------------------------------------------------------
# Shared constants (match root conftest `model` fixture)
# ---------------------------------------------------------------------------

_E = 210000.0
_NU = 0.3
_SIGMA_Y0 = 250.0
_H = 1000.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def j2():
    return J2Isotropic3D(E=_E, nu=_NU, sigma_y0=_SIGMA_Y0, H=_H)


@pytest.fixture
def fortran_mod():
    return FortranModule("j2_isotropic_3d")


@pytest.fixture
def fc_int(fortran_mod, j2):
    return FortranIntegrator.from_model(fortran_mod, "j2_isotropic_3d", j2)


@pytest.fixture
def py_int(j2):
    return PythonNumericalIntegrator(j2)


@pytest.fixture
def py_int_an(j2):
    return PythonAnalyticalIntegrator(j2)


def _j2_load(model):
    history = generate_strain_history(model)
    return FieldHistory(FieldType.STRAIN, "eps", history)


# ---------------------------------------------------------------------------
# Low-level FortranModule.call sanity
# ---------------------------------------------------------------------------

class TestFortranModuleCall:

    def test_call_returns_correct_shapes(self, fortran_mod, j2):
        """j2_isotropic_3d subroutine returns stress (6,) and ddsdde (6,6)."""
        dstran = np.array([1e-4, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        stress_f, ep_f, ddsdde_f = fortran_mod.call(
            "j2_isotropic_3d",
            j2.E, j2.nu, j2.sigma_y0, j2.H,
            np.zeros(6), 0.0, dstran,
        )
        assert np.asarray(stress_f).shape == (6,)
        assert np.asarray(ddsdde_f).shape == (6, 6)

    def test_elastic_stiffness_shape(self, fortran_mod):
        """j2_isotropic_3d_elastic_stiffness returns (6, 6)."""
        C = fortran_mod.call("j2_isotropic_3d_elastic_stiffness", _E, _NU)
        assert np.asarray(C).shape == (6, 6)

    def test_elastic_stiffness_matches_python(self, fortran_mod, j2):
        """Fortran elastic stiffness matches Python to near machine precision."""
        C_f = fortran_mod.call("j2_isotropic_3d_elastic_stiffness", j2.E, j2.nu)
        C_py = j2.elastic_stiffness(j2.initial_state())
        np.testing.assert_allclose(np.array(C_py), np.array(C_f), rtol=1e-12)

    def test_fortran_vs_python_elastic_step(self, fortran_mod, j2):
        """Elastic step: Fortran stress and tangent match Python."""
        dstran = np.array([1e-4, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        stress_f, ep_f, ddsdde_f = fortran_mod.call(
            "j2_isotropic_3d",
            j2.E, j2.nu, j2.sigma_y0, j2.H,
            np.zeros(6), 0.0, dstran,
        )
        r = PythonIntegrator(j2).stress_update(anp.array(dstran), anp.zeros(6), j2.initial_state())
        np.testing.assert_allclose(np.array(r.stress), stress_f, rtol=1e-6)
        np.testing.assert_allclose(np.array(r.ddsdde), np.array(ddsdde_f), rtol=1e-5)

    def test_fortran_vs_python_plastic_uniaxial(self, fortran_mod, j2):
        """Plastic uniaxial step: Fortran stress, ep, and tangent match Python."""
        dstran = np.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        stress_f, ep_f, ddsdde_f = fortran_mod.call(
            "j2_isotropic_3d",
            j2.E, j2.nu, j2.sigma_y0, j2.H,
            np.zeros(6), 0.0, dstran,
        )
        r = PythonIntegrator(j2).stress_update(anp.array(dstran), anp.zeros(6), j2.initial_state())
        np.testing.assert_allclose(np.array(r.stress), stress_f, rtol=1e-6)
        np.testing.assert_allclose(np.array(r.ddsdde), np.array(ddsdde_f), rtol=1e-5)
        assert abs(float(r.state["ep"]) - float(ep_f)) / (abs(float(r.state["ep"])) + 1e-14) < 1e-6

    def test_fortran_vs_python_plastic_multiaxial(self, fortran_mod, j2):
        """Plastic multiaxial step: Fortran matches Python."""
        dstran = np.array([1.5e-3, -0.5e-3, -0.5e-3, 0.5e-3, 0.0, 0.0], dtype=np.float64)
        stress_f, ep_f, ddsdde_f = fortran_mod.call(
            "j2_isotropic_3d",
            j2.E, j2.nu, j2.sigma_y0, j2.H,
            np.zeros(6), 0.0, dstran,
        )
        r = PythonIntegrator(j2).stress_update(anp.array(dstran), anp.zeros(6), j2.initial_state())
        np.testing.assert_allclose(np.array(r.stress), stress_f, rtol=1e-6)
        np.testing.assert_allclose(np.array(r.ddsdde), np.array(ddsdde_f), rtol=1e-5)


# ---------------------------------------------------------------------------
# Multi-step trajectory benchmark (CrosscheckStrainDriver)
# ---------------------------------------------------------------------------

class TestCrosscheckTrajectory:

    def test_numerical_newton_vs_fortran(self, py_int, fc_int, j2):
        """PythonNumericalIntegrator vs UMAT over full strain history."""
        cc = CrosscheckStrainDriver(py_int, fc_int)
        result = cc.run(_j2_load(j2))

        assert result.passed, f"max_stress_rel_err = {result.max_stress_rel_err:.2e}"
        assert result.max_stress_rel_err < 1e-6
        assert result.n_cases == result.n_passed

    def test_analytical_vs_fortran(self, py_int_an, fc_int, j2):
        """PythonAnalyticalIntegrator (closed-form) vs UMAT over full strain history."""
        cc = CrosscheckStrainDriver(py_int_an, fc_int)
        result = cc.run(_j2_load(j2))

        assert result.passed, f"max_stress_rel_err = {result.max_stress_rel_err:.2e}"
        assert result.max_stress_rel_err < 1e-6

    def test_scenario_parametric(self, j2_3d_scenario, fortran_mod):
        """Parametrized scenarios: each strain history passes the Fortran crosscheck."""
        model, strain_history = j2_3d_scenario
        fc_int = FortranIntegrator.from_model(fortran_mod, "j2_isotropic_3d", model)
        py_int = PythonNumericalIntegrator(model)
        load = FieldHistory(FieldType.STRAIN, "eps", strain_history)

        cc = CrosscheckStrainDriver(py_int, fc_int)
        result = cc.run(load)

        assert result.passed, f"max_stress_rel_err = {result.max_stress_rel_err:.2e}"
        assert result.max_stress_rel_err < 1e-6

    def test_multi_step_tension_unload_compression(self, j2, fortran_mod):
        """Manual multi-step loop: tension-unload-compression error stays bounded."""
        eps_y = j2.sigma_y0 / j2.E
        n = 35
        tension = np.linspace(0.0, 3.0 * eps_y, n // 2 + 1)
        compression = np.linspace(tension[-1], -3.0 * eps_y, n - len(tension) + 2)[1:]
        strain_vals = np.concatenate([tension, compression])

        ntens = j2.ntens
        history = np.zeros((len(strain_vals), ntens))
        history[:, 0] = strain_vals

        stress_py = anp.zeros(ntens)
        state_py = j2.initial_state()
        stress_f = np.zeros(ntens)
        ep_f = 0.0
        eps_prev = np.zeros(ntens)
        max_err = 0.0

        for eps in history:
            dstran = eps - eps_prev
            eps_prev = eps.copy()
            r = PythonIntegrator(j2).stress_update(anp.array(dstran), stress_py, state_py)
            stress_py, state_py = r.stress, r.state
            stress_f, ep_f, _ = fortran_mod.call(
                "j2_isotropic_3d",
                j2.E, j2.nu, j2.sigma_y0, j2.H,
                stress_f, ep_f, dstran,
            )
            err = float(np.max(np.abs(np.array(stress_f) - np.array(stress_py))
                               / (np.abs(np.array(stress_py)) + 1.0)))
            max_err = max(max_err, err)

        assert max_err < 1e-6, f"Max stress error {max_err:.2e} exceeds 1e-6"


# ---------------------------------------------------------------------------
# Stress-driven crosscheck
# ---------------------------------------------------------------------------

class TestCrosscheckStressDriven:

    def test_stress_uniaxial_sweep(self, py_int, fc_int, j2):
        """StressDriver: Fortran and Python agree on a uniaxial stress sweep."""
        sigma_max = 1.5 * j2.sigma_y0
        targets = np.array([0.5 * sigma_max, sigma_max, 0.8 * sigma_max, 0.0])
        stress_data = np.zeros((len(targets), j2.ntens))
        stress_data[:, 0] = targets
        load = FieldHistory(FieldType.STRESS, "sigma", stress_data)

        cc = CrosscheckStressDriver(py_int, fc_int)
        result = cc.run(load)

        assert result.passed, f"max_stress_rel_err = {result.max_stress_rel_err:.2e}"


# ---------------------------------------------------------------------------
# iter_run early-break and negative cases
# ---------------------------------------------------------------------------

class TestNegativeCases:

    def test_wrong_param_order_produces_failure(self, fortran_mod, j2):
        """FortranIntegrator with wrong param_fn order must fail the crosscheck."""
        py_int = PythonNumericalIntegrator(j2)
        fc_int_bad = FortranIntegrator.from_model(
            fortran_mod, "j2_isotropic_3d", j2,
            param_fn=lambda: (j2.sigma_y0, j2.H, j2.E, j2.nu),  # deliberately wrong
        )

        cc = CrosscheckStrainDriver(py_int, fc_int_bad)
        result = cc.run(_j2_load(j2))

        assert not result.passed, (
            "Expected failure with wrong param_fn, "
            f"but max_stress_rel_err = {result.max_stress_rel_err:.2e}"
        )

    def test_iter_run_early_break(self, fortran_mod, j2):
        """iter_run allows breaking on first failing step."""
        py_int = PythonNumericalIntegrator(j2)
        fc_int_bad = FortranIntegrator.from_model(
            fortran_mod, "j2_isotropic_3d", j2,
            param_fn=lambda: (j2.sigma_y0, j2.H, j2.E, j2.nu),
        )

        cc = CrosscheckStrainDriver(py_int, fc_int_bad)
        found_failure = False
        for cr in cc.iter_run(_j2_load(j2)):
            if not cr.passed:
                found_failure = True
                break

        assert found_failure, "Expected at least one failing step with wrong param_fn"


# ---------------------------------------------------------------------------
# Single-step parametric crosscheck
# ---------------------------------------------------------------------------

class TestSingleStepCrosscheck:

    def _make_cases(self, model):
        from manforge.verification.test_cases import estimate_yield_strain
        eps_y = estimate_yield_strain(model)
        ntens = model.ntens
        state0 = dict(model.initial_state())
        z = np.zeros(ntens)
        cases = []

        de = np.zeros(ntens); de[0] = 0.5 * eps_y
        cases.append({"deps": de, "stress_n": z.copy(), "state_n": dict(state0)})

        de = np.zeros(ntens); de[0] = 5.0 * eps_y
        cases.append({"deps": de, "stress_n": z.copy(), "state_n": dict(state0)})

        de = np.zeros(ntens); de[0] = 3.0 * eps_y; de[1] = -1.5 * eps_y; de[2] = -1.5 * eps_y
        cases.append({"deps": de, "stress_n": z.copy(), "state_n": dict(state0)})

        de = np.zeros(ntens); de[3] = 3.0 * eps_y
        cases.append({"deps": de, "stress_n": z.copy(), "state_n": dict(state0)})

        # pre-stressed starting point
        pre_deps = np.zeros(ntens); pre_deps[0] = 3.0 * eps_y
        r_pre = PythonIntegrator(model).stress_update(
            pre_deps, np.zeros(ntens), model.initial_state()
        )
        de2 = np.zeros(ntens); de2[0] = 2.0 * eps_y
        cases.append({
            "deps": de2,
            "stress_n": np.array(r_pre.stress),
            "state_n": {k: np.asarray(v) for k, v in r_pre.state.items()},
        })
        return cases

    def _run(self, py_int, fc_int, model):
        cc = CrosscheckStrainDriver(py_int, fc_int)
        failures = []
        for case in self._make_cases(model):
            data = case["deps"][np.newaxis]
            load = FieldHistory(FieldType.STRAIN, "eps", data)
            for cr in cc.iter_run(load, initial_stress=case["stress_n"],
                                  initial_state=case["state_n"]):
                if not cr.passed:
                    failures.append((case, cr))
        return failures

    def test_numerical_newton_single_steps(self, py_int, fc_int, j2):
        """Single-step elastic/plastic/multiaxial cases: numerical NR vs UMAT."""
        failures = self._run(py_int, fc_int, j2)
        assert not failures, (
            f"{len(failures)} case(s) failed: "
            f"max stress_rel_err = {max(cr.stress_rel_err for _, cr in failures):.2e}"
        )

    def test_analytical_single_steps(self, py_int_an, fc_int, j2):
        """Single-step cases with closed-form return mapping vs UMAT."""
        failures = self._run(py_int_an, fc_int, j2)
        assert not failures, (
            f"{len(failures)} case(s) failed: "
            f"max stress_rel_err = {max(cr.stress_rel_err for _, cr in failures):.2e}"
        )
