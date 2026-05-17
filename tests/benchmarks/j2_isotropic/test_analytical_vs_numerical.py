"""Path A benchmark: analytical return mapping vs framework Newton-Raphson.

Verifies that PythonAnalyticalIntegrator (closed-form) and
PythonNumericalIntegrator (autodiff NR) produce identical results for
J2Isotropic3D and J2Isotropic1D across a variety of strain histories.

J2IsotropicPS is excluded — it has no closed-form return mapping.

Tolerance policy (see benchmarks/README.md):
  stress  : atol=1e-6
  Δλ, ep  : atol=1e-10
  tangent : max_rel_err < 1e-5
"""

import numpy as np
import autograd.numpy as anp
import pytest

from manforge.models.j2_isotropic import J2Isotropic3D, J2Isotropic1D
from manforge.simulation.integrator import (
    PythonIntegrator,
    PythonNumericalIntegrator,
    PythonAnalyticalIntegrator,
)
from manforge.simulation.types import FieldHistory, FieldType
from manforge.simulation.driver import StrainDriver
from manforge.verification.tangent import check_tangent


# ---------------------------------------------------------------------------
# Helper: step through a cumulative strain history, compare per-step
# ---------------------------------------------------------------------------

def _run_and_compare(model, strain_history, *, check_tangent_flag=True):
    """Step through strain_history with both integrators; return max errors."""
    numerical = PythonNumericalIntegrator(model)
    analytical = PythonAnalyticalIntegrator(model)

    stress_n = np.zeros(model.ntens)
    state_n = model.initial_state()
    eps_prev = np.zeros(model.ntens)

    max_stress_err = 0.0
    max_ep_err = 0.0
    max_dlambda_err = 0.0
    max_tangent_rel_err = 0.0

    for i, eps in enumerate(strain_history):
        deps = eps - eps_prev
        eps_prev = eps.copy()

        r_num = numerical.stress_update(anp.array(deps), anp.array(stress_n), state_n)
        r_an = analytical.stress_update(anp.array(deps), anp.array(stress_n), state_n)

        s_num = np.asarray(r_num.stress)
        s_an = np.asarray(r_an.stress)
        stress_err = float(np.max(np.abs(s_an - s_num)))
        max_stress_err = max(max_stress_err, stress_err)

        ep_err = abs(float(r_an.state["ep"]) - float(r_num.state["ep"]))
        max_ep_err = max(max_ep_err, ep_err)

        if r_num.is_plastic:
            dl_err = abs(float(r_an.dlambda or 0.0) - float(r_num.dlambda or 0.0))
            max_dlambda_err = max(max_dlambda_err, dl_err)

            if check_tangent_flag:
                D_num = np.asarray(r_num.ddsdde)
                D_an = np.asarray(r_an.ddsdde)
                rel = np.abs(D_an - D_num) / (np.abs(D_num) + 1.0)
                max_tangent_rel_err = max(max_tangent_rel_err, float(np.max(rel)))

        # advance state from numerical integrator (ground truth trajectory)
        stress_n = np.asarray(r_num.stress)
        state_n = r_num.state

    return {
        "stress": max_stress_err,
        "ep": max_ep_err,
        "dlambda": max_dlambda_err,
        "tangent": max_tangent_rel_err,
    }


# ---------------------------------------------------------------------------
# Multi-step trajectory benchmark
# ---------------------------------------------------------------------------

def test_analytical_matches_numerical_trajectory(j2_scenario):
    """Analytical and numerical integrators agree over full strain history."""
    model, strain_history = j2_scenario
    errs = _run_and_compare(model, strain_history, check_tangent_flag=True)

    assert errs["stress"] < 1e-6, \
        f"max stress error = {errs['stress']:.3e}"
    assert errs["ep"] < 1e-10, \
        f"max ep error = {errs['ep']:.3e}"
    assert errs["dlambda"] < 1e-10, \
        f"max Δλ error = {errs['dlambda']:.3e}"
    assert errs["tangent"] < 1e-5, \
        f"max tangent rel err = {errs['tangent']:.3e}"


# ---------------------------------------------------------------------------
# Elastic regime: stress = C @ deps, tangent = C, ep unchanged
# ---------------------------------------------------------------------------

class TestElasticRegime:

    def test_elastic_stress_equals_C_deps(self):
        model = J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
        C = model.elastic_stiffness()
        deps = anp.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])
        state0 = model.initial_state()

        r = PythonIntegrator(model).stress_update(deps, anp.zeros(6), state0)
        np.testing.assert_allclose(np.asarray(r.stress), np.asarray(C @ deps), rtol=1e-10)

    def test_elastic_tangent_equals_C(self):
        model = J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
        C = model.elastic_stiffness()
        deps = anp.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])

        r = PythonIntegrator(model).stress_update(deps, anp.zeros(6), model.initial_state())
        np.testing.assert_allclose(np.asarray(r.ddsdde), np.asarray(C), rtol=1e-10)

    def test_elastic_state_unchanged(self):
        model = J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
        deps = anp.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])
        state0 = model.initial_state()

        r = PythonIntegrator(model).stress_update(deps, anp.zeros(6), state0)
        np.testing.assert_allclose(np.asarray(r.state["ep"]), np.asarray(state0["ep"]))

    def test_elastic_multiaxial(self):
        model = J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
        C = model.elastic_stiffness()
        deps = anp.array([5e-6, 3e-6, -2e-6, 1e-6, 0.0, 0.0])

        r = PythonIntegrator(model).stress_update(deps, anp.zeros(6), model.initial_state())
        np.testing.assert_allclose(np.asarray(r.stress), np.asarray(C @ deps), rtol=1e-10)
        np.testing.assert_allclose(np.asarray(r.ddsdde), np.asarray(C), rtol=1e-10)

    def test_elastic_from_prestress(self):
        model = J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
        C = model.elastic_stiffness()
        stress_n = anp.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        deps = anp.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])

        r = PythonIntegrator(model).stress_update(deps, stress_n, model.initial_state())
        np.testing.assert_allclose(
            np.asarray(r.stress), np.asarray(stress_n + C @ deps), rtol=1e-10
        )
        np.testing.assert_allclose(np.asarray(r.ddsdde), np.asarray(C), rtol=1e-10)


# ---------------------------------------------------------------------------
# Plastic regime: analytical formula (uniaxial strain increment)
# ---------------------------------------------------------------------------

class TestPlasticRegimeFormula:
    """Verify numerical solution against hand-derived Δλ formula.

    For uniaxial strain increment deps = [deps11, 0, ...]:
      stress_trial = C @ deps  (triaxial)
      sigma_vm_trial = 2μ * deps11
      Δλ = (sigma_vm_trial - sigma_y0) / (3μ + H)
      σ11_new = (λ+2μ)*deps11 - 2μ*Δλ
    """

    def test_plastic_stress_uniaxial_formula(self):
        model = J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
        E, nu = model.E, model.nu
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        sigma_y0, H = model.sigma_y0, model.H

        deps11 = 2e-3
        deps = anp.array([deps11, 0.0, 0.0, 0.0, 0.0, 0.0])

        r = PythonIntegrator(model).stress_update(deps, anp.zeros(6), model.initial_state())

        sigma_vm_trial = 2.0 * mu * deps11
        dlambda_analytic = (sigma_vm_trial - sigma_y0) / (3.0 * mu + H)
        sigma11_analytic = (lam + 2.0 * mu) * deps11 - 2.0 * mu * dlambda_analytic

        assert dlambda_analytic > 0.0
        np.testing.assert_allclose(float(r.stress[0]), sigma11_analytic, rtol=1e-6)

    def test_plastic_ep_matches_dlambda_formula(self):
        model = J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
        E, nu = model.E, model.nu
        mu = E / (2.0 * (1.0 + nu))
        sigma_y0, H = model.sigma_y0, model.H

        deps11 = 2e-3
        sigma_vm_trial = 2.0 * mu * deps11
        dlambda_analytic = (sigma_vm_trial - sigma_y0) / (3.0 * mu + H)

        deps = anp.array([deps11, 0.0, 0.0, 0.0, 0.0, 0.0])
        r = PythonIntegrator(model).stress_update(deps, anp.zeros(6), model.initial_state())

        np.testing.assert_allclose(float(r.state["ep"]), dlambda_analytic, rtol=1e-6)

    def test_plastic_yield_consistency(self):
        model = J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
        deps = anp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])

        r = PythonIntegrator(model).stress_update(deps, anp.zeros(6), model.initial_state())
        f = model.yield_function(r.state)
        assert abs(float(f)) < 1e-8, f"|f| = {abs(float(f)):.3e}"

    def test_plastic_tangent_differs_from_C(self):
        model = J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
        C = model.elastic_stiffness()
        deps = anp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])

        r = PythonIntegrator(model).stress_update(deps, anp.zeros(6), model.initial_state())
        assert not np.allclose(np.asarray(r.ddsdde), np.asarray(C), atol=1.0)

    def test_plastic_ep_positive(self):
        model = J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
        deps = anp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])

        r = PythonIntegrator(model).stress_update(deps, anp.zeros(6), model.initial_state())
        assert float(r.state["ep"]) > 0.0


# ---------------------------------------------------------------------------
# Analytical path selection and FD tangent verification
# ---------------------------------------------------------------------------

class TestAnalyticalPath:

    def test_auto_selects_analytical(self):
        """PythonIntegrator (auto) must yield same result as PythonAnalyticalIntegrator."""
        model = J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
        deps = anp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
        state0 = model.initial_state()

        r_auto = PythonIntegrator(model).stress_update(deps, anp.zeros(6), state0)
        r_an = PythonAnalyticalIntegrator(model).stress_update(deps, anp.zeros(6), state0)

        np.testing.assert_allclose(np.asarray(r_auto.stress), np.asarray(r_an.stress), atol=1e-12)
        np.testing.assert_allclose(np.asarray(r_auto.ddsdde), np.asarray(r_an.ddsdde), atol=1e-12)

    def test_dlambda_positive_for_plastic(self):
        model = J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
        C = model.elastic_stiffness()
        deps = anp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
        rm = model.user_defined_return_mapping(C @ deps, C, model.initial_state())
        assert float(rm.dlambda) > 0.0

    def test_analytical_raises_for_model_without_hooks(self):
        from manforge.core.material import MaterialModel

        class _NoHook(MaterialModel):
            param_names = ["E", "nu", "sigma_y0"]

            def __init__(self):
                super().__init__()
                self.E = 210000.0
                self.nu = 0.3
                self.sigma_y0 = 250.0

            def elastic_stiffness(self, state=None):
                mu = self.E / (2.0 * (1.0 + self.nu))
                lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
                return self.isotropic_C(lam, mu)

            def yield_function(self, state):
                return self.vonmises(state["stress"]) - self.sigma_y0

            def update_state(self, dlambda, state_new, state_n, *, stress_trial=None, strain_inc=None):
                return []

        with pytest.raises(NotImplementedError):
            PythonAnalyticalIntegrator(_NoHook()).stress_update(
                anp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0]),
                anp.zeros(6),
                _NoHook().initial_state(),
            )

    @pytest.mark.parametrize("deps_vec", [
        [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1e-3, -5e-4, -5e-4, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
    ])
    def test_analytical_fd_tangent_plastic(self, deps_vec):
        """Analytical closed-form tangent passes FD verification."""
        model = J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
        result = check_tangent(
            PythonAnalyticalIntegrator(model),
            anp.zeros(6),
            model.initial_state(),
            anp.array(deps_vec),
        )
        assert result.passed, f"FD check failed: max_rel_err = {result.max_rel_err:.3e}"

    def test_analytical_fd_tangent_elastic(self):
        """Elastic step: analytical tangent passes FD check."""
        model = J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
        result = check_tangent(
            PythonAnalyticalIntegrator(model),
            anp.zeros(6),
            model.initial_state(),
            anp.array([1e-4, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        assert result.passed, f"FD check failed: max_rel_err = {result.max_rel_err:.3e}"

    @pytest.mark.parametrize("deps_vec,description", [
        ([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0], "elastic"),
        ([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0], "plastic_uniaxial"),
        ([1.5e-3, -0.5e-3, -0.5e-3, 0.5e-3, 0.0, 0.0], "plastic_multiaxial"),
    ])
    def test_numerical_fd_tangent(self, deps_vec, description):
        """Numerical NR tangent passes FD verification."""
        model = J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
        result = check_tangent(
            PythonNumericalIntegrator(model),
            anp.zeros(6),
            model.initial_state(),
            anp.array(deps_vec),
        )
        assert result.passed, (
            f"[{description}] FD tangent check failed: max_rel_err = {result.max_rel_err:.3e}"
        )

    def test_prestress_two_step(self):
        """Analytical integrator works correctly from a pre-stressed state."""
        model = J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
        deps1 = anp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
        r1 = PythonIntegrator(model).stress_update(deps1, anp.zeros(6), model.initial_state())

        deps2 = anp.array([1e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
        s_num = PythonNumericalIntegrator(model).stress_update(deps2, r1.stress, r1.state).stress
        s_an = PythonAnalyticalIntegrator(model).stress_update(deps2, r1.stress, r1.state).stress

        np.testing.assert_allclose(np.asarray(s_an), np.asarray(s_num), atol=1e-6)

    def test_driver_level_consistency(self):
        """StrainDriver with analytical integrator produces stress consistent with numerical."""
        from manforge.verification.test_cases import generate_strain_history
        from manforge.simulation.types import FieldHistory, FieldType

        model = J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
        strain_data = generate_strain_history(model)
        load = FieldHistory(FieldType.STRAIN, "eps", strain_data)

        r_num = StrainDriver(PythonNumericalIntegrator(model)).run(load)
        r_an = StrainDriver(PythonAnalyticalIntegrator(model)).run(load)

        np.testing.assert_allclose(
            r_an.stress, r_num.stress, atol=1e-6,
            err_msg="Driver-level stress trajectories diverge between analytical and numerical",
        )
