"""Integration tests for the Python integrator (NR solver).

Covers:
- TestAugmentedResidual: implicit-state (augmented) NR system
- TestPartialImplicit: mixed explicit/implicit state declarations
- TestConvergenceHistory: NR residual history records
- TestDlambdaOverride: self.dlambda(R) user-overridable Δλ residual
- TestSolverNonconverged: raise_on_nonconverged=False behaviour
- TestCaseResultBaseConverged: CaseResult / ComparisonResult converged flags
"""

import autograd.numpy as anp
import autograd
import numpy as np
import pytest

import manforge  # noqa: F401 — enables float64
from manforge.core.material import MaterialModel
from manforge.core.state import Implicit, Explicit, NTENS
from manforge.core.dimension import PLANE_STRAIN, PLANE_STRESS, SOLID_3D
from manforge.models.af_kinematic import AFKinematic, AFKinematic3D, AFKinematicPS
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.models.ow_kinematic import OWKinematic3D
from manforge.simulation.driver import StrainDriver
from manforge.simulation.integrator import (
    PythonIntegrator,
    PythonNumericalIntegrator,
    PythonAnalyticalIntegrator,
)
from manforge.simulation.types import FieldHistory, FieldType
from manforge.verification.tangent import check_tangent
from manforge.verification import (
    CaseResult,
    ComparisonResult,
    CrosscheckStrainDriver,
    generate_strain_history,
)
from tests.fixtures.implicit_models import (
    _AFKinematicImplicit3D, _AFKinematicImplicitPS, _AFKinematicImplicitPE,
)


# ===========================================================================
# TestAugmentedResidual
# — from tests/integration/test_augmented_residual.py
# ===========================================================================

class TestAugmentedResidual:

    @pytest.fixture
    def af_model(self):
        return AFKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0)

    @pytest.fixture
    def implicit_model(self):
        return _AFKinematicImplicit3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0)

    @pytest.fixture
    def implicit_ps_model(self):
        return _AFKinematicImplicitPS(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0)

    # -----------------------------------------------------------------------
    # API detection
    # -----------------------------------------------------------------------

    def test_j2_has_no_implicit_states(self, model):
        assert model.implicit_state_names == []

    def test_af_has_no_implicit_states(self):
        m = AFKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0)
        assert m.implicit_state_names == []

    def test_implicit_af_has_implicit_states(self, implicit_model):
        assert "stress" in implicit_model.implicit_state_names
        assert "alpha" in implicit_model.implicit_state_names
        assert "ep" in implicit_model.implicit_state_names

    def test_implicit_ps_has_implicit_states(self, implicit_ps_model):
        assert "stress" in implicit_ps_model.implicit_state_names
        assert "alpha" in implicit_ps_model.implicit_state_names
        assert "ep" in implicit_ps_model.implicit_state_names

    # -----------------------------------------------------------------------
    # Augmented NR matches explicit NR — 3D
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("deps_vec", [
        [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
        [2e-3, 0.0, 0.0, 2e-3, 0.0, 0.0],
    ])
    def test_augmented_matches_reduced_stress_3d(self, af_model, implicit_model, deps_vec):
        deps = anp.array(deps_vec)
        stress0 = anp.zeros(6)
        state0 = af_model.initial_state()
        stress_exp = PythonIntegrator(af_model).stress_update(deps, stress0, state0).stress
        stress_imp = PythonIntegrator(implicit_model).stress_update(deps, stress0, state0).stress
        np.testing.assert_allclose(
            np.array(stress_imp), np.array(stress_exp), atol=1e-7,
            err_msg=f"Implicit stress differs from explicit for deps={deps_vec}"
        )

    @pytest.mark.parametrize("deps_vec", [
        [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
    ])
    def test_augmented_matches_reduced_state_3d(self, af_model, implicit_model, deps_vec):
        deps = anp.array(deps_vec)
        stress0 = anp.zeros(6)
        state0 = af_model.initial_state()
        state_exp = PythonIntegrator(af_model).stress_update(deps, stress0, state0).state
        state_imp = PythonIntegrator(implicit_model).stress_update(deps, stress0, state0).state
        np.testing.assert_allclose(
            np.array(state_imp["alpha"]), np.array(state_exp["alpha"]), atol=1e-7,
        )
        np.testing.assert_allclose(
            float(state_imp["ep"]), float(state_exp["ep"]), atol=1e-10,
        )

    # -----------------------------------------------------------------------
    # Yield surface consistency — implicit path
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("deps_vec", [
        [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
        [2e-3, 0.0, 0.0, 2e-3, 0.0, 0.0],
    ])
    def test_augmented_yield_consistency_3d(self, implicit_model, deps_vec):
        state0 = implicit_model.initial_state()
        deps = anp.array(deps_vec)
        _r = PythonIntegrator(implicit_model).stress_update(deps, anp.zeros(6), state0)
        f = implicit_model.yield_function(_r.state)
        assert abs(float(f)) < 1e-8, f"Yield not satisfied: f = {float(f):.3e}"

    # -----------------------------------------------------------------------
    # FD tangent verification
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("deps_vec", [
        [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
        [2e-3, 0.0, 0.0, 2e-3, 0.0, 0.0],
    ])
    @pytest.mark.slow
    def test_augmented_fd_tangent_virgin_3d(self, implicit_model, deps_vec):
        state0 = implicit_model.initial_state()
        result = check_tangent(
            PythonIntegrator(implicit_model),
            anp.zeros(6), state0, anp.array(deps_vec),
        )
        assert result.passed, (
            f"FD tangent check failed: max_rel_err = {result.max_rel_err:.3e}"
        )

    @pytest.mark.slow
    def test_augmented_fd_tangent_prestressed_3d(self, implicit_model):
        state0 = implicit_model.initial_state()
        deps1 = np.zeros(6); deps1[0] = 3e-3
        _r1 = PythonIntegrator(implicit_model).stress_update(
            anp.array(deps1), anp.zeros(6), state0
        )
        deps2 = np.zeros(6); deps2[0] = 1e-3
        result = check_tangent(
            PythonIntegrator(implicit_model), _r1.stress, _r1.state, anp.array(deps2)
        )
        assert result.passed, f"Pre-stressed FD tangent failed: {result.max_rel_err:.3e}"

    # -----------------------------------------------------------------------
    # Plane-stress
    # -----------------------------------------------------------------------

    def test_augmented_yield_consistency_plane_stress(self, implicit_ps_model):
        state0 = implicit_ps_model.initial_state()
        deps = anp.array([2e-3, 0.0, 0.0])
        _r = PythonIntegrator(implicit_ps_model).stress_update(deps, anp.zeros(3), state0)
        f = implicit_ps_model.yield_function(_r.state)
        assert abs(float(f)) < 1e-8, f"PS yield not satisfied: f = {float(f):.3e}"

    @pytest.mark.slow
    def test_augmented_fd_tangent_plane_stress(self, implicit_ps_model):
        state0 = implicit_ps_model.initial_state()
        deps = anp.array([2e-3, 0.0, 0.0])
        result = check_tangent(PythonIntegrator(implicit_ps_model), anp.zeros(3), state0, deps)
        assert result.passed, f"PS FD tangent failed: {result.max_rel_err:.3e}"

    @pytest.mark.slow
    def test_augmented_matches_reduced_stress_plane_stress(self, implicit_ps_model):
        deps = anp.array([2e-3, -1e-3, 0.0])
        explicit_model = AFKinematicPS(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0)
        state0 = explicit_model.initial_state()
        stress_exp = PythonIntegrator(explicit_model).stress_update(deps, anp.zeros(3), state0).stress
        stress_imp = PythonIntegrator(implicit_ps_model).stress_update(deps, anp.zeros(3), state0).stress
        np.testing.assert_allclose(np.array(stress_imp), np.array(stress_exp), atol=1e-7)

    # -----------------------------------------------------------------------
    # Elastic step
    # -----------------------------------------------------------------------

    def test_augmented_elastic_step_3d(self, implicit_model):
        state0 = implicit_model.initial_state()
        C = implicit_model.elastic_stiffness()
        deps = anp.array([0.5e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
        _r = PythonIntegrator(implicit_model).stress_update(deps, anp.zeros(6), state0)
        np.testing.assert_allclose(np.array(_r.stress), np.array(C @ deps), rtol=1e-10)
        np.testing.assert_allclose(np.array(_r.ddsdde), np.array(C), rtol=1e-10)
        np.testing.assert_allclose(np.array(_r.state["alpha"]), np.zeros(6), atol=1e-30)
        assert float(_r.state["ep"]) == 0.0

    # -----------------------------------------------------------------------
    # Tangent: explicit vs implicit direct comparison
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("deps_vec", [
        [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
    ])
    def test_augmented_tangent_matches_reduced_3d(self, af_model, implicit_model, deps_vec):
        deps = anp.array(deps_vec)
        state0 = af_model.initial_state()
        ddsdde_exp = PythonIntegrator(af_model).stress_update(deps, anp.zeros(6), state0).ddsdde
        ddsdde_imp = PythonIntegrator(implicit_model).stress_update(deps, anp.zeros(6), state0).ddsdde
        np.testing.assert_allclose(
            np.array(ddsdde_imp), np.array(ddsdde_exp), atol=1e-6,
        )

    # -----------------------------------------------------------------------
    # PLANE_STRAIN
    # -----------------------------------------------------------------------

    def test_implicit_pe_has_implicit_states(self):
        m = _AFKinematicImplicitPE()
        assert "stress" in m.implicit_state_names
        assert "alpha" in m.implicit_state_names
        assert "ep" in m.implicit_state_names

    def test_augmented_yield_consistency_plane_strain(self):
        model = _AFKinematicImplicitPE()
        state0 = model.initial_state()
        deps = anp.array([2e-3, 0.0, 0.0, 0.0])
        _r = PythonIntegrator(model).stress_update(deps, anp.zeros(4), state0)
        f = model.yield_function(_r.state)
        assert abs(float(f)) < 1e-8, f"PE yield not satisfied: f = {float(f):.3e}"

    @pytest.mark.slow
    def test_augmented_fd_tangent_plane_strain(self):
        model = _AFKinematicImplicitPE()
        state0 = model.initial_state()
        deps = anp.array([2e-3, 0.0, 0.0, 0.0])
        result = check_tangent(PythonIntegrator(model), anp.zeros(4), state0, deps)
        assert result.passed, f"PE FD tangent failed: {result.max_rel_err:.3e}"

    def test_augmented_matches_reduced_plane_strain(self):
        deps = anp.array([2e-3, -1e-3, 0.0, 1e-3])
        explicit_model = AFKinematic(
            dimension=PLANE_STRAIN,
            E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0
        )
        implicit_model = _AFKinematicImplicitPE()
        state0 = explicit_model.initial_state()
        _r_exp = PythonIntegrator(explicit_model).stress_update(deps, anp.zeros(4), state0)
        _r_imp = PythonIntegrator(implicit_model).stress_update(deps, anp.zeros(4), state0)
        np.testing.assert_allclose(np.array(_r_imp.stress), np.array(_r_exp.stress), atol=1e-7)
        np.testing.assert_allclose(np.array(_r_imp.ddsdde), np.array(_r_exp.ddsdde), atol=1e-6)


# ===========================================================================
# TestPartialImplicit
# — from tests/integration/test_partial_implicit.py
# ===========================================================================

class _AFAlphaImplicit(AFKinematic3D):
    """AF 3D with only alpha implicit (ep remains explicit)."""

    alpha = Implicit(shape=NTENS, doc="backstress (implicit override)")

    def update_state(self, dlambda, state_new, state_n, *, stress_trial=None, strain_inc=None):
        return [self.ep(state_n["ep"] + dlambda)]

    def state_residual(self, state_new, dlambda, state_n, *, stress_trial, strain_inc=None):
        alpha_n = state_n["alpha"]
        xi = state_new["stress"] - alpha_n
        s_xi = self.dev(xi)
        vm_safe = self.vonmises(xi)
        s_hat = s_xi / vm_safe
        scale = 1.0 + self.gamma * dlambda
        R_alpha = state_new["alpha"] * scale - alpha_n - self.C_k * dlambda * s_hat
        return [self.alpha(R_alpha)]


class _AFAlphaImplicitStress(AFKinematic3D):
    """AF 3D with alpha implicit and σ as NR unknown."""

    stress = Implicit(shape=NTENS, doc="Cauchy stress (implicit override)")
    alpha = Implicit(shape=NTENS, doc="backstress (implicit override)")

    def update_state(self, dlambda, state_new, state_n, *, stress_trial=None, strain_inc=None):
        return [self.ep(state_n["ep"] + dlambda)]

    def state_residual(self, state_new, dlambda, state_n, *, stress_trial, strain_inc=None):
        alpha_n = state_n["alpha"]
        xi = state_new["stress"] - alpha_n
        s_xi = self.dev(xi)
        vm_safe = self.vonmises(xi)
        s_hat = s_xi / vm_safe
        scale = 1.0 + self.gamma * dlambda
        R_stress = self.default_stress_residual(state_new, dlambda, stress_trial)
        R_alpha = state_new["alpha"] * scale - alpha_n - self.C_k * dlambda * s_hat
        return [self.stress(R_stress), self.alpha(R_alpha)]


_AF_PARAMS = dict(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0)


class TestPartialImplicit:

    @pytest.fixture
    def explicit_model(self):
        return AFKinematic3D(**_AF_PARAMS)

    @pytest.fixture
    def partial_model(self):
        return _AFAlphaImplicit(**_AF_PARAMS)

    @pytest.fixture
    def partial_stress_model(self):
        return _AFAlphaImplicitStress(**_AF_PARAMS)

    # -----------------------------------------------------------------------
    # API checks
    # -----------------------------------------------------------------------

    def test_partial_implicit_api(self):
        m = _AFAlphaImplicit(**_AF_PARAMS)
        assert m.implicit_state_names == ["alpha"]

    def test_partial_implicit_stress_api(self):
        m = _AFAlphaImplicitStress(**_AF_PARAMS)
        assert "alpha" in m.implicit_state_names
        assert "stress" in m.implicit_state_names

    # -----------------------------------------------------------------------
    # Stress and state match
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("deps_vec", [
        [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
    ])
    def test_partial_implicit_stress_matches_explicit(self, explicit_model, partial_model, deps_vec):
        deps = anp.array(deps_vec)
        state0 = explicit_model.initial_state()
        stress_exp = PythonIntegrator(explicit_model).stress_update(deps, anp.zeros(6), state0).stress
        stress_imp = PythonIntegrator(partial_model).stress_update(deps, anp.zeros(6), state0).stress
        np.testing.assert_allclose(
            np.array(stress_imp), np.array(stress_exp), atol=1e-7,
        )

    @pytest.mark.parametrize("deps_vec", [
        [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
    ])
    def test_partial_implicit_state_matches_explicit(self, explicit_model, partial_model, deps_vec):
        deps = anp.array(deps_vec)
        state0 = explicit_model.initial_state()
        state_exp = PythonIntegrator(explicit_model).stress_update(deps, anp.zeros(6), state0).state
        state_imp = PythonIntegrator(partial_model).stress_update(deps, anp.zeros(6), state0).state
        np.testing.assert_allclose(np.array(state_imp["alpha"]), np.array(state_exp["alpha"]), atol=1e-7)
        np.testing.assert_allclose(float(state_imp["ep"]), float(state_exp["ep"]), atol=1e-10)

    @pytest.mark.parametrize("deps_vec", [
        [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0],
    ])
    def test_implicit_stress_flag_matches_no_flag(self, partial_model, partial_stress_model, deps_vec):
        deps = anp.array(deps_vec)
        state0 = partial_model.initial_state()
        r_no_flag = PythonIntegrator(partial_model).stress_update(deps, anp.zeros(6), state0)
        r_flag = PythonIntegrator(partial_stress_model).stress_update(deps, anp.zeros(6), state0)
        np.testing.assert_allclose(np.array(r_flag.stress), np.array(r_no_flag.stress), atol=1e-7)
        np.testing.assert_allclose(
            np.array(r_flag.state["alpha"]), np.array(r_no_flag.state["alpha"]), atol=1e-7
        )

    # -----------------------------------------------------------------------
    # Yield surface consistency
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("deps_vec", [
        [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
    ])
    def test_partial_implicit_yield_consistency(self, partial_model, deps_vec):
        state0 = partial_model.initial_state()
        deps = anp.array(deps_vec)
        _r = PythonIntegrator(partial_model).stress_update(deps, anp.zeros(6), state0)
        state_with_stress = dict(_r.state)
        state_with_stress["stress"] = _r.stress
        f = partial_model.yield_function(state_with_stress)
        assert abs(float(f)) < 1e-8, f"Yield not satisfied: f = {float(f):.3e}"

    # -----------------------------------------------------------------------
    # FD tangent
    # -----------------------------------------------------------------------

    @pytest.mark.slow
    @pytest.mark.parametrize("deps_vec", [
        [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
    ])
    def test_partial_implicit_fd_tangent(self, partial_model, deps_vec):
        state0 = partial_model.initial_state()
        result = check_tangent(
            PythonIntegrator(partial_model), anp.zeros(6), state0, anp.array(deps_vec),
        )
        assert result.passed, f"FD tangent check failed: {result.max_rel_err:.3e}"

    @pytest.mark.slow
    @pytest.mark.parametrize("deps_vec", [
        [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    def test_partial_implicit_stress_fd_tangent(self, partial_stress_model, deps_vec):
        state0 = partial_stress_model.initial_state()
        result = check_tangent(
            PythonIntegrator(partial_stress_model), anp.zeros(6), state0, anp.array(deps_vec),
        )
        assert result.passed, f"FD tangent check failed (implicit stress): {result.max_rel_err:.3e}"

    # -----------------------------------------------------------------------
    # Tangent agreement between paths
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("deps_vec", [
        [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0],
    ])
    def test_partial_implicit_tangent_matches_explicit(self, explicit_model, partial_model, deps_vec):
        deps = anp.array(deps_vec)
        state0 = explicit_model.initial_state()
        ddsdde_exp = PythonIntegrator(explicit_model).stress_update(deps, anp.zeros(6), state0).ddsdde
        ddsdde_imp = PythonIntegrator(partial_model).stress_update(deps, anp.zeros(6), state0).ddsdde
        np.testing.assert_allclose(np.array(ddsdde_imp), np.array(ddsdde_exp), atol=1e-6)


# ===========================================================================
# TestConvergenceHistory
# — from tests/integration/test_convergence_history.py
# ===========================================================================

class TestConvergenceHistory:

    @pytest.fixture
    def j2_model(self):
        return J2Isotropic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)

    @pytest.fixture
    def ow_model(self):
        return OWKinematic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, C_k=50_000.0, gamma=500.0)

    def test_elastic_step_no_history(self, j2_model):
        deps = anp.array([1e-4, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = PythonIntegrator(j2_model).stress_update(deps, anp.zeros(6), j2_model.initial_state())
        assert result.is_plastic is False
        assert result.n_iterations == 0
        assert result.residual_history == []

    def test_analytical_path_history(self, j2_model):
        deps = anp.array([3e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = PythonAnalyticalIntegrator(j2_model).stress_update(
            deps, anp.zeros(6), j2_model.initial_state()
        )
        assert result.is_plastic is True
        assert result.n_iterations == 1
        assert len(result.residual_history) == 2
        assert result.residual_history[-1] == 0.0

    def test_scalar_nr_records_history(self, j2_model):
        deps = anp.array([3e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = PythonNumericalIntegrator(j2_model).stress_update(
            deps, anp.zeros(6), j2_model.initial_state()
        )
        assert result.is_plastic is True
        assert result.n_iterations >= 1
        assert len(result.residual_history) == result.n_iterations + 1
        assert result.residual_history[-1] < 1e-10

    def test_j2_linear_converges_in_one_step(self, j2_model):
        deps = anp.array([3e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = PythonNumericalIntegrator(j2_model).stress_update(
            deps, anp.zeros(6), j2_model.initial_state()
        )
        assert result.n_iterations == 1

    def test_scalar_nr_residual_decreasing(self, j2_model):
        deps = anp.array([3e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = PythonNumericalIntegrator(j2_model).stress_update(
            deps, anp.zeros(6), j2_model.initial_state()
        )
        history = result.residual_history
        for i in range(1, len(history)):
            assert history[i] < history[i - 1]

    def test_augmented_nr_records_history(self, ow_model):
        deps = np.zeros(6); deps[0] = 3e-3
        result = PythonIntegrator(ow_model).stress_update(
            anp.array(deps), anp.zeros(6), ow_model.initial_state()
        )
        assert result.is_plastic is True
        assert result.n_iterations >= 1
        assert len(result.residual_history) == result.n_iterations + 1
        assert result.residual_history[-1] < 1e-10

    def test_augmented_nr_residual_eventually_decreasing(self, ow_model):
        deps = np.zeros(6); deps[0] = 3e-3
        result = PythonIntegrator(ow_model).stress_update(
            anp.array(deps), anp.zeros(6), ow_model.initial_state()
        )
        history = result.residual_history
        assert history[-1] < history[0]

    def test_augmented_nr_quadratic_convergence(self, ow_model):
        deps = np.zeros(6); deps[0] = 3e-3
        result = PythonIntegrator(ow_model).stress_update(
            anp.array(deps), anp.zeros(6), ow_model.initial_state()
        )
        history = result.residual_history
        if len(history) >= 3:
            import math
            orders = []
            for i in range(1, len(history) - 1):
                e_prev, e_curr, e_next = history[i - 1], history[i], history[i + 1]
                if e_prev > 0 and e_curr > 0 and e_next > 0:
                    den = math.log(e_curr / e_prev)
                    if abs(den) > 1e-30:
                        orders.append(math.log(e_next / e_curr) / den)
            if orders:
                assert orders[-1] > 1.5, (
                    f"Expected near-quadratic convergence, got order {orders[-1]:.2f}. "
                    f"residual_history = {history}"
                )

    def test_driver_j2_analytical_history(self, j2_model):
        load = FieldHistory(FieldType.STRAIN, "Strain", np.linspace(0.0, 5e-3, 20))
        dr = StrainDriver(PythonIntegrator(j2_model)).run(load)
        for rm in dr.step_results:
            if rm.is_plastic:
                assert rm.n_iterations == 1
                assert len(rm.residual_history) == 2
                assert rm.residual_history[-1] == 0.0
            else:
                assert rm.n_iterations == 0
                assert rm.residual_history == []

    def test_driver_j2_autodiff_has_history(self, j2_model):
        load = FieldHistory(FieldType.STRAIN, "Strain", np.linspace(0.0, 5e-3, 20))
        dr = StrainDriver(PythonNumericalIntegrator(j2_model)).run(load)
        for rm in dr.step_results:
            if rm.is_plastic:
                assert rm.n_iterations >= 1
                assert len(rm.residual_history) == rm.n_iterations + 1
            else:
                assert rm.n_iterations == 0
                assert rm.residual_history == []

    @pytest.mark.slow
    def test_driver_ow_step_results_have_history(self, ow_model):
        load = FieldHistory(FieldType.STRAIN, "Strain", np.linspace(0.0, 5e-3, 20))
        dr = StrainDriver(PythonIntegrator(ow_model)).run(load)
        for rm in dr.step_results:
            if rm.is_plastic:
                assert rm.n_iterations >= 1
                assert len(rm.residual_history) == rm.n_iterations + 1
            else:
                assert rm.n_iterations == 0
                assert rm.residual_history == []


# ===========================================================================
# TestDlambdaOverride
# — from tests/integration/test_dlambda_override.py
# ===========================================================================

_E = 210000.0
_NU = 0.3
_SIGMA_Y0 = 250.0
_H_K = 5000.0
_LAM = _E * _NU / ((1.0 + _NU) * (1.0 - 2.0 * _NU))
_MU = _E / (2.0 * (1.0 + _NU))


class _J2ScalarWithDlambdaOverride(MaterialModel):
    """J2 without hardening; state_residual returns self.dlambda(yield_function)."""
    param_names = ["E", "nu", "sigma_y0"]
    stress = Explicit(shape=NTENS, doc="stress")
    ep = Explicit(shape=(), doc="ep")

    def __init__(self, *, E, nu, sigma_y0):
        super().__init__()
        self.E = E; self.nu = nu; self.sigma_y0 = sigma_y0

    def elastic_stiffness(self, state=None):
        return self.isotropic_C(_LAM, _MU)

    def yield_function(self, state):
        return self.vonmises(state["stress"]) - self.sigma_y0

    def update_state(self, dlambda, state_new, state_n, *, stress_trial=None, strain_inc=None):
        return [self.ep(state_n["ep"] + dlambda)]

    def state_residual(self, state_new, dlambda, state_n, *, stress_trial, strain_inc=None):
        return [self.dlambda(self.yield_function(state_new))]


class _J2ScalarDefault(MaterialModel):
    """Same model without dlambda override."""
    param_names = ["E", "nu", "sigma_y0"]
    stress = Explicit(shape=NTENS, doc="stress")
    ep = Explicit(shape=(), doc="ep")

    def __init__(self, *, E, nu, sigma_y0):
        super().__init__()
        self.E = E; self.nu = nu; self.sigma_y0 = sigma_y0

    def elastic_stiffness(self, state=None):
        return self.isotropic_C(_LAM, _MU)

    def yield_function(self, state):
        return self.vonmises(state["stress"]) - self.sigma_y0

    def update_state(self, dlambda, state_new, state_n, *, stress_trial=None, strain_inc=None):
        return [self.ep(state_n["ep"] + dlambda)]


class _KinematicWithDlambdaOverride(MaterialModel):
    """Kinematic hardening (AF) with Perzyna-like dlambda override (c=0 → standard)."""
    param_names = ["E", "nu", "sigma_y0", "H_k", "c"]
    alpha = Implicit(shape=NTENS, doc="backstress")

    def __init__(self, *, E, nu, sigma_y0, H_k, c=0.0):
        super().__init__()
        self.E = E; self.nu = nu; self.sigma_y0 = sigma_y0
        self.H_k = H_k; self.c = c

    def elastic_stiffness(self, state=None):
        return self.isotropic_C(_LAM, _MU)

    def yield_function(self, state):
        xi = state["stress"] - state["alpha"]
        return self.vonmises(xi) - self.sigma_y0

    def state_residual(self, state_new, dlambda, state_n, *, stress_trial, strain_inc=None):
        from manforge.core.state import _state_with_stress
        n = autograd.grad(
            lambda s: self.yield_function(_state_with_stress(state_new, s))
        )(state_new["stress"])
        xi_new = state_new["stress"] - state_new["alpha"]
        f_new = self.vonmises(xi_new) - self.sigma_y0
        R_alpha = (state_new["alpha"] - state_n["alpha"]
                   - dlambda * (self.H_k * n - 0.0 * state_n["alpha"]))
        R_dl = f_new - self.c * dlambda
        return [self.alpha(R_alpha), self.dlambda(R_dl)]


class _KinematicDefault(MaterialModel):
    """Same AF model without dlambda override."""
    param_names = ["E", "nu", "sigma_y0", "H_k"]
    alpha = Implicit(shape=NTENS, doc="backstress")

    def __init__(self, *, E, nu, sigma_y0, H_k):
        super().__init__()
        self.E = E; self.nu = nu; self.sigma_y0 = sigma_y0; self.H_k = H_k

    def elastic_stiffness(self, state=None):
        return self.isotropic_C(_LAM, _MU)

    def yield_function(self, state):
        xi = state["stress"] - state["alpha"]
        return self.vonmises(xi) - self.sigma_y0

    def state_residual(self, state_new, dlambda, state_n, *, stress_trial, strain_inc=None):
        from manforge.core.state import _state_with_stress
        n = autograd.grad(
            lambda s: self.yield_function(_state_with_stress(state_new, s))
        )(state_new["stress"])
        R_alpha = (state_new["alpha"] - state_n["alpha"]
                   - dlambda * (self.H_k * n - 0.0 * state_n["alpha"]))
        return [self.alpha(R_alpha)]


def _plastic_deps():
    return anp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])


class TestDlambdaOverride:

    @pytest.fixture
    def scalar_with_override(self):
        return _J2ScalarWithDlambdaOverride(E=_E, nu=_NU, sigma_y0=_SIGMA_Y0)

    @pytest.fixture
    def scalar_default(self):
        return _J2ScalarDefault(E=_E, nu=_NU, sigma_y0=_SIGMA_Y0)

    @pytest.fixture
    def kin_with_override(self):
        return _KinematicWithDlambdaOverride(E=_E, nu=_NU, sigma_y0=_SIGMA_Y0, H_k=_H_K, c=0.0)

    @pytest.fixture
    def kin_default(self):
        return _KinematicDefault(E=_E, nu=_NU, sigma_y0=_SIGMA_Y0, H_k=_H_K)

    def _su(self, model, deps, stress_n=None, state_n=None):
        if stress_n is None:
            stress_n = anp.zeros(6)
        if state_n is None:
            state_n = model.initial_state()
        return PythonIntegrator(model).stress_update(deps, stress_n, state_n)

    def test_scalar_nr_dlambda_override_matches_default(self, scalar_with_override, scalar_default):
        deps = _plastic_deps()
        r_ovr = self._su(scalar_with_override, deps)
        r_def = self._su(scalar_default, deps)
        np.testing.assert_allclose(np.asarray(r_ovr.stress), np.asarray(r_def.stress), atol=1e-10)
        np.testing.assert_allclose(float(r_ovr.dlambda), float(r_def.dlambda), atol=1e-10)
        np.testing.assert_allclose(float(r_ovr.state["ep"]), float(r_def.state["ep"]), atol=1e-10)

    def test_scalar_nr_dlambda_override_is_plastic(self, scalar_with_override):
        r = self._su(scalar_with_override, _plastic_deps())
        assert r.is_plastic
        assert float(r.dlambda) > 0.0

    def test_vector_nr_dlambda_override_c0_matches_default(self, kin_with_override, kin_default):
        deps = _plastic_deps()
        r_ovr = self._su(kin_with_override, deps)
        r_def = self._su(kin_default, deps)
        np.testing.assert_allclose(np.asarray(r_ovr.stress), np.asarray(r_def.stress), atol=1e-8)
        np.testing.assert_allclose(float(r_ovr.dlambda), float(r_def.dlambda), atol=1e-8)
        np.testing.assert_allclose(
            np.asarray(r_ovr.state["alpha"]), np.asarray(r_def.state["alpha"]), atol=1e-8
        )

    def test_vector_nr_dlambda_override_converges(self, kin_with_override):
        r = self._su(kin_with_override, _plastic_deps())
        assert r.is_plastic
        assert r.return_mapping is not None
        assert r.return_mapping.n_iterations < 50
        hist = r.return_mapping.residual_history
        assert hist[-1] < 1e-8, f"NR residual did not converge: {hist[-1]:.3e}"

    def test_vector_nr_omit_dlambda_fallback_matches_default(self, kin_with_override, kin_default):
        deps = _plastic_deps()
        r_ovr_c0 = self._su(kin_with_override, deps)
        r_def = self._su(kin_default, deps)
        np.testing.assert_allclose(np.asarray(r_ovr_c0.stress), np.asarray(r_def.stress), atol=1e-8)

    def test_scalar_nr_tangent_is_finite(self, scalar_with_override):
        r = self._su(scalar_with_override, _plastic_deps())
        assert r.is_plastic
        ddsdde = np.asarray(r.ddsdde)
        assert np.all(np.isfinite(ddsdde))
        assert np.linalg.norm(ddsdde) > 0.0

    def test_vector_nr_tangent_is_finite(self, kin_with_override):
        r = self._su(kin_with_override, _plastic_deps())
        assert r.is_plastic
        ddsdde = np.asarray(r.ddsdde)
        assert np.all(np.isfinite(ddsdde))
        assert np.linalg.norm(ddsdde) > 0.0


# ===========================================================================
# TestSolverNonconverged
# — from tests/unit/test_solver_nonconverged.py
# ===========================================================================

class TestSolverNonconverged:

    @pytest.fixture
    def j2(self):
        return J2Isotropic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)

    def _plastic_trial(self, j2):
        state_n = j2.initial_state()
        stress_trial = anp.array([500.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return stress_trial, state_n

    def test_default_raises(self, j2):
        st, s_n = self._plastic_trial(j2)
        with pytest.raises(RuntimeError, match="did not converge"):
            PythonNumericalIntegrator(j2, max_iter=0).return_mapping(st, s_n)

    def test_false_returns_nonconverged_result(self, j2):
        st, s_n = self._plastic_trial(j2)
        result = PythonNumericalIntegrator(
            j2, max_iter=0, raise_on_nonconverged=False
        ).return_mapping(st, s_n)
        assert result.converged is False
        assert result.n_iterations == 0
        assert isinstance(result.residual_history, list)

    def test_converged_case_has_flag_true(self, j2):
        st, s_n = self._plastic_trial(j2)
        result = PythonNumericalIntegrator(j2).return_mapping(st, s_n)
        assert result.converged is True

    def test_stress_update_elastic_always_converged(self, j2):
        deps = anp.array([1e-6, 0, 0, 0, 0, 0])
        r = PythonIntegrator(j2).stress_update(deps, anp.zeros(6), j2.initial_state())
        assert r.converged is True
        assert r.is_plastic is False

    def test_stress_update_forwards_flag(self, j2):
        deps = anp.array([3e-3, 0, 0, 0, 0, 0])
        r = PythonNumericalIntegrator(
            j2, max_iter=0, raise_on_nonconverged=False
        ).stress_update(deps, anp.zeros(6), j2.initial_state())
        assert r.is_plastic is True
        assert r.converged is False


# ===========================================================================
# TestCaseResultBaseConverged
# — from tests/unit/test_case_result_base_converged.py
# ===========================================================================

class TestCaseResultBaseConverged:

    @pytest.fixture
    def j2_model(self):
        return J2Isotropic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)

    def test_default_converged_true(self):
        cr = CaseResult(index=0)
        assert cr.a_converged is True
        assert cr.b_converged is True

    def test_can_set_false(self):
        cr = CaseResult(index=0, a_converged=False, b_converged=False)
        assert cr.a_converged is False
        assert cr.b_converged is False

    def test_default_counters_zero(self):
        cr = ComparisonResult(passed=True, n_cases=0, n_passed=0)
        assert cr.n_a_nonconverged == 0
        assert cr.n_b_nonconverged == 0

    def test_stress_update_crosscheck_all_converged(self, j2_model):
        py_a = PythonNumericalIntegrator(j2_model)
        py_b = PythonAnalyticalIntegrator(j2_model)
        cc = CrosscheckStrainDriver(py_a, py_b)
        history = generate_strain_history(j2_model)
        load = FieldHistory(FieldType.STRAIN, "eps", history)
        result = cc.run(load)
        assert result.n_a_nonconverged == 0
        assert result.n_b_nonconverged == 0

    def test_base_run_counts_nonconverged_via_stub(self):
        from manforge.verification.comparator_base import Comparator

        class _Stub(Comparator):
            def iter_run(self):
                yield CaseResult(index=0, passed=True, a_converged=False, b_converged=True)
                yield CaseResult(index=1, passed=True, a_converged=True, b_converged=False)
                yield CaseResult(index=2, passed=True, a_converged=False, b_converged=False)

        result = _Stub().run()
        assert result.n_a_nonconverged == 2
        assert result.n_b_nonconverged == 2
        assert result.n_cases == 3
