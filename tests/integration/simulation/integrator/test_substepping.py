"""Integration tests for non-convergence reporting and SubSteppingIntegrator.

Covers:
- TestUserDefinedNonconverged: the ``user_defined`` return-mapping path honours
  ``raise_on_nonconverged`` the same way ``numerical_newton`` does
- TestDelegation: SubSteppingIntegrator forwards the StressIntegrator surface
- TestPassthrough: a converging increment is returned untouched (m=1)
- TestSubdivision: a failing increment is retried at 2, 4, 8, ... substeps
- TestExhaustion: behaviour once max_substeps is used up
- TestTangentApproximation: the subdivided ddsdde is the last substep's
"""

import numpy as np
import pytest

import manforge  # noqa: F401 — enables float64
from manforge.core.result import ReturnMappingResult
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.simulation.driver import StrainDriver
from manforge.simulation.integrator import (
    PythonAnalyticalIntegrator,
    PythonNumericalIntegrator,
    SubSteppingIntegrator,
)
from manforge.simulation.types import FieldHistory, FieldType


PARAMS = dict(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)


class _FailsAboveDlambda(J2Isotropic3D):
    """J2 whose closed-form corrector declares defeat on large increments.

    Stands in for a model that genuinely loses its admissible branch when Δε
    is too large (the plane-stress YU flow rule is the real motivating case).
    The closed-form J2 return itself never fails, so the failure is injected on
    a Δλ threshold — that keeps the trigger a function of increment size, which
    is exactly what subdivision is supposed to fix.
    """

    # Measured against J2(E=210e3, sigma_y0=250, H=1000) at Δε11 = 3e-3:
    # the undivided step needs Δλ = 9.64e-4 and 2 substeps still need 9.64e-4
    # (the first of the two is elastic), while 4 substeps need only 4.98e-4.
    # A limit in between therefore fails at m = 1 and 2 and succeeds at m = 4,
    # which is what pins the doubling schedule in TestSubdivision.
    dlambda_limit = 6.0e-4

    def user_defined_return_mapping(self, stress_trial, C, state_n):
        rm = super().user_defined_return_mapping(stress_trial, C, state_n)
        if float(rm.dlambda) > self.dlambda_limit:
            return ReturnMappingResult(
                stress=rm.stress,
                state=rm.state,
                dlambda=rm.dlambda,
                n_iterations=rm.n_iterations,
                residual_history=rm.residual_history,
                converged=False,
            )
        return rm


def _plastic_deps(scale=1.0):
    return np.array([3.0e-3 * scale, 0.0, 0.0, 0.0, 0.0, 0.0])


def _zero_start(model):
    return np.zeros(model.ntens), model.initial_state()


def _uniaxial_ramp(n_steps, deps11):
    """Monotonic ε11 ramp with a constant increment of *deps11*."""
    return FieldHistory(
        type=FieldType.STRAIN,
        name="Strain",
        data=np.cumsum(np.full(n_steps, deps11)),
    )


# ===========================================================================
# TestUserDefinedNonconverged — stage 1: the two paths share one convention
# ===========================================================================

class TestUserDefinedNonconverged:

    @pytest.fixture
    def model(self):
        return _FailsAboveDlambda(**PARAMS)

    def test_user_defined_path_raises(self, model):
        stress_n, state_n = _zero_start(model)
        with pytest.raises(RuntimeError, match="converged=False"):
            PythonAnalyticalIntegrator(model).stress_update(
                _plastic_deps(), stress_n, state_n
            )

    def test_user_defined_path_returns_flag_when_not_raising(self, model):
        stress_n, state_n = _zero_start(model)
        r = PythonAnalyticalIntegrator(
            model, raise_on_nonconverged=False
        ).stress_update(_plastic_deps(), stress_n, state_n)
        assert r.converged is False
        assert r.is_plastic is True

    def test_converging_increment_is_unaffected(self, model):
        """The gate must not fire on a plastic increment the corrector handles."""
        stress_n, state_n = _zero_start(model)
        r = PythonAnalyticalIntegrator(model).stress_update(
            _plastic_deps(scale=0.7), stress_n, state_n
        )
        assert r.is_plastic is True
        assert r.converged is True

    def test_message_names_the_model_and_the_hook(self, model):
        stress_n, state_n = _zero_start(model)
        with pytest.raises(RuntimeError) as exc:
            PythonAnalyticalIntegrator(model).stress_update(
                _plastic_deps(), stress_n, state_n
            )
        msg = str(exc.value)
        assert "_FailsAboveDlambda" in msg
        assert "user_defined_return_mapping" in msg


# ===========================================================================
# TestDelegation
# ===========================================================================

class TestDelegation:

    @pytest.fixture
    def inner(self):
        return PythonNumericalIntegrator(J2Isotropic3D(**PARAMS))

    @pytest.fixture
    def wrapped(self, inner):
        return SubSteppingIntegrator(inner)

    def test_ntens(self, inner, wrapped):
        assert wrapped.ntens == inner.ntens

    def test_dimension(self, inner, wrapped):
        assert wrapped.dimension is inner.dimension

    def test_initial_state(self, inner, wrapped):
        assert wrapped.initial_state().keys() == inner.initial_state().keys()

    def test_elastic_stiffness(self, inner, wrapped):
        state = inner.initial_state()
        np.testing.assert_allclose(
            np.asarray(wrapped.elastic_stiffness(state)),
            np.asarray(inner.elastic_stiffness(state)),
        )

    def test_absent_elastic_stiffness_is_not_faked(self):
        """StressDriver probes with hasattr, so the wrapper must not add one."""

        class _NoStiffness:
            dimension = J2Isotropic3D(**PARAMS).dimension
            ntens = 6

            def initial_state(self):
                return {}

            def stress_update(self, strain_inc, stress_n, state_n):
                raise AssertionError("not called")

        assert not hasattr(SubSteppingIntegrator(_NoStiffness()), "elastic_stiffness")

    def test_rejects_max_substeps_below_one(self, inner):
        with pytest.raises(ValueError, match="max_substeps"):
            SubSteppingIntegrator(inner, max_substeps=0)


# ===========================================================================
# TestPassthrough — an increment that already worked must not change
# ===========================================================================

class TestPassthrough:

    @pytest.fixture
    def model(self):
        return J2Isotropic3D(**PARAMS)

    @pytest.mark.parametrize("deps", [
        pytest.param(np.array([1.0e-6, 0, 0, 0, 0, 0]), id="elastic"),
        pytest.param(_plastic_deps(), id="plastic"),
    ])
    def test_result_is_the_inner_result_object(self, model, deps):
        inner = PythonNumericalIntegrator(model)
        stress_n, state_n = _zero_start(model)
        r = SubSteppingIntegrator(inner).stress_update(deps, stress_n, state_n)
        ref = inner.stress_update(deps, stress_n, state_n)

        np.testing.assert_array_equal(np.asarray(r.stress), np.asarray(ref.stress))
        np.testing.assert_array_equal(np.asarray(r.ddsdde), np.asarray(ref.ddsdde))
        assert r.n_iterations == ref.n_iterations
        assert r.is_plastic == ref.is_plastic
        assert r.residual_history == ref.residual_history

    def test_n_substeps_is_one(self, model):
        inner = PythonNumericalIntegrator(model)
        stress_n, state_n = _zero_start(model)
        r = SubSteppingIntegrator(inner).stress_update(_plastic_deps(), stress_n, state_n)
        assert r.n_substeps == 1

    def test_elastic_step_keeps_the_elastic_shortcut(self, model):
        """m=1 passthrough must not manufacture a ReturnMappingResult."""
        inner = PythonNumericalIntegrator(model)
        stress_n, state_n = _zero_start(model)
        r = SubSteppingIntegrator(inner).stress_update(
            np.array([1.0e-6, 0, 0, 0, 0, 0]), stress_n, state_n
        )
        assert r.return_mapping is None
        assert r.is_plastic is False

    def test_full_history_is_unchanged_under_the_wrapper(self, model):
        """A driver run that never failed must produce identical trajectories."""
        load = _uniaxial_ramp(40, 2.0e-4)
        inner = PythonNumericalIntegrator(model)

        plain = list(StrainDriver(inner).iter_run(load))
        wrapped = list(StrainDriver(SubSteppingIntegrator(inner)).iter_run(load))

        assert len(plain) == len(wrapped)
        for a, b in zip(plain, wrapped):
            np.testing.assert_array_equal(
                np.asarray(a.result.stress), np.asarray(b.result.stress)
            )
            assert a.result.n_iterations == b.result.n_iterations
            assert b.result.n_substeps == 1


# ===========================================================================
# TestSubdivision
# ===========================================================================

class TestSubdivision:

    @pytest.fixture
    def model(self):
        return _FailsAboveDlambda(**PARAMS)

    @pytest.fixture
    def inner(self, model):
        return PythonAnalyticalIntegrator(model)

    def test_subdivision_rescues_a_failing_increment(self, model, inner):
        stress_n, state_n = _zero_start(model)
        with pytest.raises(RuntimeError):
            inner.stress_update(_plastic_deps(), stress_n, state_n)

        r = SubSteppingIntegrator(inner).stress_update(
            _plastic_deps(), stress_n, state_n
        )
        assert r.converged is True
        assert r.n_substeps > 1

    def test_doubling_stops_at_the_first_success(self, model, inner):
        """Δλ needs 4 substeps here, so it must report 4 and not 8."""
        stress_n, state_n = _zero_start(model)
        r = SubSteppingIntegrator(inner).stress_update(
            _plastic_deps(), stress_n, state_n
        )
        assert r.n_substeps == 4

    def test_n_substeps_is_a_power_of_two(self, model, inner):
        stress_n, state_n = _zero_start(model)
        r = SubSteppingIntegrator(inner).stress_update(
            _plastic_deps(), stress_n, state_n
        )
        m = r.n_substeps
        assert m & (m - 1) == 0

    def test_matches_a_hand_rolled_substep_march(self, model, inner):
        """The subdivided state must equal marching Δε/m through the inner integrator."""
        stress_n, state_n = _zero_start(model)
        r = SubSteppingIntegrator(inner).stress_update(
            _plastic_deps(), stress_n, state_n
        )

        deps = _plastic_deps() / r.n_substeps
        s, st = stress_n, state_n
        for _ in range(r.n_substeps):
            ref = inner.stress_update(deps, s, st)
            s, st = ref.stress, ref.state

        np.testing.assert_allclose(np.asarray(r.stress), np.asarray(s), rtol=1e-14)
        np.testing.assert_allclose(float(r.state["ep"]), float(st["ep"]), rtol=1e-14)

    def test_subdivided_step_carries_a_return_mapping(self, model, inner):
        """The reached state cannot be rebuilt from the elastic shortcut."""
        stress_n, state_n = _zero_start(model)
        r = SubSteppingIntegrator(inner).stress_update(
            _plastic_deps(), stress_n, state_n
        )
        assert r.return_mapping is not None

    def test_dlambda_is_the_sum_over_substeps(self, model, inner):
        stress_n, state_n = _zero_start(model)
        r = SubSteppingIntegrator(inner).stress_update(
            _plastic_deps(), stress_n, state_n
        )
        np.testing.assert_allclose(
            float(r.dlambda), float(r.state["ep"]) - float(state_n["ep"]), rtol=1e-12
        )

    def test_iterations_and_residuals_accumulate(self, model, inner):
        """Sums over substeps — elastic substeps contribute zero, not one."""
        stress_n, state_n = _zero_start(model)
        deps = _plastic_deps()
        r = SubSteppingIntegrator(inner).stress_update(deps, stress_n, state_n)

        s, st = stress_n, state_n
        n_iter, res_hist = 0, []
        for _ in range(r.n_substeps):
            ref = inner.stress_update(deps / r.n_substeps, s, st)
            n_iter += ref.n_iterations
            res_hist.extend(ref.residual_history)
            s, st = ref.stress, ref.state

        assert r.n_iterations == n_iter
        assert r.residual_history == res_hist

    def test_stress_trial_is_the_undivided_prediction(self, model, inner):
        """The caller asked about the big increment, so σ_trial is the big one."""
        stress_n, state_n = _zero_start(model)
        deps = _plastic_deps()
        r = SubSteppingIntegrator(inner).stress_update(deps, stress_n, state_n)

        C = np.asarray(model.elastic_stiffness(state_n))
        np.testing.assert_allclose(
            np.asarray(r.stress_trial), stress_n + C @ deps, rtol=1e-14
        )

    def test_is_plastic_when_any_substep_yielded(self, model, inner):
        stress_n, state_n = _zero_start(model)
        r = SubSteppingIntegrator(inner).stress_update(
            _plastic_deps(), stress_n, state_n
        )
        assert r.is_plastic is True

    def test_subdivided_solution_approaches_the_refined_one(self, model, inner):
        """Subdivision must land on the physically refined answer, not merely converge."""
        stress_n, state_n = _zero_start(model)
        deps = _plastic_deps()

        r = SubSteppingIntegrator(inner).stress_update(deps, stress_n, state_n)

        s, st = stress_n, state_n
        for _ in range(64):
            ref = inner.stress_update(deps / 64, s, st)
            s, st = ref.stress, ref.state

        np.testing.assert_allclose(np.asarray(r.stress), np.asarray(s), rtol=1e-3)

    def test_driver_run_completes_where_the_bare_integrator_cannot(self, model, inner):
        load = _uniaxial_ramp(30, 3.0e-3)

        with pytest.raises(RuntimeError):
            list(StrainDriver(inner).iter_run(load))

        steps = list(StrainDriver(SubSteppingIntegrator(inner)).iter_run(load))
        assert len(steps) == len(load.data)
        assert all(s.result.converged for s in steps)
        assert any(s.result.n_substeps > 1 for s in steps)

    def test_wrapped_integrator_may_report_by_flag_instead_of_raising(self, model):
        """Failure detection must work for either inner raise_on_nonconverged."""
        inner = PythonAnalyticalIntegrator(model, raise_on_nonconverged=False)
        stress_n, state_n = _zero_start(model)
        r = SubSteppingIntegrator(inner).stress_update(
            _plastic_deps(), stress_n, state_n
        )
        assert r.converged is True
        assert r.n_substeps > 1


# ===========================================================================
# TestExhaustion
# ===========================================================================

class TestExhaustion:

    @pytest.fixture
    def hopeless(self):
        """max_iter=0 never converges, however small the increment gets."""
        return PythonNumericalIntegrator(J2Isotropic3D(**PARAMS), max_iter=0)

    def test_raises_by_default(self, hopeless):
        stress_n, state_n = _zero_start(hopeless)
        with pytest.raises(RuntimeError, match="even at 4 substeps"):
            SubSteppingIntegrator(hopeless, max_substeps=4).stress_update(
                _plastic_deps(), stress_n, state_n
            )

    def test_returns_flag_when_not_raising(self, hopeless):
        stress_n, state_n = _zero_start(hopeless)
        r = SubSteppingIntegrator(
            hopeless, max_substeps=4, raise_on_nonconverged=False
        ).stress_update(_plastic_deps(), stress_n, state_n)
        assert r.converged is False

    def test_failed_step_holds_state_at_step_n(self, hopeless):
        stress_n, state_n = _zero_start(hopeless)
        r = SubSteppingIntegrator(
            hopeless, max_substeps=4, raise_on_nonconverged=False
        ).stress_update(_plastic_deps(), stress_n, state_n)
        np.testing.assert_array_equal(np.asarray(r.stress), stress_n)
        np.testing.assert_allclose(float(r.state["ep"]), float(state_n["ep"]))

    def test_failed_step_reports_max_substeps(self, hopeless):
        stress_n, state_n = _zero_start(hopeless)
        r = SubSteppingIntegrator(
            hopeless, max_substeps=4, raise_on_nonconverged=False
        ).stress_update(_plastic_deps(), stress_n, state_n)
        assert r.n_substeps == 4

    def test_failed_step_still_offers_a_usable_tangent(self, hopeless):
        """StressDriver inverts ddsdde before it looks at converged."""
        stress_n, state_n = _zero_start(hopeless)
        r = SubSteppingIntegrator(
            hopeless, max_substeps=4, raise_on_nonconverged=False
        ).stress_update(_plastic_deps(), stress_n, state_n)
        assert np.all(np.isfinite(np.asarray(r.ddsdde)))

    def test_max_substeps_one_is_a_bare_passthrough(self):
        """m=1 only: the wrapper adds nothing but the failure convention."""
        model = _FailsAboveDlambda(**PARAMS)
        inner = PythonAnalyticalIntegrator(model)
        stress_n, state_n = _zero_start(model)
        with pytest.raises(RuntimeError, match="even at 1 substeps"):
            SubSteppingIntegrator(inner, max_substeps=1).stress_update(
                _plastic_deps(), stress_n, state_n
            )


# ===========================================================================
# TestTangentApproximation
# ===========================================================================

class TestTangentApproximation:
    """Pins the documented approximation: ddsdde is the LAST substep's tangent.

    It is not the chain-rule product over substeps.  Fixing this test in place
    is deliberate — if someone later composes the tangent exactly, this test
    should fail and be rewritten, not silently keep passing.
    """

    @pytest.fixture
    def model(self):
        return _FailsAboveDlambda(**PARAMS)

    def test_ddsdde_equals_the_last_substep_tangent(self, model):
        inner = PythonAnalyticalIntegrator(model)
        stress_n, state_n = _zero_start(model)
        deps = _plastic_deps()
        r = SubSteppingIntegrator(inner).stress_update(deps, stress_n, state_n)

        sub = deps / r.n_substeps
        s, st = stress_n, state_n
        last = None
        for _ in range(r.n_substeps):
            last = inner.stress_update(sub, s, st)
            s, st = last.stress, last.state

        np.testing.assert_allclose(
            np.asarray(r.ddsdde), np.asarray(last.ddsdde), rtol=1e-14
        )

    def test_stress_driver_still_converges_with_the_approximate_tangent(self, model):
        """The approximation may slow the outer NR; it must not break it.

        ``max_substeps`` is raised well above the default here: StressDriver's
        outer NR probes with trial Δε far larger than the converged one, and this
        double's Δλ limit is only satisfiable by subdividing those probes hard.
        A real model's admissible branch does not shrink with Δλ this way.
        """
        from manforge.simulation.driver import StressDriver

        integrator = SubSteppingIntegrator(
            PythonAnalyticalIntegrator(model), max_substeps=256
        )
        targets = np.linspace(0.0, 280.0, 12)
        data = np.zeros((len(targets), 6))
        data[:, 0] = targets
        load = FieldHistory(type=FieldType.STRESS, name="Stress", data=data)

        steps = list(StressDriver(integrator).iter_run(load))
        assert len(steps) == len(targets)
        assert all(s.converged for s in steps)
