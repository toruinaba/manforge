"""compare_models — several models over one loading history."""

import numpy as np
import pytest

from returnmap.fitting import FitTarget, Observation, ParameterSpace, fit_params
from returnmap.models import (
    AFKinematic1D, AFKinematic3D, J2Isotropic1D, J2Isotropic3D, OWKinematic1D,
)
from returnmap.simulation import (
    FieldHistory, FieldType, FortranIntegrator, MixedDriver, ModelComparisonResult,
    ModelRun, PythonAnalyticalIntegrator, PythonIntegrator,
    PythonNumericalIntegrator, StrainDriver, StressDriver, compare_models,
)

BASE = dict(E=206e3, nu=0.3, sigma_y0=250.0)
J2 = J2Isotropic1D(**BASE, H=1000.0)
AF = AFKinematic1D(**BASE, C_k=20e3, gamma=100.0)
OW = OWKinematic1D(**BASE, C_k=20e3, gamma=100.0)
# Absurd recovery: the return mapping has no admissible root at the first
# plastic step, so the run stops there.
UNSOLVABLE = OWKinematic1D(**BASE, C_k=1e9, gamma=1e12)

LOAD = FieldHistory.cyclic_strain([6e-3, -6e-3], n_per_segment=15)
N = 30


@pytest.fixture(scope="module")
def comparison():
    return compare_models({"j2": J2, "af": AF, "ow": OW}, LOAD)


class _FakeUmat:
    """Stands in for an f2py module: the J2 step, computed in Python."""

    def call(self, name, *args):
        E, nu, sigma_y0, H, stress_n, ep, dstran = args
        model = J2Isotropic3D(E=E, nu=nu, sigma_y0=sigma_y0, H=H)
        state_n = {"stress": np.asarray(stress_n), "ep": np.array(ep)}
        r = PythonIntegrator(model).stress_update(
            np.asarray(dstran), np.asarray(stress_n), state_n
        )
        return np.asarray(r.stress), float(r.state["ep"]), np.asarray(r.ddsdde)


def uniaxial(integrator):
    """Uniaxial stress for any dimension, driven by the axial strain."""
    if integrator.ntens == 1:
        return StrainDriver(integrator)
    return MixedDriver(integrator, prescribed_strain_idx=[0])


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

class TestResponse:
    def test_one_run_per_model_in_the_order_given(self, comparison):
        assert isinstance(comparison, ModelComparisonResult)
        assert list(comparison.runs) == ["j2", "af", "ow"]
        assert all(isinstance(run, ModelRun) for run in comparison.runs.values())
        assert [run.name for run in comparison.runs.values()] == ["j2", "af", "ow"]
        assert comparison.n_load_steps == N
        assert comparison.converged

    @pytest.mark.parametrize("name, model", [("j2", J2), ("af", AF), ("ow", OW)])
    def test_each_run_is_the_run_of_that_model_alone(self, comparison, name, model):
        alone = StrainDriver(PythonIntegrator(model)).run(LOAD)
        run = comparison.runs[name]
        np.testing.assert_array_equal(run.result.stress, alone.stress)
        np.testing.assert_array_equal(run.result.strain, alone.strain)
        np.testing.assert_array_equal(run.result.history("ep"), alone.history("ep"))

    def test_a_run_reads_like_its_driver_result(self, comparison):
        run = comparison.runs["af"]
        np.testing.assert_array_equal(run.stress, run.result.stress)
        np.testing.assert_array_equal(run.strain, run.result.strain)
        np.testing.assert_array_equal(run.history("alpha"), run.result.history("alpha"))
        assert run.n_steps == N
        assert run.converged and run.failed_step is None and run.error == ""

    def test_the_models_do_differ(self, comparison):
        assert not np.allclose(comparison.runs["j2"].stress, comparison.runs["af"].stress)


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

class TestCost:
    def test_iterations_are_averaged_over_plastic_steps(self, comparison):
        run = comparison.runs["af"]
        steps = run.result.step_results
        n_plastic = sum(r.is_plastic for r in steps)
        assert 0 < n_plastic < N
        assert run.n_plastic_steps == n_plastic
        assert run.n_iterations == sum(r.n_iterations for r in steps)
        assert run.iterations_per_plastic_step == run.n_iterations / n_plastic

    def test_a_closed_form_takes_one_update_per_plastic_step(self, comparison):
        assert comparison.runs["j2"].iterations_per_plastic_step == 1.0

    def test_time_is_measured_per_step(self, comparison):
        for run in comparison.runs.values():
            assert run.elapsed > 0.0
            assert run.time_per_step == run.elapsed / N

    def test_strain_control_takes_one_stress_update_per_step(self, comparison):
        assert all(r.n_outer_iterations == N for r in comparison.runs.values())

    def test_no_plastic_step_means_no_average(self):
        elastic = FieldHistory(FieldType.STRAIN, np.linspace(0.0, 1e-4, 5))
        run = compare_models({"j2": J2}, elastic).runs["j2"]
        assert run.n_plastic_steps == 0
        assert run.n_iterations == 0
        assert run.iterations_per_plastic_step is None

    def test_an_integrator_that_reports_no_plasticity_flag(self):
        """A Fortran UMAT says neither which steps were plastic nor how many
        iterations they took; the run reports "unknown", not zero."""
        model = J2Isotropic3D(**BASE, H=1000.0)
        load = FieldHistory.cyclic_strain([6e-3, -6e-3], n_per_segment=15, ntens=6)
        c = compare_models(
            {"python": model,
             "umat": FortranIntegrator.from_model(_FakeUmat(), "j2", model)},
            load,
        )
        python, umat = c.runs["python"], c.runs["umat"]
        np.testing.assert_allclose(umat.stress, python.stress, rtol=1e-12)
        assert python.n_plastic_steps > 0
        assert umat.n_plastic_steps is None
        assert umat.n_iterations is None
        assert umat.iterations_per_plastic_step is None
        assert umat.time_per_step > 0.0
        assert c.summary().splitlines()[2].split()[:4] == ["umat", f"{N}/{N}", "—", "—"]


# ---------------------------------------------------------------------------
# Choosing how a model is solved, and how it is driven
# ---------------------------------------------------------------------------

class TestIntegratorsAndDrivers:
    def test_an_integrator_is_run_as_given(self):
        model = J2Isotropic3D(**BASE, H=1000.0)
        load = FieldHistory.cyclic_strain([6e-3, -6e-3], n_per_segment=15, ntens=6)
        c = compare_models(
            {"closed form": PythonAnalyticalIntegrator(model),
             "autodiff": PythonNumericalIntegrator(model)},
            load,
        )
        np.testing.assert_allclose(
            c.runs["autodiff"].stress, c.runs["closed form"].stress, rtol=1e-8)
        assert c.runs["closed form"].iterations_per_plastic_step == 1.0

    def test_nr_jacobian_reaches_the_bare_models(self):
        exact = compare_models({"af": AF}, LOAD).runs["af"]
        broyden = compare_models({"af": AF}, LOAD, nr_jacobian="broyden").runs["af"]
        np.testing.assert_allclose(broyden.stress, exact.stress, rtol=1e-8)
        assert broyden.n_iterations > exact.n_iterations

    def test_one_factory_drives_models_of_different_dimensions(self):
        c = compare_models(
            {"1D": AF, "3D": AFKinematic3D(**BASE, C_k=20e3, gamma=100.0)},
            LOAD, driver_factory=uniaxial,
        )
        assert c.converged
        assert c.runs["3D"].stress.shape == (N, 6)
        # Uniaxial stress either way, so the axial responses agree.
        np.testing.assert_allclose(
            c.runs["3D"].stress[:, 0], c.runs["1D"].stress[:, 0], rtol=1e-6)
        np.testing.assert_allclose(c.runs["3D"].stress[:, 1:], 0.0, atol=1e-6)
        # The mixed driver iterates; the strain driver does not.
        assert c.runs["1D"].n_outer_iterations == N
        assert c.runs["3D"].n_outer_iterations > N

    def test_a_stress_history_is_stress_controlled(self):
        """Same default as fit_params and crosscheck."""
        target = FieldHistory(FieldType.STRESS, np.linspace(0.0, 300.0, 10).reshape(-1, 1))
        run = compare_models({"j2": J2}, target).runs["j2"]
        alone = StressDriver(PythonIntegrator(J2)).run(target)
        np.testing.assert_array_equal(run.strain, alone.strain)
        np.testing.assert_allclose(run.stress[-1, 0], 300.0, atol=1e-6)
        assert run.n_outer_iterations > 10


# ---------------------------------------------------------------------------
# Non-convergence is a row of the table, not an exception
# ---------------------------------------------------------------------------

class TestNonConvergence:
    def test_a_model_that_stops_does_not_stop_the_others(self):
        c = compare_models({"af": AF, "unsolvable": UNSOLVABLE, "j2": J2}, LOAD)
        bad = c.runs["unsolvable"]
        assert not c.converged
        assert not bad.converged
        assert bad.failed_step is not None and bad.failed_step < N
        assert bad.n_steps == bad.failed_step
        assert bad.stress.shape == (bad.n_steps, 1)
        assert c.runs["af"].converged and c.runs["j2"].converged
        assert c.runs["j2"].n_steps == N

    def test_an_integrator_that_raises_is_reported_the_same_way(self):
        """PythonIntegrator raises on non-convergence by default."""
        reported = compare_models({"m": UNSOLVABLE}, LOAD).runs["m"]
        raised = compare_models({"m": PythonIntegrator(UNSOLVABLE)}, LOAD).runs["m"]
        assert not raised.converged
        assert raised.failed_step == reported.failed_step
        assert raised.n_steps == reported.n_steps
        np.testing.assert_array_equal(raised.stress, reported.stress)
        assert raised.error.startswith("RuntimeError")
        assert reported.error == ""

    def test_failure_at_the_first_step(self):
        jump = FieldHistory(FieldType.STRAIN, np.array([0.05]))
        c = compare_models({"m": UNSOLVABLE}, jump)
        run = c.runs["m"]
        assert run.failed_step == 0 and run.n_steps == 0
        assert run.stress.shape == (0, 1) and run.strain.shape == (0, 1)
        assert run.iterations_per_plastic_step is None
        assert run.time_per_step == run.elapsed
        assert "no — stopped at step 0" in c.summary()

    def test_a_mistake_in_the_call_still_raises(self):
        wrong_width = FieldHistory(FieldType.STRAIN, np.zeros((5, 6)))
        with pytest.raises(ValueError):
            compare_models({"j2": J2}, wrong_width)


# ---------------------------------------------------------------------------
# Goodness of fit
# ---------------------------------------------------------------------------

class TestFit:
    @pytest.fixture(scope="class")
    def target(self):
        """The AF model's own response, as the "experiment"."""
        stress = StrainDriver(PythonIntegrator(AF)).run(LOAD).stress[:, 0]
        return FitTarget.stress_strain(LOAD.data, stress)

    def test_not_evaluated_unless_asked(self, comparison):
        assert all(r.fit_residual is None and r.fit_error == ""
                   for r in comparison.runs.values())

    def test_residual_is_the_objective_value(self, target):
        c = compare_models({"j2": J2, "af": AF, "ow": OW}, LOAD, targets=[target])
        assert c.runs["af"].fit_residual == 0.0
        for name, model in (("j2", J2), ("ow", OW)):
            expected = target.residual(PythonIntegrator(model), None)
            assert c.runs[name].fit_residual == expected > 0.0

    def test_agrees_with_what_fit_params_reports(self, target):
        fit = fit_params(
            J2, [target],
            ParameterSpace(free={"H": (1000.0, (0.0, 50e3))}, fixed=dict(BASE)),
        )
        run = compare_models({"fitted": fit.model}, LOAD, targets=[target]).runs["fitted"]
        assert run.fit_residual == pytest.approx(fit.residual, rel=1e-12)

    def test_targets_are_summed_and_run_with_their_own_load(self, target):
        short = FieldHistory(FieldType.STRAIN, np.linspace(0.0, 3e-3, 8))
        other = FitTarget(
            load=short,
            observations=[Observation(values=np.zeros(8), field="ep")],
        )
        both = compare_models({"j2": J2}, LOAD, targets=[target, other]).runs["j2"]
        integrator = PythonIntegrator(J2)
        assert both.fit_residual == (
            target.residual(integrator, None) + other.residual(integrator, None)
        )

    def test_a_model_that_cannot_run_a_target_has_no_residual(self, target):
        c = compare_models({"af": AF, "m": UNSOLVABLE}, LOAD, targets=[target])
        run = c.runs["m"]
        assert run.fit_residual is None
        assert "did not converge" in run.fit_error
        assert c.runs["af"].fit_residual == 0.0
        assert "m: fit not evaluated" in c.summary()

    def test_targets_must_be_fit_targets(self, target):
        with pytest.raises(TypeError, match="sequence of FitTarget"):
            compare_models({"j2": J2}, LOAD, targets=target)


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

class TestSummary:
    def test_one_row_per_model(self, comparison):
        lines = comparison.summary().splitlines()
        assert lines[0].split()[0] == "model"
        assert [line.split()[0] for line in lines[1:]] == ["j2", "af", "ow"]
        assert all(f"{N}/{N}" in line and line.endswith("yes") for line in lines[1:])
        assert str(comparison) == comparison.summary()

    def test_columns_appear_only_when_they_say_something(self, comparison):
        plain = comparison.summary().splitlines()[0]
        assert "fit residual" not in plain and "outer it/step" not in plain

        mixed = compare_models(
            {"3D": AFKinematic3D(**BASE, C_k=20e3, gamma=100.0)},
            FieldHistory(FieldType.STRAIN, np.linspace(0.0, 3e-3, 5).reshape(-1, 1)),
            driver_factory=uniaxial,
        )
        assert "outer it/step" in mixed.summary().splitlines()[0]

        fitted = compare_models(
            {"j2": J2}, LOAD,
            targets=[FitTarget.stress_strain(LOAD.data, np.zeros(N))],
        )
        assert "fit residual" in fitted.summary().splitlines()[0]

    def test_a_stopped_run_says_where(self):
        c = compare_models({"af": AF, "unsolvable": UNSOLVABLE}, LOAD)
        step = c.runs["unsolvable"].failed_step
        row = c.summary().splitlines()[2]
        assert row.split()[:2] == ["unsolvable", f"{step}/{N}"]
        assert row.endswith(f"no — stopped at step {step}")


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

class TestArguments:
    def test_models_must_be_a_mapping(self):
        with pytest.raises(TypeError, match="mapping of name to model"):
            compare_models([J2, AF], LOAD)

    def test_models_must_not_be_empty(self):
        with pytest.raises(ValueError, match="no models"):
            compare_models({}, LOAD)

    def test_a_class_is_not_a_model(self):
        with pytest.raises(TypeError, match="is the class J2Isotropic1D"):
            compare_models({"j2": J2Isotropic1D}, LOAD)

    def test_neither_model_nor_integrator(self):
        with pytest.raises(TypeError, match="StressIntegrator"):
            compare_models({"j2": object()}, LOAD)
