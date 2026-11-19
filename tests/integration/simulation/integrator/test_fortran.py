"""FortranIntegrator against a stand-in for the f2py module.

The compiled-UMAT tests (``@pytest.mark.fortran``) need gfortran.  These do
not: ``_FakeUmat`` answers ``call`` the way an f2py module would, computing the
J2 step in Python, so the integrator's own logic — the StressIntegrator
contract, the elastic tangent with and without a stiffness hook, the missing
tangent fallback — is exercised on every host.
"""

import numpy as np
import pytest

from manforge.models import J2Isotropic3D
from manforge.simulation import (
    StressDriver, FortranIntegrator, PythonIntegrator, StressIntegrator, FieldHistory,
    FieldType,
)

PARAMS = dict(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)


class _FakeUmat:
    """One subroutine, ``j2``: (E, nu, sigma_y0, H, stress_n, ep, dstran)."""

    def __init__(self, *, with_tangent=True):
        self.with_tangent = with_tangent
        self.strain_incs = []   # the dstran of every UMAT call, in order

    def call(self, name, *args):
        assert name == "j2", name
        E, nu, sigma_y0, H, stress_n, ep, dstran = args
        self.strain_incs.append(np.asarray(dstran, dtype=float).copy())
        model = J2Isotropic3D(E=E, nu=nu, sigma_y0=sigma_y0, H=H)
        state_n = {"stress": np.asarray(stress_n), "ep": np.array(ep)}
        r = PythonIntegrator(model).stress_update(
            np.asarray(dstran), np.asarray(stress_n), state_n
        )
        out = (np.asarray(r.stress), float(r.state["ep"]))
        return out + ((np.asarray(r.ddsdde),) if self.with_tangent else ())


@pytest.fixture
def model():
    return J2Isotropic3D(**PARAMS)


def _without_hook(fake, model, **kwargs):
    """A FortranIntegrator with no elastic-stiffness hook at all."""
    return FortranIntegrator(
        fake, "j2",
        dimension=model.dimension,
        initial_state=model.initial_state,
        param_fn=lambda: tuple(getattr(model, n) for n in model.param_names),
        state_names=["ep"],
        **kwargs,
    )


def _stress_load(model, peak):
    sigma = np.zeros((6, model.ntens))
    sigma[:, 0] = np.linspace(50.0, peak, 6)
    return FieldHistory(FieldType.STRESS, "Stress", sigma)


# ---------------------------------------------------------------------------
# The StressIntegrator contract
# ---------------------------------------------------------------------------

class TestContract:
    def test_fulfils_the_contract_with_and_without_hook(self, model):
        assert isinstance(
            FortranIntegrator.from_model(_FakeUmat(), "j2", model), StressIntegrator
        )
        assert isinstance(_without_hook(_FakeUmat(), model), StressIntegrator)

    def test_reports_no_increment(self, model):
        """A UMAT is a closed target: no trial state, so no increment."""
        integrator = FortranIntegrator.from_model(_FakeUmat(), "j2", model)
        deps = np.full(6, 1e-6)
        assert integrator.increment(deps, np.zeros(6), model.initial_state()) is None
        r = integrator.stress_update(deps, np.zeros(6), model.initial_state())
        assert r.increment is None
        assert r.stress_trial is None

    def test_raise_on_nonconverged_defaults_to_true(self, model):
        assert FortranIntegrator.from_model(
            _FakeUmat(), "j2", model
        ).raise_on_nonconverged is True

    def test_raise_on_nonconverged_is_settable(self, model):
        assert FortranIntegrator.from_model(
            _FakeUmat(), "j2", model, raise_on_nonconverged=False
        ).raise_on_nonconverged is False
        assert _without_hook(
            _FakeUmat(), model, raise_on_nonconverged=False
        ).raise_on_nonconverged is False


# ---------------------------------------------------------------------------
# elastic_tangent
# ---------------------------------------------------------------------------

class TestElasticTangent:
    def test_hook_is_used_without_calling_the_umat(self, model):
        fake = _FakeUmat()
        integrator = FortranIntegrator.from_model(fake, "j2", model)
        state = model.initial_state()
        C = integrator.elastic_tangent(np.zeros(6), state)
        np.testing.assert_allclose(C, np.asarray(model.elastic_stiffness(state)))
        assert fake.strain_incs == []

    def test_without_hook_probes_the_umat_at_zero_increment(self, model):
        fake = _FakeUmat()
        integrator = _without_hook(fake, model)
        state = model.initial_state()
        C = integrator.elastic_tangent(np.zeros(6), state)
        np.testing.assert_allclose(C, np.asarray(model.elastic_stiffness(state)))
        assert len(fake.strain_incs) == 1
        np.testing.assert_array_equal(fake.strain_incs[0], np.zeros(6))

    def test_subroutine_hook_takes_precedence_over_the_model(self, model):
        class _WithStiffnessSub(_FakeUmat):
            def call(self, name, *args):
                if name == "j2_C":
                    return 2.0 * np.asarray(model.isotropic_stiffness())
                return super().call(name, *args)

        integrator = FortranIntegrator.from_model(
            _WithStiffnessSub(), "j2", model, elastic_stiffness_subroutine="j2_C"
        )
        np.testing.assert_allclose(
            integrator.elastic_tangent(np.zeros(6), model.initial_state()),
            2.0 * np.asarray(model.isotropic_stiffness()),
        )


# ---------------------------------------------------------------------------
# A UMAT that returns no tangent
# ---------------------------------------------------------------------------

class TestMissingTangent:
    def test_falls_back_to_the_hook(self, model):
        integrator = FortranIntegrator.from_model(
            _FakeUmat(with_tangent=False), "j2", model
        )
        state = model.initial_state()
        r = integrator.stress_update(np.full(6, 1e-6), np.zeros(6), state)
        np.testing.assert_allclose(
            np.asarray(r.ddsdde), np.asarray(model.elastic_stiffness(state))
        )

    def test_without_hook_the_error_propagates(self, model):
        integrator = _without_hook(_FakeUmat(with_tangent=False), model)
        with pytest.raises(ValueError, match="ddsdde"):
            integrator.stress_update(np.full(6, 1e-6), np.zeros(6), model.initial_state())


# ---------------------------------------------------------------------------
# Under a driver
# ---------------------------------------------------------------------------

class TestUnderStressDriver:
    @pytest.mark.parametrize("build", [
        lambda fake, model: FortranIntegrator.from_model(fake, "j2", model),
        _without_hook,
    ], ids=["with_hook", "without_hook"])
    def test_matches_the_python_integrator(self, model, build):
        """Same history, same answer — whichever way the starting guess comes."""
        load = _stress_load(model, peak=320.0)   # crosses yield at 250
        ref = StressDriver(PythonIntegrator(model)).run(load)
        got = StressDriver(build(_FakeUmat(), model)).run(load)

        assert got.converged
        np.testing.assert_allclose(got.stress, ref.stress, rtol=1e-9, atol=1e-8)
        np.testing.assert_allclose(got.strain, ref.strain, rtol=1e-9, atol=1e-14)

    def test_outer_failure_raises_by_default(self, model):
        driver = StressDriver(
            FortranIntegrator.from_model(_FakeUmat(), "j2", model), max_iter=1
        )
        with pytest.raises(RuntimeError, match="NR did not converge"):
            driver.run(_stress_load(model, peak=320.0))

    def test_outer_failure_is_a_value_when_asked(self, model):
        driver = StressDriver(
            FortranIntegrator.from_model(
                _FakeUmat(), "j2", model, raise_on_nonconverged=False
            ),
            max_iter=1,
        )
        result = driver.run(_stress_load(model, peak=320.0))
        assert result.converged is False
        # Elastic steps solve in one iteration; the first plastic one cannot.
        assert result.failed_step is not None and result.failed_step > 0
        assert len(result.step_results) == result.failed_step


# ---------------------------------------------------------------------------
# The UMAT's own convergence report
# ---------------------------------------------------------------------------

class _ReportingUmat(_FakeUmat):
    """Appends (converged, fail_code) to the return, as a ported NR driver does.

    ``fail_above``: steps whose axial strain increment exceeds it are reported
    as failed, with the stress left at a non-solution.
    """

    def __init__(self, fail_above):
        super().__init__()
        self.fail_above = fail_above

    def call(self, name, *args):
        stress, ep, ddsdde = super().call(name, *args)
        failed = abs(float(args[-1][0])) > self.fail_above
        if failed:
            stress = stress + 1.0e3          # what a non-converged iterate looks like
        return stress, ep, ddsdde, 0 if failed else 1, 7 if failed else 0


def _failure(ret):
    return None if ret[3] else f"fail_code={int(ret[4])}"


class TestUmatFailure:
    def test_without_the_hook_a_failed_umat_looks_converged(self, model):
        """The reason the hook exists: the iterate is returned as a solution."""
        integ = FortranIntegrator.from_model(_ReportingUmat(fail_above=1e-3), "j2", model)
        deps = np.zeros(6)
        deps[0] = 5e-3
        r = integ.stress_update(deps, np.zeros(6), model.initial_state())
        assert r.converged is True

    def test_failure_raises_with_the_umats_description(self, model):
        integ = FortranIntegrator.from_model(
            _ReportingUmat(fail_above=1e-3), "j2", model, parse_umat_failure=_failure)
        deps = np.zeros(6)
        deps[0] = 5e-3
        with pytest.raises(RuntimeError, match=r"'j2' reported that it did not converge: fail_code=7"):
            integ.stress_update(deps, np.zeros(6), model.initial_state())

    def test_converged_call_is_unaffected(self, model):
        integ = FortranIntegrator.from_model(
            _ReportingUmat(fail_above=1e-3), "j2", model, parse_umat_failure=_failure)
        deps = np.zeros(6)
        deps[0] = 5e-4
        r = integ.stress_update(deps, np.zeros(6), model.initial_state())
        assert r.converged is True

    def test_failure_is_a_value_when_asked(self, model):
        integ = FortranIntegrator.from_model(
            _ReportingUmat(fail_above=1e-3), "j2", model,
            parse_umat_failure=_failure, raise_on_nonconverged=False)
        deps = np.zeros(6)
        deps[0] = 5e-3
        r = integ.stress_update(deps, np.zeros(6), model.initial_state())
        assert r.converged is False

    def test_driver_stops_at_the_failed_step(self, model):
        from manforge.simulation import StrainDriver

        integ = FortranIntegrator.from_model(
            _ReportingUmat(fail_above=1e-3), "j2", model,
            parse_umat_failure=_failure, raise_on_nonconverged=False)
        eps = np.zeros((4, 6))
        eps[:, 0] = [5e-4, 1e-3, 3e-3, 4e-3]      # the third increment is 2e-3
        result = StrainDriver(integ).run(FieldHistory(FieldType.STRAIN, "Strain", eps))
        assert result.converged is False and result.failed_step == 2
        assert len(result.step_results) == 2

    def test_crosscheck_does_not_compare_a_failed_iterate_as_a_solution(self, model):
        from manforge.verification import crosscheck

        integ = FortranIntegrator.from_model(
            _ReportingUmat(fail_above=1e-3), "j2", model,
            parse_umat_failure=_failure, raise_on_nonconverged=False)
        eps = np.zeros((4, 6))
        eps[:, 0] = [5e-4, 1e-3, 3e-3, 4e-3]
        result = crosscheck(
            PythonIntegrator(model), integ, FieldHistory(FieldType.STRAIN, "Strain", eps))
        assert not result.passed
        assert result.first_failure.index == 2
        assert result.first_failure.b_converged is False
        assert result.n_b_nonconverged == 1
