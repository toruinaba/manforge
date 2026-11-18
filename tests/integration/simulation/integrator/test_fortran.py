"""FortranIntegrator against a stand-in for the f2py module.

The compiled-UMAT tests (``@pytest.mark.fortran``) need gfortran.  These do
not: ``_FakeUmat`` answers ``call`` the way an f2py module would, computing the
J2 step in Python, so the integrator's own logic — the StressIntegrator
contract, the elastic tangent with and without a stiffness hook, the missing
tangent fallback — is exercised on every host.
"""

import numpy as np
import pytest

from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.simulation.driver import StressDriver
from manforge.simulation.integrator import (
    FortranIntegrator,
    PythonIntegrator,
    StressIntegrator,
)
from manforge.simulation.types import FieldHistory, FieldType

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
                    return 2.0 * np.asarray(model.elastic_stiffness(None))
                return super().call(name, *args)

        integrator = FortranIntegrator.from_model(
            _WithStiffnessSub(), "j2", model, elastic_stiffness_subroutine="j2_C"
        )
        np.testing.assert_allclose(
            integrator.elastic_tangent(np.zeros(6), model.initial_state()),
            2.0 * np.asarray(model.elastic_stiffness(None)),
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
