"""Integration tests for run_simulation — real models, driver in the loop."""

import numpy as np
import pytest

from returnmap.core import PLANE_STRESS, UNIAXIAL_1D
from returnmap.models import J2Isotropic
from returnmap.service import default_registry, run_simulation

J2 = dict(E=206e3, nu=0.3, sigma_y0=250.0, H=1000.0)
AF = dict(E=206e3, nu=0.3, sigma_y0=250.0, C_k=20e3, gamma=100.0)
OW = dict(E=206e3, nu=0.3, sigma_y0=250.0, C_k=20e3, gamma=100.0)

# Templates as a deployment has them: registered at the param_meta defaults,
# which differ from the values the tests request (E = 210 GPa vs 206 GPa).
_REG = default_registry()
J2_1D = _REG.get("j2_isotropic")
AF_1D = _REG.get("af_kinematic")
OW_1D = _REG.get("ow_kinematic")


class TestStrainControl:
    def test_j2_monotonic_matches_analytical(self):
        """Elastoplastic uniaxial: final stress = (σ_y0 + H·ep) exactly for J2 linear."""
        eps_max = 5e-3
        out = run_simulation(J2_1D, J2, np.linspace(0, eps_max, 50),
                             include_fields=["ep"])
        assert out.converged
        ep = out.fields["ep"][-1]
        assert ep > 0.0  # went plastic
        np.testing.assert_allclose(
            out.stress[-1, 0], J2["sigma_y0"] + J2["H"] * ep, rtol=1e-8)

    def test_elastic_run_stays_elastic(self):
        out = run_simulation(J2_1D, J2, np.linspace(0, 1e-4, 10),
                             include_fields=["ep"])
        assert out.converged
        assert out.fields["ep"][-1] == 0.0
        np.testing.assert_allclose(out.stress[:, 0], J2["E"] * out.strain[:, 0])

    @pytest.mark.parametrize("model,params", [(AF_1D, AF), (OW_1D, OW)],
                             ids=["af", "ow"])
    def test_cyclic_autodiff_models(self, model, params):
        """Autodiff-NR models complete a small cyclic history."""
        from returnmap.simulation import FieldHistory
        load = FieldHistory.cyclic_strain([4e-3, -4e-3, 4e-3], n_per_segment=10).data
        out = run_simulation(model, params, load)
        assert out.converged
        assert out.n_steps == 30

    def test_default_jacobian_is_exact_as_in_every_layer(self):
        load = np.linspace(0, 3e-3, 10)
        default = run_simulation(AF_1D, AF, load)
        exact = run_simulation(AF_1D, AF, load, nr_jacobian="exact")
        np.testing.assert_array_equal(default.stress, exact.stress)

    def test_broyden_jacobian_option(self):
        load = np.linspace(0, 3e-3, 10)
        exact = run_simulation(AF_1D, AF, load)
        out = run_simulation(AF_1D, AF, load, nr_jacobian="broyden")
        assert out.converged
        np.testing.assert_allclose(out.stress, exact.stress, rtol=1e-8)


class TestStressControl:
    def test_plastic_target_reached(self):
        out = run_simulation(J2_1D, J2, np.linspace(0, 300.0, 30),
                             control="stress", include_fields=["ep"])
        assert out.converged
        np.testing.assert_allclose(out.stress[-1, 0], 300.0, atol=1e-6)
        # beyond yield: plastic strain accumulated and strain exceeds elastic
        assert out.fields["ep"][-1] > 0.0
        assert out.strain[-1, 0] > 300.0 / J2["E"]


class TestNonConvergence:
    def test_partial_history_returned(self):
        """Unsolvable parameters: partial history + failed_step, no exception."""
        # OW with absurd recovery: NR cannot track the backstress evolution
        # at a large strain jump.
        bad = dict(OW, gamma=1e12, C_k=1e9)
        out = run_simulation(OW_1D, bad, np.array([0.0, 0.05]))
        assert not out.converged
        assert out.failed_step is not None
        assert out.n_steps == out.failed_step  # failed step excluded
        assert out.stress.shape == (out.n_steps, 1)

    def test_zero_steps_shape(self):
        """Failure at the very first step still yields well-formed arrays."""
        bad = dict(OW, gamma=1e12, C_k=1e9)
        out = run_simulation(OW_1D, bad, np.array([0.05]),
                             include_fields=["ep"])
        if out.converged:
            pytest.skip("first step unexpectedly solvable")
        assert out.n_steps == 0
        assert out.stress.shape == (0, 1)
        assert out.fields["ep"].shape == (0,)


    # 1D closed form: Δλ = f_trial / (E + H).
    @pytest.mark.parametrize("H", [-300_000.0, -206e3],
                             ids=["negative-dlambda", "division-by-zero"])
    def test_closed_form_model_with_unsolvable_parameters(self, H):
        """A closed form returns a number for any parameters; the run must
        still report that it is not a solution, not hand back negative or
        non-finite plastic strain as converged."""
        params = dict(J2, H=H)
        with np.errstate(all="ignore"):
            out = run_simulation(J2_1D, params, np.linspace(0.0, 6e-3, 5),
                                 include_fields=["ep"])
        assert not out.converged
        assert out.failed_step is not None
        assert out.n_steps == out.failed_step
        assert np.all(np.isfinite(out.stress))
        assert np.all(out.fields["ep"] >= 0.0)


class TestRegistryRoundTrip:
    def test_describe_then_run_with_defaults(self):
        """The descriptor's defaults form a runnable parameter set (WebUI path)."""
        reg = default_registry()
        for name in reg.names():
            d = reg.describe(name)
            params = {k: m.default for k, m in d.param_meta.items()}
            assert None not in params.values(), name
            out = run_simulation(reg.get(name), params, np.linspace(0, 2e-3, 10))
            assert out.converged, name
            assert out.n_steps == 10


class TestTemplate:
    """What the template contributes: class, dimension, non-parameter settings."""

    def test_template_is_left_untouched(self):
        before = dict(J2_1D.params)
        run_simulation(J2_1D, J2, np.linspace(0, 5e-3, 10))
        assert J2_1D.params == before
        assert before != J2

    def test_dimension_comes_from_the_template(self):
        """A dimension-free family runs in the dimension its template names."""
        template = J2Isotropic(dimension=PLANE_STRESS, **J2)
        load = np.zeros((10, 3))
        load[:, 0] = np.linspace(0, 5e-3, 10)
        out = run_simulation(template, J2, load, include_fields=["ep"])
        assert out.converged
        assert out.stress.shape == (10, 3)
        assert out.fields["ep"][-1] > 0.0

    def test_non_parameter_setting_carries_over(self):
        """A setting outside param_names is the template's to choose."""

        class ScaledYield(J2Isotropic):
            def __init__(self, *, factor, **params):
                super().__init__(**params)
                self.factor = factor

            def yield_function(self, state):
                return self.vonmises(state["stress"]) - self.factor * (
                    self.sigma_y0 + self.H * state["ep"]
                )

        load = np.linspace(0, 5e-3, 10)
        for factor in (1.0, 1.5):
            template = ScaledYield(factor=factor, dimension=UNIAXIAL_1D, **J2)
            out = run_simulation(template, J2, load, include_fields=["ep"])
            assert out.converged
            ep = out.fields["ep"][-1]
            assert ep > 0.0
            np.testing.assert_allclose(
                out.stress[-1, 0], factor * (J2["sigma_y0"] + J2["H"] * ep), rtol=1e-8)
