"""Integration tests for run_simulation — real models, driver in the loop."""

import numpy as np
import pytest

from manforge.models import AFKinematic1D, J2Isotropic1D, OWKinematic1D
from manforge.service import default_registry, run_simulation

J2 = dict(E=206e3, nu=0.3, sigma_y0=250.0, H=1000.0)
AF = dict(E=206e3, nu=0.3, sigma_y0=250.0, C_k=20e3, gamma=100.0)
OW = dict(E=206e3, nu=0.3, sigma_y0=250.0, C_k=20e3, gamma=100.0)


class TestStrainControl:
    def test_j2_monotonic_matches_analytical(self):
        """Elastoplastic uniaxial: final stress = (σ_y0 + H·ep) exactly for J2 linear."""
        eps_max = 5e-3
        out = run_simulation(J2Isotropic1D, J2, np.linspace(0, eps_max, 50),
                             include_fields=["ep"])
        assert out.converged
        ep = out.fields["ep"][-1]
        assert ep > 0.0  # went plastic
        np.testing.assert_allclose(
            out.stress[-1, 0], J2["sigma_y0"] + J2["H"] * ep, rtol=1e-8)

    def test_elastic_run_stays_elastic(self):
        out = run_simulation(J2Isotropic1D, J2, np.linspace(0, 1e-4, 10),
                             include_fields=["ep"])
        assert out.converged
        assert out.fields["ep"][-1] == 0.0
        np.testing.assert_allclose(out.stress[:, 0], J2["E"] * out.strain[:, 0])

    @pytest.mark.parametrize("model_cls,params", [
        (AFKinematic1D, AF), (OWKinematic1D, OW)])
    def test_cyclic_autodiff_models(self, model_cls, params):
        """Autodiff-NR models complete a small cyclic history via broyden."""
        from manforge.simulation.types import FieldHistory
        load = FieldHistory.cyclic_strain([4e-3, -4e-3, 4e-3], n_per_segment=10).data
        out = run_simulation(model_cls, params, load)
        assert out.converged
        assert out.n_steps == 30

    def test_exact_jacobian_option(self):
        out = run_simulation(AFKinematic1D, AF, np.linspace(0, 3e-3, 10),
                             nr_jacobian="exact")
        assert out.converged


class TestStressControl:
    def test_plastic_target_reached(self):
        out = run_simulation(J2Isotropic1D, J2, np.linspace(0, 300.0, 30),
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
        out = run_simulation(OWKinematic1D, bad, np.array([0.0, 0.05]))
        assert not out.converged
        assert out.failed_step is not None
        assert out.n_steps == out.failed_step  # failed step excluded
        assert out.stress.shape == (out.n_steps, 1)

    def test_zero_steps_shape(self):
        """Failure at the very first step still yields well-formed arrays."""
        bad = dict(OW, gamma=1e12, C_k=1e9)
        out = run_simulation(OWKinematic1D, bad, np.array([0.05]),
                             include_fields=["ep"])
        if out.converged:
            pytest.skip("first step unexpectedly solvable")
        assert out.n_steps == 0
        assert out.stress.shape == (0, 1)
        assert out.fields["ep"].shape == (0,)


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
