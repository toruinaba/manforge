"""Tests for JacobianBlocks and ad_jacobian_blocks."""

import autograd
import autograd.numpy as anp
import numpy as np
import pytest

from manforge.core.jacobian import ad_jacobian_blocks, JacobianBlocks
from manforge.simulation.integrator import PythonIntegrator
from manforge.models.af_kinematic import AFKinematic3D
from manforge.models.ow_kinematic import OWKinematic3D


@pytest.fixture
def af_model():
    return AFKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=50000.0, gamma=300.0)


@pytest.fixture
def ow_model():
    return OWKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=50000.0, gamma=300.0)


def _plastic_result(model, strain_scale=3e-3):
    deps = (lambda _a: (_a.__setitem__(0, strain_scale), _a)[1])(np.zeros(model.ntens))
    state0 = model.initial_state()
    return PythonIntegrator(model).stress_update(deps, anp.zeros(model.ntens), state0), state0


# ---------------------------------------------------------------------------
# Structure tests (reduced hardening — J2)
# ---------------------------------------------------------------------------

class TestReducedBlocks:
    def test_returns_jacobian_blocks(self, model):
        result, state0 = _plastic_result(model)
        jac = ad_jacobian_blocks(model, result, state0)
        assert isinstance(jac, JacobianBlocks)

    def test_fixed_block_shapes(self, model):
        result, state0 = _plastic_result(model)
        jac = ad_jacobian_blocks(model, result, state0)
        ntens = model.ntens
        assert jac.dstress_dsigma.shape == (ntens, ntens)
        assert jac.dstress_ddlambda.shape == (ntens,)
        assert jac.dyield_dsigma.shape == (ntens,)
        assert jac.full.shape == (ntens + 1, ntens + 1)

    def test_state_blocks_are_none(self, model):
        result, state0 = _plastic_result(model)
        jac = ad_jacobian_blocks(model, result, state0)
        assert jac.dstress_dstate is None
        assert jac.dyield_dstate is None
        assert jac.dstate_dsigma is None
        assert jac.dstate_ddlambda is None
        assert jac.dstate_dstate is None

    def test_dyield_dsigma_matches_flow_direction(self, model):
        result, state0 = _plastic_result(model)
        jac = ad_jacobian_blocks(model, result, state0)

        # flow direction is autograd.grad of yield_function w.r.t. stress
        n_ad = autograd.grad(lambda s: model.yield_function(s, result.state))(result.stress)
        np.testing.assert_allclose(
            np.array(jac.dyield_dsigma), np.array(n_ad), rtol=1e-10
        )

    def test_full_matrix_blocks_consistent(self, model):
        result, state0 = _plastic_result(model)
        jac = ad_jacobian_blocks(model, result, state0)
        ntens = model.ntens

        np.testing.assert_array_equal(jac.dstress_dsigma, jac.full[:ntens, :ntens])
        np.testing.assert_array_equal(jac.dstress_ddlambda, jac.full[:ntens, ntens])
        np.testing.assert_array_equal(jac.dyield_dsigma, jac.full[ntens, :ntens])
        np.testing.assert_array_equal(jac.dyield_ddlambda, jac.full[ntens, ntens])

    def test_elastic_step_raises_no_error(self, model):
        deps = anp.array([1e-4, 0, 0, 0, 0, 0])
        state0 = model.initial_state()
        result = PythonIntegrator(model).stress_update(deps, anp.zeros(6), state0)
        # elastic step: dlambda=0, stress_trial==stress, but jacobian should still work
        jac = ad_jacobian_blocks(model, result, state0)
        assert isinstance(jac, JacobianBlocks)


# ---------------------------------------------------------------------------
# Structure tests (reduced hardening — AF, tensor state)
# ---------------------------------------------------------------------------

class TestReducedBlocksAF:
    def test_af_state_blocks_are_none(self, af_model):
        result, state0 = _plastic_result(af_model)
        jac = ad_jacobian_blocks(af_model, result, state0)
        # AF is reduced hardening — state blocks should be None
        assert jac.dstate_dsigma is None
        assert jac.dstate_dstate is None


# ---------------------------------------------------------------------------
# Structure tests (augmented hardening — OW)
# ---------------------------------------------------------------------------

class TestAugmentedBlocks:
    def test_returns_jacobian_blocks(self, ow_model):
        result, state0 = _plastic_result(ow_model)
        jac = ad_jacobian_blocks(ow_model, result, state0)
        assert isinstance(jac, JacobianBlocks)

    def test_state_block_keys(self, ow_model):
        result, state0 = _plastic_result(ow_model)
        jac = ad_jacobian_blocks(ow_model, result, state0)
        # OW has state_names = ["alpha", "ep"] -> sorted keys
        assert set(jac.dstate_dsigma.keys()) == {"alpha", "ep"}
        assert set(jac.dstate_ddlambda.keys()) == {"alpha", "ep"}
        assert set(jac.dstate_dstate.keys()) == {"alpha", "ep"}

    def test_state_block_shapes(self, ow_model):
        result, state0 = _plastic_result(ow_model)
        jac = ad_jacobian_blocks(ow_model, result, state0)
        ntens = ow_model.ntens
        # alpha: shape (ntens,) -> n_state contribution = ntens
        # ep: scalar -> n_state contribution = 1
        assert jac.dstate_dsigma["alpha"].shape == (ntens, ntens)
        assert jac.dstate_dsigma["ep"].shape == (1, ntens)
        assert jac.dstate_ddlambda["alpha"].shape == (ntens,)
        assert jac.dstate_ddlambda["ep"].shape == (1,)

    @pytest.mark.slow
    def test_full_matrix_size(self, ow_model):
        result, state0 = _plastic_result(ow_model)
        jac = ad_jacobian_blocks(ow_model, result, state0)
        ntens = ow_model.ntens
        # n_state = ntens (alpha) + 1 (ep)
        n_state = ntens + 1
        assert jac.full.shape == (ntens + 1 + n_state, ntens + 1 + n_state)

    def test_dyield_dsigma_matches_flow_direction(self, ow_model):
        result, state0 = _plastic_result(ow_model)
        jac = ad_jacobian_blocks(ow_model, result, state0)

        n_ad = autograd.grad(lambda s: ow_model.yield_function(s, result.state))(result.stress)
        np.testing.assert_allclose(
            np.array(jac.dyield_dsigma), np.array(n_ad), rtol=1e-10
        )


# ---------------------------------------------------------------------------
# ReturnMappingResult path
# ---------------------------------------------------------------------------

class TestReturnMappingResultPath:
    def test_with_explicit_stress_trial(self, model):
        deps = (lambda _a: (_a.__setitem__(0, 3e-3), _a)[1])(np.zeros(model.ntens))
        state0 = model.initial_state()
        C = model.elastic_stiffness(state0)
        st = C @ deps
        rm = PythonIntegrator(model).return_mapping(st, state0)
        jac = ad_jacobian_blocks(model, rm, state0, stress_trial=st)
        assert isinstance(jac, JacobianBlocks)
        assert jac.dstress_dsigma.shape == (model.ntens, model.ntens)

    def test_matches_stress_update_result(self, model):
        deps = (lambda _a: (_a.__setitem__(0, 3e-3), _a)[1])(np.zeros(model.ntens))
        state0 = model.initial_state()
        C = model.elastic_stiffness(state0)
        st = C @ deps
        rm = PythonIntegrator(model).return_mapping(st, state0)
        jac_rm = ad_jacobian_blocks(model, rm, state0, stress_trial=st)

        su = PythonIntegrator(model).stress_update(deps, anp.zeros(model.ntens), state0)
        jac_su = ad_jacobian_blocks(model, su, state0)

        np.testing.assert_allclose(
            np.array(jac_rm.dstress_dsigma), np.array(jac_su.dstress_dsigma), rtol=1e-10
        )

    def test_missing_stress_trial_raises(self, model):
        deps = (lambda _a: (_a.__setitem__(0, 3e-3), _a)[1])(np.zeros(model.ntens))
        state0 = model.initial_state()
        C = model.elastic_stiffness(state0)
        rm = PythonIntegrator(model).return_mapping(C @ deps, state0)
        with pytest.raises(ValueError, match="stress_trial must be provided"):
            ad_jacobian_blocks(model, rm, state0)
