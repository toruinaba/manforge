"""Tests for JacobianBlocks and ad_jacobian_blocks."""

import autograd
import autograd.numpy as anp
import numpy as np
import pytest

from manforge.verification.jacobian import ad_jacobian_blocks, JacobianBlocks
from manforge.simulation.integrator import PythonIntegrator
from manforge.models.af_kinematic import AFKinematic3D
from manforge.models.ow_kinematic import OWKinematic3D
from manforge.core import Implicit, NTENS, SCALAR
from manforge.core.material import MaterialModel3D


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
        assert jac.part["stress"]["stress"].shape == (ntens, ntens)
        assert jac.part["stress"]["dlambda"].shape == (ntens,)
        assert jac.part["dlambda"]["stress"].shape == (ntens,)
        assert jac.full.shape == (ntens + 1, ntens + 1)

    def test_state_blocks_absent_for_j2(self, model):
        result, state0 = _plastic_result(model)
        jac = ad_jacobian_blocks(model, result, state0)
        # J2 has no implicit non-stress states; only stress and dlambda rows
        assert set(jac.row_names()) == {"stress", "dlambda"}
        assert set(jac.col_names()) == {"stress", "dlambda"}

    def test_part_dlambda_stress_matches_flow_direction(self, model):
        result, state0 = _plastic_result(model)
        jac = ad_jacobian_blocks(model, result, state0)

        from manforge.core.state import _state_with_stress
        n_ad = autograd.grad(lambda s: model.yield_function(_state_with_stress(result.state, s)))(result.stress)
        np.testing.assert_allclose(
            np.array(jac.part["dlambda"]["stress"]), np.array(n_ad), rtol=1e-10
        )

    def test_full_matrix_blocks_consistent(self, model):
        result, state0 = _plastic_result(model)
        jac = ad_jacobian_blocks(model, result, state0)
        layout = jac.layout
        ntens = model.ntens

        np.testing.assert_array_equal(
            jac.part["stress"]["stress"].reshape(ntens, ntens),
            jac.full[layout.slot_slice("stress"), layout.slot_slice("stress")],
        )
        np.testing.assert_array_equal(
            jac.part["stress"]["dlambda"].reshape(ntens),
            jac.full[layout.slot_slice("stress"), layout.slot_slice("dlambda")].reshape(ntens),
        )
        np.testing.assert_array_equal(
            jac.part["dlambda"]["stress"].reshape(ntens),
            jac.full[layout.slot_slice("dlambda"), layout.slot_slice("stress")].reshape(ntens),
        )

    def test_elastic_step_raises_no_error(self, model):
        deps = anp.array([1e-4, 0, 0, 0, 0, 0])
        state0 = model.initial_state()
        result = PythonIntegrator(model).stress_update(deps, anp.zeros(6), state0)
        jac = ad_jacobian_blocks(model, result, state0)
        assert isinstance(jac, JacobianBlocks)


# ---------------------------------------------------------------------------
# Structure tests (reduced hardening — AF, tensor state)
# ---------------------------------------------------------------------------

class TestReducedBlocksAF:
    def test_af_state_blocks_are_empty_dicts(self, af_model):
        result, state0 = _plastic_result(af_model)
        jac = ad_jacobian_blocks(af_model, result, state0)
        # AF is reduced hardening — no implicit non-stress states
        assert set(jac.row_names()) == {"stress", "dlambda"}
        assert set(jac.col_names()) == {"stress", "dlambda"}


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
        # OW has implicit_keys = ["alpha", "ep"] (declaration order)
        assert set(jac.row_names()) == {"stress", "dlambda", "alpha", "ep"}
        assert set(jac.col_names()) == {"stress", "dlambda", "alpha", "ep"}

    def test_state_block_shapes(self, ow_model):
        result, state0 = _plastic_result(ow_model)
        jac = ad_jacobian_blocks(ow_model, result, state0)
        ntens = ow_model.ntens
        # alpha: NTENS shape; ep: SCALAR shape
        assert jac.part["alpha"]["stress"].shape == (ntens, ntens)
        assert jac.part["alpha"]["dlambda"].shape == (ntens,)
        assert jac.part["ep"]["stress"].shape == (ntens,)
        assert jac.part["ep"]["dlambda"].shape == ()

    @pytest.mark.slow
    def test_full_matrix_size(self, ow_model):
        result, state0 = _plastic_result(ow_model)
        jac = ad_jacobian_blocks(ow_model, result, state0)
        ntens = ow_model.ntens
        n_state = ntens + 1  # alpha (ntens) + ep (1)
        assert jac.full.shape == (ntens + 1 + n_state, ntens + 1 + n_state)

    def test_part_dlambda_stress_matches_flow_direction(self, ow_model):
        result, state0 = _plastic_result(ow_model)
        jac = ad_jacobian_blocks(ow_model, result, state0)

        from manforge.core.state import _state_with_stress
        n_ad = autograd.grad(lambda s: ow_model.yield_function(_state_with_stress(result.state, s)))(result.stress)
        np.testing.assert_allclose(
            np.array(jac.part["dlambda"]["stress"]), np.array(n_ad), rtol=1e-10
        )

    def test_part_shapes_match_slot_shape(self, ow_model):
        result, state0 = _plastic_result(ow_model)
        jac = ad_jacobian_blocks(ow_model, result, state0)
        layout = jac.layout
        col_names = jac.col_names()
        for col_state in col_names:
            row = layout.residual_name_for(col_state)
            shp_row = layout.slot_shape(col_state)
            for col in col_names:
                shp_col = layout.slot_shape(col)
                expected = shp_row + shp_col
                actual = np.asarray(jac.part[row][col]).shape
                assert actual == expected, (
                    f"part[{row!r}][{col!r}].shape={actual}, expected {expected}"
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
        assert jac.part["stress"]["stress"].shape == (model.ntens, model.ntens)

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
            np.array(jac_rm.part["stress"]["stress"]),
            np.array(jac_su.part["stress"]["stress"]),
            rtol=1e-10,
        )

    def test_missing_stress_trial_raises(self, model):
        deps = (lambda _a: (_a.__setitem__(0, 3e-3), _a)[1])(np.zeros(model.ntens))
        state0 = model.initial_state()
        C = model.elastic_stiffness(state0)
        rm = PythonIntegrator(model).return_mapping(C @ deps, state0)
        with pytest.raises(ValueError, match="stress_trial must be provided"):
            ad_jacobian_blocks(model, rm, state0)


# ---------------------------------------------------------------------------
# residual_name opt-in
# ---------------------------------------------------------------------------

class _ResidualNameModel(MaterialModel3D):
    param_names = ["E", "nu", "sigma_y0", "H"]
    dlambda_residual_name = "R_yield"
    stress = Implicit(shape=NTENS, doc="Cauchy stress")
    alpha  = Implicit(shape=NTENS, doc="backstress", residual_name="R_alpha")
    ep     = Implicit(shape=SCALAR, doc="eqv. plastic strain", residual_name="R_ep")

    def __init__(self, E, nu, sigma_y0, H):
        self.E = E; self.nu = nu; self.sigma_y0 = sigma_y0; self.H = H

    def yield_function(self, state):
        return self._vonmises(state["stress"] - state["alpha"]) - (self.sigma_y0 + self.H * state["ep"])

    def state_residual(self, state_new, dlambda, state_n, state_trial, *, stress_trial):
        import autograd as _ag
        from manforge.core.state import _state_with_stress
        stress = state_new["stress"]
        C = self.elastic_stiffness(state_new)
        n = _ag.grad(lambda s: self.yield_function(_state_with_stress(state_new, s)))(stress)
        R_alpha = state_new["alpha"] - state_n["alpha"] - dlambda * n
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        R_stress = self.default_stress_residual(state_new, dlambda, stress_trial)
        return [self.stress(R_stress), self.alpha(R_alpha), self.ep(R_ep)]


class TestResidualNameOptIn:
    @pytest.fixture
    def rn_model(self):
        return _ResidualNameModel(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)

    def test_row_names_use_residual_names(self, rn_model):
        result, state0 = _plastic_result(rn_model)
        jac = ad_jacobian_blocks(rn_model, result, state0)
        assert set(jac.row_names()) == {"stress", "R_yield", "R_alpha", "R_ep"}

    def test_col_names_use_state_names(self, rn_model):
        result, state0 = _plastic_result(rn_model)
        jac = ad_jacobian_blocks(rn_model, result, state0)
        assert set(jac.col_names()) == {"stress", "dlambda", "alpha", "ep"}

    def test_custom_row_access(self, rn_model):
        result, state0 = _plastic_result(rn_model)
        jac = ad_jacobian_blocks(rn_model, result, state0)
        # custom residual_name rows exist
        assert jac.part["R_alpha"]["stress"].shape == (rn_model.ntens, rn_model.ntens)
        assert jac.part["R_yield"]["stress"].shape == (rn_model.ntens,)
        # default state name rows do NOT exist
        assert "alpha" not in jac.part
        assert "dlambda" not in jac.part
