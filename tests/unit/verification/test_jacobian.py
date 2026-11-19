"""Unit tests for JacobianBlocks structure and ``integrator.jacobian``.

Pure structure/shape tests that build a StressUpdateResult directly.
Integrator-driven compare tests live in tests/integration/verification/test_jacobian.py.

Absorbed from tests/unit/test_jacobian_blocks.py.
"""

import autograd
import autograd.numpy as anp
import numpy as np
import pytest

from manforge.simulation import JacobianBlocks, PythonIntegrator
from manforge.models import AFKinematic3D, OWKinematic3D
from manforge.core import Implicit, NTENS, SCALAR, MaterialModel


@pytest.fixture
def af_model():
    return AFKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=50000.0, gamma=300.0)


@pytest.fixture
def ow_model():
    return OWKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=50000.0, gamma=300.0)


def _plastic_result(model, strain_scale=3e-3):
    deps = np.zeros(model.ntens); deps[0] = strain_scale
    state0 = model.initial_state()
    return PythonIntegrator(model).stress_update(deps, anp.zeros(model.ntens), state0), state0


# ---------------------------------------------------------------------------
# Structure tests (reduced hardening — J2)
# ---------------------------------------------------------------------------

class TestReducedBlocks:
    def test_returns_jacobian_blocks(self, model):
        result, state0 = _plastic_result(model)
        jac = PythonIntegrator(model).jacobian(result)
        assert isinstance(jac, JacobianBlocks)

    def test_fixed_block_shapes(self, model):
        result, state0 = _plastic_result(model)
        jac = PythonIntegrator(model).jacobian(result)
        ntens = model.ntens
        assert jac.part["stress"]["stress"].shape == (ntens, ntens)
        assert jac.part["stress"]["dlambda"].shape == (ntens,)
        assert jac.part["dlambda"]["stress"].shape == (ntens,)
        assert jac.full.shape == (ntens + 1, ntens + 1)

    def test_state_blocks_absent_for_j2(self, model):
        result, state0 = _plastic_result(model)
        jac = PythonIntegrator(model).jacobian(result)
        assert set(jac.row_names()) == {"stress", "dlambda"}
        assert set(jac.col_names()) == {"stress", "dlambda"}

    def test_part_dlambda_stress_matches_flow_direction(self, model):
        result, state0 = _plastic_result(model)
        jac = PythonIntegrator(model).jacobian(result)
        from manforge.core.state import _state_with_stress
        n_ad = autograd.grad(
            lambda s: model.yield_function(_state_with_stress(result.state, s))
        )(result.stress)
        np.testing.assert_allclose(
            np.array(jac.part["dlambda"]["stress"]), np.array(n_ad), rtol=1e-10
        )

    def test_full_matrix_blocks_consistent(self, model):
        result, state0 = _plastic_result(model)
        jac = PythonIntegrator(model).jacobian(result)
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

    def test_elastic_step_raises_no_error(self, model):
        deps = anp.array([1e-4, 0, 0, 0, 0, 0])
        state0 = model.initial_state()
        result = PythonIntegrator(model).stress_update(deps, anp.zeros(6), state0)
        jac = PythonIntegrator(model).jacobian(result)
        assert isinstance(jac, JacobianBlocks)


# ---------------------------------------------------------------------------
# Structure tests (reduced hardening — AF)
# ---------------------------------------------------------------------------

class TestReducedBlocksAF:
    def test_af_state_blocks_are_empty_dicts(self, af_model):
        result, state0 = _plastic_result(af_model)
        jac = PythonIntegrator(af_model).jacobian(result)
        assert set(jac.row_names()) == {"stress", "dlambda"}
        assert set(jac.col_names()) == {"stress", "dlambda"}


# ---------------------------------------------------------------------------
# Structure tests (augmented hardening — OW)
# ---------------------------------------------------------------------------

class TestAugmentedBlocks:
    def test_returns_jacobian_blocks(self, ow_model):
        result, state0 = _plastic_result(ow_model)
        jac = PythonIntegrator(ow_model).jacobian(result)
        assert isinstance(jac, JacobianBlocks)

    def test_state_block_keys(self, ow_model):
        result, state0 = _plastic_result(ow_model)
        jac = PythonIntegrator(ow_model).jacobian(result)
        assert set(jac.row_names()) == {"stress", "dlambda", "alpha", "ep"}
        assert set(jac.col_names()) == {"stress", "dlambda", "alpha", "ep"}

    def test_state_block_shapes(self, ow_model):
        result, state0 = _plastic_result(ow_model)
        jac = PythonIntegrator(ow_model).jacobian(result)
        ntens = ow_model.ntens
        assert jac.part["alpha"]["stress"].shape == (ntens, ntens)
        assert jac.part["alpha"]["dlambda"].shape == (ntens,)
        assert jac.part["ep"]["stress"].shape == (ntens,)
        assert jac.part["ep"]["dlambda"].shape == ()

    @pytest.mark.slow
    def test_full_matrix_size(self, ow_model):
        result, state0 = _plastic_result(ow_model)
        jac = PythonIntegrator(ow_model).jacobian(result)
        ntens = ow_model.ntens
        n_state = ntens + 1
        assert jac.full.shape == (ntens + 1 + n_state, ntens + 1 + n_state)

    def test_part_shapes_match_slot_shape(self, ow_model):
        result, state0 = _plastic_result(ow_model)
        jac = PythonIntegrator(ow_model).jacobian(result)
        layout = jac.layout
        col_names = jac.col_names()
        for col_state in col_names:
            row = layout.residual_name_for(col_state)
            shp_row = layout.slot_shape(col_state)
            for col in col_names:
                shp_col = layout.slot_shape(col)
                expected = shp_row + shp_col
                actual = np.asarray(jac.part[row][col]).shape
                assert actual == expected


# ---------------------------------------------------------------------------
# residual_name opt-in
# ---------------------------------------------------------------------------

class _ResidualNameModel(MaterialModel):
    param_names = ["E", "nu", "sigma_y0", "H"]
    dlambda_residual_name = "R_yield"
    stress = Implicit(shape=NTENS, doc="Cauchy stress")
    alpha  = Implicit(shape=NTENS, doc="backstress", residual_name="R_alpha")
    ep     = Implicit(shape=SCALAR, doc="eqv. plastic strain", residual_name="R_ep")

    def yield_function(self, state):
        return self.vonmises(state["stress"] - state["alpha"]) - (self.sigma_y0 + self.H * state["ep"])

    def state_residual(self, new, dlambda, inc):
        import autograd as _ag
        from manforge.core.state import _state_with_stress
        stress = new["stress"]
        n = _ag.grad(lambda s: self.yield_function(_state_with_stress(new, s)))(stress)
        R_alpha = new["alpha"] - inc.state_n["alpha"] - dlambda * n
        R_ep = new["ep"] - inc.state_n["ep"] - dlambda
        R_stress = self.default_stress_residual(new, dlambda, inc.trial["stress"])
        return [self.stress(R_stress), self.alpha(R_alpha), self.ep(R_ep)]


class TestResidualNameOptIn:
    @pytest.fixture
    def rn_model(self):
        return _ResidualNameModel(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)

    def test_row_names_use_residual_names(self, rn_model):
        result, state0 = _plastic_result(rn_model)
        jac = PythonIntegrator(rn_model).jacobian(result)
        assert set(jac.row_names()) == {"stress", "R_yield", "R_alpha", "R_ep"}

    def test_col_names_use_state_names(self, rn_model):
        result, state0 = _plastic_result(rn_model)
        jac = PythonIntegrator(rn_model).jacobian(result)
        assert set(jac.col_names()) == {"stress", "dlambda", "alpha", "ep"}

    def test_custom_row_access(self, rn_model):
        result, state0 = _plastic_result(rn_model)
        jac = PythonIntegrator(rn_model).jacobian(result)
        assert jac.part["R_alpha"]["stress"].shape == (rn_model.ntens, rn_model.ntens)
        assert jac.part["R_yield"]["stress"].shape == (rn_model.ntens,)
        assert "alpha" not in jac.part
        assert "dlambda" not in jac.part


# ---------------------------------------------------------------------------
# What a block's error is relative to
# ---------------------------------------------------------------------------

class TestBlockScale:
    """``_block_scale(own, row, col)``: the block's own size, unless it is round-off."""

    def test_a_block_with_a_size_of_its_own_is_measured_against_it(self):
        from manforge.verification.jacobian import _block_scale

        # Small next to what its row implies (1e-7 of it), but far above round-off.
        assert _block_scale(own=4e-7, row=1.0, col=0.25) == 4e-7
        assert _block_scale(own=1.6e5, row=1e3, col=2e-4) == 1.6e5

    def test_round_off_is_measured_against_what_row_and_column_imply(self):
        from manforge.verification.jacobian import _block_scale

        # 1.4e-13 against 700 / 0.67: autodiff noise in a block that is zero.
        assert _block_scale(own=1.4e-13, row=700.0, col=0.67) == pytest.approx(700.0 / 0.67)
        assert _block_scale(own=0.0, row=700.0, col=0.67) == pytest.approx(700.0 / 0.67)

    def test_an_unknown_that_never_moved_implies_nothing(self):
        from manforge.verification.jacobian import _block_scale

        assert _block_scale(own=3.0, row=10.0, col=0.0) == 3.0
        assert _block_scale(own=0.0, row=10.0, col=0.0) == 0.0
        assert _block_scale(own=3.0, row=float("nan"), col=1.0) == 3.0

    @pytest.mark.parametrize("unit", [1e-9, 1.0, 1e6])
    def test_the_choice_does_not_depend_on_the_units(self, unit):
        """Row sizes and blocks scale together; the ratio that decides does not."""
        from manforge.verification.jacobian import _block_scale

        assert _block_scale(own=4e-7 * unit, row=1.0 * unit, col=0.25) == 4e-7 * unit
        assert _block_scale(own=1e-13 * unit, row=1.0 * unit, col=0.25) == pytest.approx(4.0 * unit)
