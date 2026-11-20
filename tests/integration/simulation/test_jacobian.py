"""simulation/jacobian.py: the residual Jacobian, autodiff and hand-derived."""

import numpy as np
import pytest

from returnmap.simulation import (
    StrainDriver, PythonNumericalIntegrator, JacobianBlocks, FieldHistory, FieldType,
    hand_jacobian,
)
from returnmap.simulation.jacobian import hand_jacobian_blocks, residual_jacobian
from tests.fixtures.hand_block_models import OW_PARAMS, OWWithHandBlocks


def _multiaxial(n=12):
    """Direct and shear components together — where convention errors show."""
    t = np.linspace(0.0, 1.0, n + 1)[1:]
    data = np.zeros((n, 6))
    data[:, 0] = 4e-3 * t
    data[:, 3] = 6e-3 * np.sin(np.pi * t)
    return FieldHistory(FieldType.STRAIN, data)


@pytest.fixture
def model():
    return OWWithHandBlocks(**OW_PARAMS)


@pytest.fixture
def plastic_result(model):
    steps = StrainDriver(PythonNumericalIntegrator(model)).run(_multiaxial()).step_results
    result = steps[-1]
    assert result.is_plastic
    return result


def test_integrator_jacobian_is_the_autodiff_residual_jacobian(model, plastic_result):
    integrator = PythonNumericalIntegrator(model)
    jac = integrator.jacobian(plastic_result)
    ref = residual_jacobian(model, plastic_result, plastic_result.increment)
    assert isinstance(jac, JacobianBlocks)
    np.testing.assert_array_equal(np.asarray(jac.full), np.asarray(ref.full))
    assert jac.part["alpha"]["stress"].shape == (6, 6)
    assert {s for row in jac.source.values() for s in row.values()} == {"ad"}


def test_integrator_jacobian_rejects_a_result_without_increment(model, plastic_result):
    from dataclasses import replace
    with pytest.raises(ValueError, match="no increment"):
        PythonNumericalIntegrator(model).jacobian(replace(plastic_result, increment=None))


def test_hand_blocks_are_shaped_like_the_reference(model, plastic_result):
    inc = plastic_result.increment
    blocks = hand_jacobian_blocks(model, plastic_result.state, plastic_result.dlambda, inc)
    ref = residual_jacobian(model, plastic_result, inc)

    assert set(blocks) == set(type(model)._hand_blocks)
    for (row, col), value in blocks.items():
        assert value.shape == np.asarray(ref.part[row][col]).shape, (row, col)
    assert blocks[("ep", "ep")].shape == ()          # a scalar return becomes 0-d
    assert blocks[("dlambda", "stress")].shape == (6,)


def test_arguments_are_filled_by_name(model, plastic_result):
    inc = plastic_result.increment
    hand_jacobian_blocks(model, plastic_result.state, plastic_result.dlambda, inc)
    assert model.seen["inc"] is inc
    assert model.seen["state_n"] is inc.state_n
    assert float(model.seen["dlambda"]) == float(plastic_result.dlambda)


def test_hand_jacobian_marks_where_each_block_came_from(model, plastic_result):
    inc = plastic_result.increment
    hand = hand_jacobian(model, plastic_result)
    ref = residual_jacobian(model, plastic_result, inc)

    declared = set(type(model)._hand_blocks)
    for row in hand.row_names():
        for col in hand.col_names():
            expected = "hand" if (row, col) in declared else "ad"
            assert hand.source[row][col] == expected, (row, col)
            if expected == "ad":
                np.testing.assert_array_equal(
                    np.asarray(hand.part[row][col]), np.asarray(ref.part[row][col]))
    # The declared blocks are exact here, so the assembled matrix is the reference.
    np.testing.assert_allclose(np.asarray(hand.full), np.asarray(ref.full), rtol=1e-10, atol=1e-9)


def test_wrongly_sized_block_is_reported_by_name(plastic_result):
    from returnmap.core import hand_block

    class _Misshapen(OWWithHandBlocks):
        @hand_block("ep", "stress")
        def dRep_ds(self, state):
            return np.zeros(3)                       # ntens is 6

    m = _Misshapen(**OW_PARAMS)
    with pytest.raises(ValueError, match=r"_Misshapen\.dRep_ds.*\[ep\]\[stress\].*expected \(6,\)"):
        hand_jacobian_blocks(m, plastic_result.state, plastic_result.dlambda,
                             plastic_result.increment)
