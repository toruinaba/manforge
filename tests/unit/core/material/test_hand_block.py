"""hand_block: declaring hand-derived Jacobian blocks on a model (no integrator)."""

import numpy as np
import pytest

from manforge.core.material import HandBlock, hand_block, verified_against_fortran
from manforge.models import J2Isotropic3D, OWKinematic3D
from tests.fixtures.hand_block_models import (
    OW_PARAMS, OWWithApproxBlock, OWWithHandBlocks, OWWithWrongBlock,
)


# ---------------------------------------------------------------------------
# The decorator
# ---------------------------------------------------------------------------

def test_decorator_only_attaches_a_record():
    """The method stays an ordinary method, callable directly."""
    m = OWWithHandBlocks(**OW_PARAMS)
    assert m.dRep_dep() == 1.0
    assert OWWithHandBlocks.dRep_dep.__hand_block__ == HandBlock("ep", "ep")


def test_approx_needs_a_reason():
    """A tolerance without its reason reads like a bug that was waved through."""
    with pytest.raises(ValueError, match="needs a note"):
        hand_block("ep", "ep", approx=1e-2)
    with pytest.raises(ValueError, match="needs a note"):
        hand_block("ep", "ep", approx=1e-2, note="   ")


def test_approx_must_be_positive():
    with pytest.raises(ValueError, match="approx must be > 0"):
        hand_block("ep", "ep", approx=0.0, note="x")


def test_stacks_with_verified_against_fortran():
    class _Both(OWKinematic3D):
        @hand_block("ep", "ep")
        @verified_against_fortran("ow_dRep_dep")
        def dRep_dep(self):
            return 1.0

    assert _Both._hand_blocks == {("ep", "ep"): "dRep_dep"}
    assert _Both._fortran_bindings["dRep_dep"].subroutine == "ow_dRep_dep"


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def test_blocks_are_collected_by_row_and_column():
    blocks = OWWithHandBlocks._hand_blocks
    assert len(blocks) == 8
    assert blocks[("dlambda", "stress")] == "dRy_ds"
    assert blocks[("ep", "alpha")] == "dRep_da"


def test_model_without_declarations_has_none():
    assert J2Isotropic3D._hand_blocks == {}
    assert OWKinematic3D._hand_blocks == {}


def test_subclass_redeclaration_keeps_one_entry_per_block():
    assert len(OWWithWrongBlock._hand_blocks) == 8
    assert OWWithWrongBlock._hand_blocks[("ep", "dlambda")] == "dRep_dl"
    assert OWWithWrongBlock(**OW_PARAMS).dRep_dl(0.0) == +1.0
    assert OWWithHandBlocks(**OW_PARAMS).dRep_dl(0.0) == -1.0


def test_override_without_the_decorator_is_still_the_block():
    class _Plain(OWWithHandBlocks):
        def dRep_dep(self):
            return 7.0

    assert _Plain._hand_blocks[("ep", "ep")] == "dRep_dep"
    assert _Plain(**OW_PARAMS).dRep_dep() == 7.0


# ---------------------------------------------------------------------------
# Checked when the class is created
# ---------------------------------------------------------------------------

def test_unknown_row_is_rejected():
    with pytest.raises(TypeError, match=r"row 'beta' is not a residual row"):
        class _Bad(OWKinematic3D):
            @hand_block("beta", "stress")
            def f(self, state):
                return np.zeros(6)


def test_unknown_column_is_rejected():
    with pytest.raises(TypeError, match=r"column 'beta' is not an unknown"):
        class _Bad(OWKinematic3D):
            @hand_block("stress", "beta")
            def f(self, state):
                return np.zeros(6)


def test_explicit_state_is_not_a_column():
    """Only NR unknowns have columns; an explicit state is not one."""
    with pytest.raises(TypeError, match="column 'ep' is not an unknown"):
        class _Bad(J2Isotropic3D):
            @hand_block("stress", "ep")
            def f(self, state):
                return np.zeros(6)


def test_argument_the_framework_cannot_supply_is_rejected():
    with pytest.raises(TypeError, match=r"takes \['stress_trial'\]"):
        class _Bad(OWKinematic3D):
            @hand_block("ep", "ep")
            def f(self, state, stress_trial):
                return 1.0


def test_two_methods_for_one_block_is_rejected():
    with pytest.raises(TypeError, match=r"\[ep\]\[ep\] is declared by both"):
        class _Bad(OWKinematic3D):
            @hand_block("ep", "ep")
            def first(self):
                return 1.0

            @hand_block("ep", "ep")
            def second(self):
                return 1.0


def test_custom_residual_names_are_the_row_names():
    from manforge.core.material import MaterialModel
    from manforge.core.state import Implicit, NTENS

    class _Named(MaterialModel):
        param_names = ["E", "nu"]
        dlambda_residual_name = "R_yield"
        stress = Implicit(shape=NTENS)
        alpha = Implicit(shape=NTENS, residual_name="R_alpha")

        def yield_function(self, state): ...
        def state_residual(self, new, dlambda, inc): ...

        @hand_block("R_alpha", "alpha")
        def dRa_da(self, state):
            return np.eye(6)

        @hand_block("R_yield", "dlambda")
        def dRy_dl(self):
            return 0.0

    assert set(_Named._hand_blocks) == {("R_alpha", "alpha"), ("R_yield", "dlambda")}

    with pytest.raises(TypeError, match="row 'alpha' is not a residual row"):
        class _Bad(_Named):
            @hand_block("alpha", "alpha")       # the row is R_alpha
            def other(self, state):
                return np.eye(6)


def test_fixture_variants_declare_what_they_say():
    assert OWWithApproxBlock.dRep_dep.__hand_block__.approx == 1e-2
    assert "1e-3" in OWWithApproxBlock.dRep_dep.__hand_block__.note
