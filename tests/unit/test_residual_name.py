"""Tests for StateField.residual_name / effective_residual_name and collision detection."""

import pytest

from manforge.core.state import Implicit, Explicit, NTENS, SCALAR
from manforge.core.material import MaterialModel3D
from manforge.simulation._layout import ResidualLayout


# ---------------------------------------------------------------------------
# StateField.effective_residual_name
# ---------------------------------------------------------------------------

class TestResidualNameValidation:
    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="non-empty str"):
            Implicit(shape=NTENS, residual_name="")

    def test_non_string_raises(self):
        with pytest.raises(ValueError, match="non-empty str"):
            Implicit(shape=NTENS, residual_name=123)

    def test_dlambda_residual_name_empty_string_raises(self):
        with pytest.raises(ValueError, match="non-empty str"):
            class _Bad(MaterialModel3D):
                param_names = ["E", "nu", "sigma_y0"]
                dlambda_residual_name = ""
                def __init__(self, E, nu, sigma_y0): ...
                def yield_function(self, state): ...

    def test_dlambda_residual_name_none_raises(self):
        with pytest.raises(ValueError, match="non-empty str"):
            class _Bad(MaterialModel3D):
                param_names = ["E", "nu", "sigma_y0"]
                dlambda_residual_name = None
                def __init__(self, E, nu, sigma_y0): ...
                def yield_function(self, state): ...


class TestEffectiveResidualName:
    def test_default_is_field_name(self):
        f = Implicit(shape=NTENS)
        object.__setattr__(f, "name", "alpha")
        assert f.effective_residual_name == "alpha"

    def test_explicit_residual_name(self):
        f = Implicit(shape=NTENS, residual_name="R_alpha")
        object.__setattr__(f, "name", "alpha")
        assert f.effective_residual_name == "R_alpha"

    def test_explicit_none_falls_back_to_name(self):
        f = Implicit(shape=NTENS, residual_name=None)
        object.__setattr__(f, "name", "alpha")
        assert f.effective_residual_name == "alpha"

    def test_explicit_field_carries_residual_name(self):
        f = Explicit(shape=SCALAR, residual_name="R_ep")
        object.__setattr__(f, "name", "ep")
        assert f.effective_residual_name == "R_ep"


# ---------------------------------------------------------------------------
# Collision detection in __init_subclass__
# ---------------------------------------------------------------------------

class TestCollisionDetection:
    def test_residual_name_collision_between_fields(self):
        with pytest.raises(ValueError, match="R_clash"):
            class _Bad(MaterialModel3D):
                param_names = ["E", "nu", "sigma_y0"]
                alpha = Implicit(shape=NTENS, residual_name="R_clash")
                ep    = Implicit(shape=SCALAR, residual_name="R_clash")

                def __init__(self, E, nu, sigma_y0): ...
                def yield_function(self, state): ...
                def state_residual(self, s, d, sn, st, *, stress_trial): ...

    def test_residual_name_collides_with_other_state_name(self):
        with pytest.raises(ValueError, match="collides"):
            class _Bad2(MaterialModel3D):
                param_names = ["E", "nu", "sigma_y0"]
                alpha = Implicit(shape=NTENS, residual_name="ep")  # clashes with ep name
                ep    = Implicit(shape=SCALAR)

                def __init__(self, E, nu, sigma_y0): ...
                def yield_function(self, state): ...
                def state_residual(self, s, d, sn, st, *, stress_trial): ...

    def test_dlambda_residual_name_collides_with_residual_name(self):
        with pytest.raises(ValueError, match="R_alpha"):
            class _Bad3(MaterialModel3D):
                param_names = ["E", "nu", "sigma_y0"]
                dlambda_residual_name = "R_alpha"
                alpha = Implicit(shape=NTENS, residual_name="R_alpha")

                def __init__(self, E, nu, sigma_y0): ...
                def yield_function(self, state): ...
                def state_residual(self, s, d, sn, st, *, stress_trial): ...

    def test_dlambda_residual_name_collides_with_state_name(self):
        with pytest.raises(ValueError, match="alpha"):
            class _Bad4(MaterialModel3D):
                param_names = ["E", "nu", "sigma_y0"]
                dlambda_residual_name = "alpha"
                alpha = Implicit(shape=NTENS)

                def __init__(self, E, nu, sigma_y0): ...
                def yield_function(self, state): ...
                def state_residual(self, s, d, sn, st, *, stress_trial): ...

    def test_no_collision_is_fine(self):
        class _Ok(MaterialModel3D):
            param_names = ["E", "nu", "sigma_y0"]
            dlambda_residual_name = "R_yield"
            alpha = Implicit(shape=NTENS, residual_name="R_alpha")
            ep    = Implicit(shape=SCALAR, residual_name="R_ep")

            def __init__(self, E, nu, sigma_y0): ...
            def yield_function(self, state): ...
            def state_residual(self, s, d, sn, st, *, stress_trial): ...

        assert _Ok.state_fields["alpha"].effective_residual_name == "R_alpha"
        assert _Ok.dlambda_residual_name == "R_yield"


# ---------------------------------------------------------------------------
# ResidualLayout.residual_names / residual_name_for
# ---------------------------------------------------------------------------

class TestResidualLayout:
    def _make_default_model(self):
        from manforge.models.ow_kinematic import OWKinematic3D
        return OWKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=50000.0, gamma=300.0)

    def test_default_residual_names_equal_state_names(self):
        model = self._make_default_model()
        layout = ResidualLayout.from_model(model)
        # Default: each residual name == state name
        assert layout.residual_names() == ("stress", "dlambda", "alpha", "ep")

    def test_custom_residual_names(self):
        class _Custom(MaterialModel3D):
            param_names = ["E", "nu", "sigma_y0"]
            dlambda_residual_name = "R_yield"
            alpha = Implicit(shape=NTENS, residual_name="R_alpha")
            ep    = Implicit(shape=SCALAR, residual_name="R_ep")

            def __init__(self, E, nu, sigma_y0):
                self.E = E; self.nu = nu; self.sigma_y0 = sigma_y0

            def yield_function(self, state): ...
            def state_residual(self, s, d, sn, st, *, stress_trial): ...

        model = _Custom(210000.0, 0.3, 250.0)
        layout = ResidualLayout.from_model(model)
        assert layout.residual_names() == ("stress", "R_yield", "R_alpha", "R_ep")
        assert layout.residual_name_for("alpha") == "R_alpha"
        assert layout.residual_name_for("dlambda") == "R_yield"
        assert layout.residual_name_for("stress") == "stress"
