"""Unit tests for ResidualLayout (simulation/_layout.py).

Absorbed from tests/unit/test_residual_name.py (layout part).
"""

import pytest
from manforge.core.state import Implicit, Explicit, NTENS, SCALAR
from manforge.core.material import MaterialModel
from manforge.simulation._layout import ResidualLayout


class TestResidualLayout:
    def _make_default_model(self):
        from manforge.models.ow_kinematic import OWKinematic3D
        return OWKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=50000.0, gamma=300.0)

    def test_default_residual_names_equal_state_names(self):
        model = self._make_default_model()
        layout = ResidualLayout.from_model(model)
        assert layout.residual_names() == ("stress", "dlambda", "alpha", "ep")

    def test_custom_residual_names(self):
        class _Custom(MaterialModel):
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
