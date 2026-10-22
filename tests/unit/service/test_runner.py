"""Tests for run_simulation input validation and output contract (pure logic).

Driver-in-the-loop behaviour (actual integration, non-convergence, stress
control) is covered in tests/integration/service/test_runner.py.
"""

import numpy as np
import pytest

from manforge.models import J2Isotropic1D
from manforge.service import run_simulation

PARAMS = dict(E=206e3, nu=0.3, sigma_y0=250.0, H=1000.0)
LOAD = np.linspace(0.0, 1e-3, 5)


class TestParamValidation:
    def test_missing_params_listed(self):
        with pytest.raises(ValueError, match=r"missing \['H', 'nu', 'sigma_y0'\]"):
            run_simulation(J2Isotropic1D, {"E": 206e3}, LOAD)

    def test_extra_params_listed(self):
        bad = dict(PARAMS, bogus=1.0)
        with pytest.raises(ValueError, match=r"unexpected \['bogus'\]"):
            run_simulation(J2Isotropic1D, bad, LOAD)

    def test_missing_and_extra_both_listed(self):
        with pytest.raises(ValueError, match=r"missing.*unexpected"):
            run_simulation(J2Isotropic1D, {"E": 206e3, "bogus": 1.0}, LOAD)


class TestFieldValidation:
    def test_unknown_field_lists_available(self):
        with pytest.raises(ValueError, match=r"\['nope'\].*available.*ep"):
            run_simulation(J2Isotropic1D, PARAMS, LOAD, include_fields=["nope"])


class TestLoadValidation:
    def test_3d_load_raises(self):
        with pytest.raises(ValueError, match="1-D or 2-D"):
            run_simulation(J2Isotropic1D, PARAMS, np.zeros((3, 1, 1)))

    def test_ntens_mismatch_raises(self):
        with pytest.raises(ValueError, match="ntens=1"):
            run_simulation(J2Isotropic1D, PARAMS, np.zeros((5, 6)))

    def test_unknown_control_raises(self):
        with pytest.raises(ValueError, match="control"):
            run_simulation(J2Isotropic1D, PARAMS, LOAD, control="mixed")


class TestOutputContract:
    def test_plain_list_input_accepted(self):
        out = run_simulation(J2Isotropic1D, PARAMS, [0.0, 5e-4, 1e-3])
        assert out.converged
        assert out.n_steps == 3

    def test_shapes_stay_2d_for_1d_model(self):
        out = run_simulation(J2Isotropic1D, PARAMS, LOAD)
        assert out.stress.shape == (5, 1)
        assert out.strain.shape == (5, 1)

    def test_fields_empty_by_default(self):
        out = run_simulation(J2Isotropic1D, PARAMS, LOAD)
        assert out.fields == {}

    def test_include_fields_selected_only(self):
        out = run_simulation(J2Isotropic1D, PARAMS, LOAD, include_fields=["ep"])
        assert set(out.fields) == {"ep"}
        assert out.fields["ep"].shape == (5,)

    def test_progress_callback_called_per_step(self):
        calls = []
        run_simulation(J2Isotropic1D, PARAMS, LOAD,
                       progress=lambda done, total: calls.append((done, total)))
        assert calls == [(i + 1, 5) for i in range(5)]
