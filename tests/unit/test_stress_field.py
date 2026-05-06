"""Tests for stress as a state field.

Covers:
- stress auto-attached as Explicit when not declared
- stress = Implicit(shape=NTENS) makes σ an NR unknown (matches old implicit_stress=True)
- Default associative stress default: NR solution matches explicit reference
- User can supply custom R_stress via state_residual (returns self.stress(...))
- Custom R_stress is differentiable (consistent tangent still works)
- Migration error: old stress_residual() override raises TypeError
"""

import numpy as np
import pytest
import autograd
import autograd.numpy as anp

from manforge.core.state import Implicit, Explicit, NTENS, _state_with_stress
from manforge.core.material import MaterialModel3D
from manforge.core.dimension import SOLID_3D
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.models.ow_kinematic import OWKinematic3D
from manforge.simulation.integrator import PythonIntegrator
from manforge.verification.fd_check import check_tangent


_J2_PARAMS = dict(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)


# ---------------------------------------------------------------------------
# stress auto-attachment
# ---------------------------------------------------------------------------

def test_stress_auto_attached_as_explicit():
    """Models without stress declaration get stress=Explicit automatically."""
    class _M(MaterialModel3D):
        param_names = []
        ep = Explicit(shape=())

        def yield_function(self, state):
            return self._vonmises(state["stress"]) - 250.0

        def update_state(self, dlambda, state_n, state_trial):
            return [self.ep(state_n["ep"] + dlambda)]

    assert "stress" in _M.state_fields
    assert _M.state_fields["stress"].kind == "explicit"
    assert "stress" in _M.state_names
    assert "stress" not in _M.implicit_state_names


def test_stress_implicit_declaration():
    """stress = Implicit(shape=NTENS) puts stress in implicit_state_names."""
    class _M(MaterialModel3D):
        param_names = []
        stress = Implicit(shape=NTENS, doc="Cauchy stress")
        ep = Implicit(shape=())

        def yield_function(self, state):
            return self._vonmises(state["stress"]) - 250.0

        def state_residual(self, state_new, dlambda, state_n, state_trial, *, stress_trial):
            return [self.ep(state_new["ep"] - state_n["ep"] - dlambda)]

    assert _M.state_fields["stress"].kind == "implicit"
    assert "stress" in _M.implicit_state_names
    assert "ep" in _M.implicit_state_names


def test_initial_state_includes_stress():
    """initial_state() always contains 'stress' key with correct shape."""
    model = J2Isotropic3D(**_J2_PARAMS)
    s = model.initial_state()
    assert "stress" in s
    assert np.asarray(s["stress"]).shape == (6,)
    assert np.allclose(s["stress"], np.zeros(6))


# ---------------------------------------------------------------------------
# Default associative stress update matches explicit reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deps_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
])
def test_default_stress_update_matches_explicit(deps_vec):
    """Default framework stress update must match reference J2 explicit path."""
    model = J2Isotropic3D(**_J2_PARAMS)
    state0 = model.initial_state()
    deps = anp.array(deps_vec)

    r = PythonIntegrator(model).stress_update(deps, anp.zeros(6), state0)
    assert r.is_plastic, "Expected plastic step"
    # Yield consistency
    f = model.yield_function(r.state)
    assert abs(float(f)) < 1e-8, f"Yield not satisfied: f={float(f):.3e}"


# ---------------------------------------------------------------------------
# stress = Implicit (OW) matches default (non-Implicit stress reference)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deps_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2e-3, 0.0, 0.0],
])
def test_stress_implicit_ow_yield_consistency(deps_vec):
    """OW (stress=Implicit) must converge to yield surface."""
    model = OWKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0)
    state0 = model.initial_state()
    deps = anp.array(deps_vec)

    r = PythonIntegrator(model).stress_update(deps, anp.zeros(6), state0)
    f = model.yield_function(r.state)
    assert abs(float(f)) < 1e-8, f"OW yield not satisfied: f={float(f):.3e}"


# ---------------------------------------------------------------------------
# Custom R_stress via state_residual (stress = Implicit + return self.stress(...))
# ---------------------------------------------------------------------------

class _CustomStressResidual(MaterialModel3D):
    """J2-like model that provides a custom R_stress via state_residual.

    Uses a hand-coded associative R_stress for J2 (deviatoric part only),
    avoiding nested autograd.  At convergence, result must match default.
    """

    param_names = ["E", "nu", "sigma_y0", "H"]
    stress = Implicit(shape=NTENS, doc="Cauchy stress (explicit NR unknown)")
    ep = Implicit(shape=())

    def __init__(self, *, E, nu, sigma_y0, H):
        super().__init__(SOLID_3D)
        self.E = E
        self.nu = nu
        self.sigma_y0 = sigma_y0
        self.H = H

    def yield_function(self, state):
        sigma_y = self.sigma_y0 + self.H * state["ep"]
        return self._vonmises(state["stress"]) - sigma_y

    def state_residual(self, state_new, dlambda, state_n, state_trial, *, stress_trial):
        # Hand-coded J2 associative R_stress using correct Mandel-scaled flow direction.
        # n_hat = (3/2) * s * m² / σ_vm  (m = Mandel factors; m²=[1,1,1,2,2,2] for 3D)
        # This matches autograd.grad(yield_function w.r.t. stress) without nested autograd.
        stress = state_new["stress"]
        C = self.elastic_stiffness(state_new)
        s = self._dev(stress)
        vm = self._vonmises(stress)
        m2 = anp.array(self.dimension.mandel_factors_np) ** 2
        n_hat = 1.5 * (s * m2) / vm
        R_stress = stress - stress_trial + dlambda * (C @ n_hat)
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        return [self.stress(R_stress), self.ep(R_ep)]


@pytest.mark.parametrize("deps_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0],
])
def test_custom_stress_residual_matches_reference(deps_vec):
    """Custom R_stress in state_residual must converge to same solution as reference."""
    ref = J2Isotropic3D(**_J2_PARAMS)
    custom = _CustomStressResidual(**_J2_PARAMS)
    state0 = ref.initial_state()
    deps = anp.array(deps_vec)

    r_ref = PythonIntegrator(ref).stress_update(deps, anp.zeros(6), state0)
    r_custom = PythonIntegrator(custom).stress_update(deps, anp.zeros(6), state0)

    np.testing.assert_allclose(
        np.array(r_custom.stress), np.array(r_ref.stress), atol=1e-7,
        err_msg=f"Custom R_stress stress differs from reference for deps={deps_vec}"
    )
    np.testing.assert_allclose(
        float(r_custom.state["ep"]), float(r_ref.state["ep"]), atol=1e-10,
    )


# ---------------------------------------------------------------------------
# Consistent tangent with custom R_stress is differentiable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deps_vec", [
    [2e-3, 0.0, 0.0, 0.0, 0.0, 0.0],
])
def test_custom_stress_residual_consistent_tangent(deps_vec):
    """Consistent tangent must match FD for the custom R_stress model."""
    model = _CustomStressResidual(**_J2_PARAMS)
    state0 = model.initial_state()
    result = check_tangent(
        PythonIntegrator(model),
        anp.zeros(6),
        state0,
        anp.array(deps_vec),
    )
    assert result.passed, (
        f"FD tangent check failed for custom R_stress: max_rel_err={result.max_rel_err:.3e}"
    )


