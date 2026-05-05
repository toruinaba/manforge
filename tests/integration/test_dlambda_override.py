"""Integration tests for self.dlambda(R) — user-overridable Δλ residual.

Covers three paths:
1. Scalar-NR with dlambda override (stress=Explicit, no Implicit state)
2. Vector-NR with dlambda override (alpha=Implicit)
3. Vector-NR fallback: omitting self.dlambda uses yield_function default

All tests verify that the override path produces correct physics by comparing
against either the default path or analytically expected results.
"""

import numpy as np
import pytest
import autograd.numpy as anp

from manforge.core.material import MaterialModel3D
from manforge.core.state import Implicit, Explicit, NTENS
from manforge.core.stress_state import SOLID_3D
from manforge.simulation.integrator import PythonIntegrator


def stress_update(model, deps, stress_n, state_n):
    return PythonIntegrator(model).stress_update(deps, stress_n, state_n)


# ---------------------------------------------------------------------------
# Shared elastic constants (steel-like)
# ---------------------------------------------------------------------------

_E = 210000.0
_NU = 0.3
_SIGMA_Y0 = 250.0
_H = 1000.0          # isotropic hardening
_H_K = 5000.0        # kinematic hardening stiffness

_LAM = _E * _NU / ((1.0 + _NU) * (1.0 - 2.0 * _NU))
_MU = _E / (2.0 * (1.0 + _NU))


# ---------------------------------------------------------------------------
# Model 1: scalar-NR, dlambda override ≡ yield_function (identity check)
# ---------------------------------------------------------------------------

class _J2ScalarWithDlambdaOverride(MaterialModel3D):
    """J2 without hardening.  state_residual returns self.dlambda(yield_function)
    which is semantically identical to the default.  Used to verify that the
    scalar-NR path produces the same result whether the override is present or not.
    """
    param_names = ["E", "nu", "sigma_y0"]
    stress = Explicit(shape=NTENS, doc="stress")
    ep = Explicit(shape=(), doc="ep")

    def __init__(self, *, E, nu, sigma_y0):
        super().__init__()
        self.E = E
        self.nu = nu
        self.sigma_y0 = sigma_y0

    def elastic_stiffness(self, state=None):
        return self.isotropic_C(_LAM, _MU)

    def yield_function(self, state):
        return self._vonmises(state["stress"]) - self.sigma_y0

    def update_state(self, dlambda, state_n, state_trial):
        ep_new = state_n["ep"] + dlambda
        stress_new = self.default_stress_update(dlambda, state_n, state_trial)
        return [self.ep(ep_new), self.stress(stress_new)]

    def state_residual(self, state_new, dlambda, state_n, state_trial):
        # Explicit override that is mathematically identical to the default
        return [self.dlambda(self.yield_function(state_new))]


class _J2ScalarDefault(MaterialModel3D):
    """Same model without the dlambda override — uses default yield_function path."""
    param_names = ["E", "nu", "sigma_y0"]
    stress = Explicit(shape=NTENS, doc="stress")
    ep = Explicit(shape=(), doc="ep")

    def __init__(self, *, E, nu, sigma_y0):
        super().__init__()
        self.E = E
        self.nu = nu
        self.sigma_y0 = sigma_y0

    def elastic_stiffness(self, state=None):
        return self.isotropic_C(_LAM, _MU)

    def yield_function(self, state):
        return self._vonmises(state["stress"]) - self.sigma_y0

    def update_state(self, dlambda, state_n, state_trial):
        ep_new = state_n["ep"] + dlambda
        stress_new = self.default_stress_update(dlambda, state_n, state_trial)
        return [self.ep(ep_new), self.stress(stress_new)]


def _plastic_strain_increment():
    """Return a uniaxial strain increment that causes yielding."""
    return anp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.fixture
def scalar_with_override():
    return _J2ScalarWithDlambdaOverride(E=_E, nu=_NU, sigma_y0=_SIGMA_Y0)


@pytest.fixture
def scalar_default():
    return _J2ScalarDefault(E=_E, nu=_NU, sigma_y0=_SIGMA_Y0)


def test_scalar_nr_dlambda_override_matches_default(scalar_with_override, scalar_default):
    """Scalar-NR: self.dlambda(yield_function) override gives same result as default."""
    deps = _plastic_strain_increment()
    state_n_ovr = scalar_with_override.initial_state()
    state_n_def = scalar_default.initial_state()

    r_ovr = stress_update(scalar_with_override, deps, anp.zeros(6), state_n_ovr)
    r_def = stress_update(scalar_default, deps, anp.zeros(6), state_n_def)

    np.testing.assert_allclose(
        np.asarray(r_ovr.stress), np.asarray(r_def.stress), atol=1e-10,
        err_msg="stress differs between dlambda-override and default paths"
    )
    np.testing.assert_allclose(
        float(r_ovr.dlambda), float(r_def.dlambda), atol=1e-10,
        err_msg="dlambda differs between dlambda-override and default paths"
    )
    np.testing.assert_allclose(
        float(r_ovr.state["ep"]), float(r_def.state["ep"]), atol=1e-10,
        err_msg="ep differs between dlambda-override and default paths"
    )


def test_scalar_nr_dlambda_override_is_plastic(scalar_with_override):
    """Confirm the scalar-NR override step actually enters the plastic domain."""
    deps = _plastic_strain_increment()
    state_n = scalar_with_override.initial_state()
    r = stress_update(scalar_with_override, deps, anp.zeros(6), state_n)
    assert r.is_plastic, "expected plastic step"
    assert float(r.dlambda) > 0.0


# ---------------------------------------------------------------------------
# Model 2: vector-NR, kinematic hardening, Perzyna-like dlambda override
# ---------------------------------------------------------------------------

class _KinematicWithDlambdaOverride(MaterialModel3D):
    """Kinematic hardening (AF).  state_residual returns self.dlambda(R_dl)
    where R_dl = f − c·Δλ  with  c = 0.0  (reduces to standard plasticity).

    Setting c → 0 means the Perzyna term vanishes and results must match the
    purely rate-independent model without the dlambda override.
    """
    param_names = ["E", "nu", "sigma_y0", "H_k", "c"]
    alpha = Implicit(shape=NTENS, doc="backstress")

    def __init__(self, *, E, nu, sigma_y0, H_k, c=0.0):
        super().__init__()
        self.E = E
        self.nu = nu
        self.sigma_y0 = sigma_y0
        self.H_k = H_k
        self.c = c

    def elastic_stiffness(self, state=None):
        return self.isotropic_C(_LAM, _MU)

    def yield_function(self, state):
        xi = state["stress"] - state["alpha"]
        return self._vonmises(xi) - self.sigma_y0

    def state_residual(self, state_new, dlambda, state_n, state_trial):
        xi_new = state_new["stress"] - state_new["alpha"]
        f_new = self._vonmises(xi_new) - self.sigma_y0
        # flow direction n = df/dσ at current iterate
        import autograd
        from manforge.core.state import _state_with_stress
        n = autograd.grad(
            lambda s: self.yield_function(_state_with_stress(state_new, s))
        )(state_new["stress"])
        # Armstrong-Frederick backstress evolution
        R_alpha = (state_new["alpha"] - state_n["alpha"]
                   - dlambda * (self.H_k * n - 0.0 * state_n["alpha"]))
        # Perzyna-like consistency: f − c·Δλ = 0  (c=0 → standard)
        R_dl = f_new - self.c * dlambda
        return [self.alpha(R_alpha), self.dlambda(R_dl)]


class _KinematicDefault(MaterialModel3D):
    """Same AF model without dlambda override (uses yield_function default)."""
    param_names = ["E", "nu", "sigma_y0", "H_k"]
    alpha = Implicit(shape=NTENS, doc="backstress")

    def __init__(self, *, E, nu, sigma_y0, H_k):
        super().__init__()
        self.E = E
        self.nu = nu
        self.sigma_y0 = sigma_y0
        self.H_k = H_k

    def elastic_stiffness(self, state=None):
        return self.isotropic_C(_LAM, _MU)

    def yield_function(self, state):
        xi = state["stress"] - state["alpha"]
        return self._vonmises(xi) - self.sigma_y0

    def state_residual(self, state_new, dlambda, state_n, state_trial):
        import autograd
        from manforge.core.state import _state_with_stress
        n = autograd.grad(
            lambda s: self.yield_function(_state_with_stress(state_new, s))
        )(state_new["stress"])
        R_alpha = (state_new["alpha"] - state_n["alpha"]
                   - dlambda * (self.H_k * n - 0.0 * state_n["alpha"]))
        return [self.alpha(R_alpha)]


@pytest.fixture
def kin_with_override():
    return _KinematicWithDlambdaOverride(E=_E, nu=_NU, sigma_y0=_SIGMA_Y0, H_k=_H_K, c=0.0)


@pytest.fixture
def kin_default():
    return _KinematicDefault(E=_E, nu=_NU, sigma_y0=_SIGMA_Y0, H_k=_H_K)


def test_vector_nr_dlambda_override_c0_matches_default(kin_with_override, kin_default):
    """Vector-NR: Perzyna c=0 override gives same result as no-override."""
    deps = _plastic_strain_increment()
    state_n_ovr = kin_with_override.initial_state()
    state_n_def = kin_default.initial_state()

    r_ovr = stress_update(kin_with_override, deps, anp.zeros(6), state_n_ovr)
    r_def = stress_update(kin_default, deps, anp.zeros(6), state_n_def)

    np.testing.assert_allclose(
        np.asarray(r_ovr.stress), np.asarray(r_def.stress), atol=1e-8,
        err_msg="stress differs: c=0 override vs no-override"
    )
    np.testing.assert_allclose(
        float(r_ovr.dlambda), float(r_def.dlambda), atol=1e-8,
        err_msg="dlambda differs: c=0 override vs no-override"
    )
    np.testing.assert_allclose(
        np.asarray(r_ovr.state["alpha"]), np.asarray(r_def.state["alpha"]), atol=1e-8,
        err_msg="alpha differs: c=0 override vs no-override"
    )


def test_vector_nr_dlambda_override_converges(kin_with_override):
    """Verify the Perzyna override path converges in NR."""
    deps = _plastic_strain_increment()
    state_n = kin_with_override.initial_state()
    r = stress_update(kin_with_override, deps, anp.zeros(6), state_n)
    assert r.is_plastic
    assert r.return_mapping is not None
    assert r.return_mapping.n_iterations < 50
    # Residual history should decrease monotonically (or at least end small)
    hist = r.return_mapping.residual_history
    assert hist[-1] < 1e-8, f"NR residual did not converge: {hist[-1]:.3e}"


def test_vector_nr_omit_dlambda_fallback_matches_default(kin_with_override, kin_default):
    """Omitting self.dlambda in state_residual falls back to yield_function."""
    # kin_default has no self.dlambda — this test is redundant but explicit
    deps = _plastic_strain_increment()
    state_n_ovr = kin_with_override.initial_state()
    state_n_def = kin_default.initial_state()

    r_ovr_c0 = stress_update(kin_with_override, deps, anp.zeros(6), state_n_ovr)
    r_def = stress_update(kin_default, deps, anp.zeros(6), state_n_def)

    # c=0 case with override should match the no-override default
    np.testing.assert_allclose(
        np.asarray(r_ovr_c0.stress), np.asarray(r_def.stress), atol=1e-8
    )


# ---------------------------------------------------------------------------
# Model 3: consistent tangent still works with dlambda override
# ---------------------------------------------------------------------------

def test_scalar_nr_tangent_is_finite(scalar_with_override):
    """Consistent tangent must be finite and non-zero when dlambda is overridden."""
    deps = _plastic_strain_increment()
    state_n = scalar_with_override.initial_state()
    r = stress_update(scalar_with_override, deps, anp.zeros(6), state_n)
    assert r.is_plastic
    ddsdde = np.asarray(r.ddsdde)
    assert np.all(np.isfinite(ddsdde)), "tangent contains inf/nan"
    assert np.linalg.norm(ddsdde) > 0.0, "tangent is zero"


def test_vector_nr_tangent_is_finite(kin_with_override):
    """Consistent tangent with Perzyna override is finite and non-zero."""
    deps = _plastic_strain_increment()
    state_n = kin_with_override.initial_state()
    r = stress_update(kin_with_override, deps, anp.zeros(6), state_n)
    assert r.is_plastic
    ddsdde = np.asarray(r.ddsdde)
    assert np.all(np.isfinite(ddsdde)), "tangent contains inf/nan"
    assert np.linalg.norm(ddsdde) > 0.0, "tangent is zero"
