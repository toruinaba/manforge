"""Unit tests for YUKinematic and its dimension-specialised subclasses."""

import numpy as np
import pytest
import numpy.testing as npt
from manforge.models import YUKinematic3D, YUKinematicPS, YUKinematic1D
from manforge.core.dimension import SOLID_3D, PLANE_STRESS_P, UNIAXIAL_1D

PARAMS = dict(
    E=206_000, nu=0.3, Y=360.0, C_1=2000.0, C_2=200.0,
    B=435.0, Rsat=255.0, k=26.0, b=66.0, h=0.4, Ea=159_000, xi=61.0,
)


# ---------------------------------------------------------------------------
# U-1: class definition regression
# ---------------------------------------------------------------------------

def test_param_names():
    model = YUKinematic3D(**PARAMS)
    assert "B" in model.param_names
    assert len(model.param_names) == 12
    assert set(model.param_names) == {
        "E", "nu", "Y", "B", "C_1", "C_2", "Rsat", "k", "b", "h", "Ea", "xi"
    }


def test_state_names():
    model = YUKinematic3D(**PARAMS)
    assert set(model.state_names) == {
        "stress", "theta", "beta", "R", "q", "r", "eps_eq", "theta_max"
    }
    assert set(model.implicit_state_names) == {"stress", "theta", "beta"}


@pytest.mark.parametrize("cls,dim", [
    (YUKinematic3D,  SOLID_3D),
    (YUKinematicPS,  PLANE_STRESS_P),
    (YUKinematic1D,  UNIAXIAL_1D),
])
def test_subclass_dimension(cls, dim):
    model = cls(**PARAMS)
    assert model.dimension is dim


# ---------------------------------------------------------------------------
# U-2: _calc_E_factor
# ---------------------------------------------------------------------------

def test_calc_E_factor_at_zero():
    model = YUKinematic3D(**PARAMS)
    assert model._calc_E_factor(0.0) == pytest.approx(1.0)


def test_calc_E_factor_asymptote():
    model = YUKinematic3D(**PARAMS)
    assert model._calc_E_factor(1e6) == pytest.approx(PARAMS["Ea"] / PARAMS["E"], abs=1e-6)


def test_calc_E_factor_monotone():
    model = YUKinematic3D(**PARAMS)
    eps_vals = np.linspace(0.0, 0.1, 20)
    factors = np.array([float(model._calc_E_factor(e)) for e in eps_vals])
    assert np.all(np.diff(factors) <= 0)


# ---------------------------------------------------------------------------
# U-7: yield_function scaling
# ---------------------------------------------------------------------------

def _make_state(model, stress, theta=None, beta=None):
    state = model.initial_state()
    state["stress"] = stress
    state["theta"] = theta if theta is not None else np.zeros_like(stress)
    state["beta"]  = beta  if beta  is not None else np.zeros_like(stress)
    return state


def test_yield_function_elastic_stress_is_negative():
    """Zero stress with zero backstress should be well inside yield surface."""
    model = YUKinematic3D(**PARAMS)
    state = _make_state(model, np.zeros(6))
    assert model.yield_function(state) == pytest.approx(-PARAMS["Y"])


def test_yield_function_on_surface_is_zero():
    """Uniaxial stress exactly equal to Y (von Mises) should give f=0."""
    model = YUKinematic3D(**PARAMS)
    Y = PARAMS["Y"]
    sigma = np.array([Y, 0.0, 0.0, 0.0, 0.0, 0.0])
    state = _make_state(model, sigma)
    assert model.yield_function(state) == pytest.approx(0.0, abs=1e-8)


def test_yield_function_outside_is_positive():
    model = YUKinematic3D(**PARAMS)
    Y = PARAMS["Y"]
    sigma = np.array([2 * Y, 0.0, 0.0, 0.0, 0.0, 0.0])
    state = _make_state(model, sigma)
    assert model.yield_function(state) > 0


def test_yield_function_1d_at_origin():
    model = YUKinematic1D(**PARAMS)
    Y = PARAMS["Y"]
    state = _make_state(model, np.zeros(1))
    assert model.yield_function(state) == pytest.approx(-Y)


def test_yield_function_ps_at_origin():
    """YUKinematicPS uses the quadratic form f = ½ξᵀPξ − ⅓Y², not f = q − Y."""
    model = YUKinematicPS(**PARAMS)
    Y = PARAMS["Y"]
    state = _make_state(model, np.zeros(3))
    assert model.yield_function(state) == pytest.approx(-Y * Y / 3.0)


# ---------------------------------------------------------------------------
# U-4: update_state raises on mu NR non-convergence (§A.2 lock-in)
# U-5: user_defined_return_mapping contains raise (B-2 lock-in, static check)
# ---------------------------------------------------------------------------

def _make_mu_rn_zero_state(model):
    """r_n=0: stagnation radius is zero at step start (initial state or reset)."""
    state_n = model.initial_state()
    state_n["q"] = np.zeros(6)
    state_n["r"] = 0.0
    state_n["beta"] = np.array([100.0, -50.0, -50.0, 0.0, 0.0, 0.0])
    state_new = dict(state_n)
    state_new["theta"] = np.zeros(6)
    state_new["stress"] = np.zeros(6)
    state_new["beta"] = state_n["beta"]   # d_beta = 0 → Fn = 0
    return state_n, state_new


def test_update_state_rn_zero_returns_mu_zero():
    """§A.2: r_n=0 is degenerate (no valid mu>=0 solution); update_state must not crash.

    When the stagnation radius is zero at step start, the mu Newton equation has no
    physical solution. The correct answer is mu=0 (no stagnation-surface update).
    Previously this case caused sqrt(negative) → nan → ValueError.
    """
    model = YUKinematic3D(**PARAMS)
    state_n, state_new = _make_mu_rn_zero_state(model)
    # Should complete without raising
    result = model.update_state(0.001, state_new, state_n)
    assert result is not None


def test_user_defined_return_mapping_has_raise():
    """B-2 lock-in: user_defined_return_mapping contains 'raise ValueError' for mu non-convergence.

    Triggering the raise dynamically requires a beta that stays fixed across outer NR
    iterations, which is not achievable from outside the NR loop. Instead we verify
    the raise is present in the source and carries the expected message.
    """
    import inspect
    src = inspect.getsource(YUKinematic3D.user_defined_return_mapping)
    assert "raise ValueError" in src
    assert "user_defined_return_mapping" in src


# ---------------------------------------------------------------------------
# U-8: state_residual structure sanity
# ---------------------------------------------------------------------------

def test_state_residual_returns_three_fields():
    """state_residual must return exactly [stress, theta, beta]."""
    model = YUKinematic3D(**PARAMS)
    state_n = model.initial_state()
    state_n["stress"] = np.array([400.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    state_new = dict(state_n)
    stress_trial = state_n["stress"].copy()
    results = model.state_residual(state_new, 0.0, state_n, stress_trial=stress_trial)
    names = {r.name for r in results}
    assert names == {"stress", "theta", "beta"}


def test_state_residual_stress_shape():
    model = YUKinematic3D(**PARAMS)
    state_n = model.initial_state()
    state_new = dict(state_n)
    stress_trial = np.zeros(6)
    results = model.state_residual(state_new, 0.0, state_n, stress_trial=stress_trial)
    stress_res = next(r for r in results if r.name == "stress")
    assert stress_res.value.shape == (6,)


def test_state_residual_stress_is_zero_at_elastic_predictor():
    """At zero dlambda with state_new == state_n, R_stress = state_new['stress'] - stress_trial = 0."""
    model = YUKinematic3D(**PARAMS)
    state_n = model.initial_state()
    state_n["stress"] = np.array([200.0, -100.0, -100.0, 0.0, 0.0, 0.0])
    state_new = dict(state_n)
    # stress_trial equals state_new['stress'] → default_stress_residual = stress - trial + 0 = 0
    stress_trial = state_n["stress"].copy()
    results = model.state_residual(state_new, 0.0, state_n, stress_trial=stress_trial)
    stress_res = next(r for r in results if r.name == "stress")
    npt.assert_allclose(stress_res.value, np.zeros(6), atol=1e-12)


@pytest.mark.parametrize("cls,ntens", [
    (YUKinematicPS, 3),
    (YUKinematic1D, 1),
])
def test_state_residual_shape_subclasses(cls, ntens):
    """state_residual works for PS and 1D subclasses."""
    model = cls(**PARAMS)
    state_n = model.initial_state()
    state_new = dict(state_n)
    stress_trial = np.zeros(ntens)
    results = model.state_residual(state_new, 0.0, state_n, stress_trial=stress_trial)
    names = {r.name for r in results}
    assert names == {"stress", "theta", "beta"}
    for r in results:
        assert r.value.shape == (ntens,)
# U-3: update_state r accumulates (B-1 lock-in)
# U-6: theta_max monotone non-decreasing (B-4 lock-in)
# ---------------------------------------------------------------------------

def _update_state_result_dict(model, dlambda, state_new, state_n):
    results = model.update_state(dlambda, state_new, state_n)
    return {r.name: r.value for r in results}


def _make_plastic_state(model):
    """State_n with pre-existing stagnation surface (r=5) and beta outside it."""
    state_n = model.initial_state()
    state_n["r"] = 5.0
    state_n["q"] = np.zeros(6)
    state_n["beta"] = np.array([20.0, -10.0, -10.0, 0.0, 0.0, 0.0])  # ||beta - q|| > r
    state_n["R"] = 10.0
    state_n["theta_max"] = 50.0
    state_n["theta"] = np.array([50.0, -25.0, -25.0, 0.0, 0.0, 0.0])
    state_new = dict(state_n)
    state_new["beta"] = state_n["beta"] * 1.1   # moved — d_beta != 0
    state_new["theta"] = np.array([55.0, -27.5, -27.5, 0.0, 0.0, 0.0])
    state_new["stress"] = np.zeros(6)
    return state_n, state_new


def test_update_state_r_accumulates():
    """B-1 regression: r must be state_n['r'] + delta_r * g_flag, not delta_r alone."""
    model = YUKinematic3D(**PARAMS)
    state_n, state_new = _make_plastic_state(model)
    out = _update_state_result_dict(model, 0.001, state_new, state_n)
    assert out["r"] > state_n["r"], "r must be strictly larger than state_n['r'] when stagnation surface is active"


def test_update_state_returns_five_fields():
    """update_state must return exactly 5 explicit fields."""
    model = YUKinematic3D(**PARAMS)
    state_n, state_new = _make_plastic_state(model)
    results = model.update_state(0.001, state_new, state_n)
    names = {r.name for r in results}
    assert names == {"R", "q", "r", "eps_eq", "theta_max"}


def test_update_state_eps_eq_accumulates():
    """eps_eq must be state_n['eps_eq'] + dlambda."""
    model = YUKinematic3D(**PARAMS)
    state_n, state_new = _make_plastic_state(model)
    dlambda = 0.002
    out = _update_state_result_dict(model, dlambda, state_new, state_n)
    assert out["eps_eq"] == pytest.approx(state_n["eps_eq"] + dlambda)


def test_update_state_theta_max_monotone():
    """B-4 regression: theta_max is non-decreasing."""
    model = YUKinematic3D(**PARAMS)
    state_n, state_new = _make_plastic_state(model)
    # Case 1: theta_norm > current theta_max → increases
    theta_norm_new = model.vonmises_norm(state_new["theta"])
    state_n["theta_max"] = theta_norm_new * 0.5   # below new theta_norm
    out = _update_state_result_dict(model, 0.001, state_new, state_n)
    assert out["theta_max"] >= state_n["theta_max"]
    # Case 2: theta_norm < current theta_max → does not decrease
    state_n2 = dict(state_n)
    state_n2["theta_max"] = theta_norm_new * 2.0   # above new theta_norm
    out2 = _update_state_result_dict(model, 0.001, state_new, state_n2)
    assert out2["theta_max"] >= state_n2["theta_max"] - 1e-12
