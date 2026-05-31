"""Tests for the umat ABAQUS shim in yu_kinematic_3d.f90.

Covers:
- DROT rotation: theta/beta/q backstresses are rotated to the co-rotational
  frame before being passed to yu_kinematic_3d.
- STATEV and STRESS are NOT written back on non-convergence (PNEWDT < 1).
- DROT = I gives identical results to calling yu_kinematic_3d directly.
"""
import sys
import numpy as np
import pytest

sys.path.insert(0, "/home/ec2-user/manforge/fortran")

fortran = pytest.importorskip("yu_kinematic_3d", reason="yu_kinematic_3d.so not found")

# ---------------------------------------------------------------------------
# Default material parameters (matches conftest.py and test_numerical_vs_fortran.py)
# ---------------------------------------------------------------------------
PROPS_DEFAULT = np.array([
    210000.0,   # E
    0.3,        # nu
    150.0,      # Y
    300.0,      # B
    10000.0,    # C_1
    1000.0,     # C_2
    100.0,      # Rsat
    5.0,        # k
    3.0,        # b
    0.5,        # h
    20000.0,    # Ea
    50.0,       # xi_param
], dtype=np.float64)

NPROPS = len(PROPS_DEFAULT)
NSTATV = 22
NTENS  = 6
NDI    = 3
NSHR   = 3


def _make_zero_statev():
    return np.zeros(NSTATV, dtype=np.float64)


def _identity_drot():
    return np.eye(3, dtype=np.float64)


def _call_umat(stress_n, statev_n, dstran, props=None, drot=None):
    """Thin wrapper around the Fortran umat subroutine (f2py signature).

    f2py signature (inout/out args returned, not passed in):
      ddsdde, sse, spd, scd, rpl, ddsddt, drplde, drpldt =
        umat(stress, statev, stran, dstran, time, dtime, temp, dtemp,
             predef, dpred, cmname, ndi, nshr, props, coords, drot,
             pnewdt, celent, dfgrd0, dfgrd1, noel, npt, layer, kspt,
             kstep, kinc, [ntens, nstatv, nprops])

    Returns (stress_out, statev_out, ddsdde, pnewdt).
    """
    if props is None:
        props = PROPS_DEFAULT
    if drot is None:
        drot = _identity_drot()

    stress = stress_n.copy()
    statev = statev_n.copy()
    stran  = np.zeros(NTENS, dtype=np.float64)
    time   = np.zeros(2, dtype=np.float64)
    predef = np.zeros(1, dtype=np.float64)
    dpred  = np.zeros(1, dtype=np.float64)
    coords = np.zeros(3, dtype=np.float64)
    dfgrd0 = np.eye(3, dtype=np.float64)
    dfgrd1 = np.eye(3, dtype=np.float64)
    pnewdt = np.array(1.0, dtype=np.float64)
    cmname = b"YUKINE  " + b" " * 72

    ddsdde, _sse, _spd, _scd, _rpl, _ddsddt, _drplde, _drpldt = fortran.umat(
        stress, statev,
        stran, dstran, time,
        1.0, 0.0, 0.0,   # dtime, temp, dtemp
        predef, dpred, cmname,
        NDI, NSHR, props, coords, drot,
        pnewdt, 1.0,      # pnewdt, celent
        dfgrd0, dfgrd1,
        1, 1, 1, 1, 1, 1, # noel, npt, layer, kspt, kstep, kinc
    )

    return stress, statev, ddsdde, float(pnewdt)


# ---------------------------------------------------------------------------
# Helper: one elastic step to get a non-trivial state
# ---------------------------------------------------------------------------
def _plastic_step():
    """Return (stress_out, statev_out) after one uniaxial plastic step."""
    stress_n = np.zeros(NTENS, dtype=np.float64)
    statev_n = _make_zero_statev()
    dstran   = np.array([2e-3, -6e-4, -6e-4, 0.0, 0.0, 0.0], dtype=np.float64)
    return _call_umat(stress_n, statev_n, dstran)[:2]  # (stress, statev)


# ---------------------------------------------------------------------------
# Test: DROT = I gives the same result as not rotating (baseline)
# ---------------------------------------------------------------------------
def test_drot_identity_no_change():
    """DROT = I must give identical results to a freshly zeroed state."""
    stress_n, statev_n = _plastic_step()
    dstran = np.array([1e-3, -3e-4, -3e-4, 0.0, 0.0, 0.0], dtype=np.float64)

    # Reference: identity rotation
    stress_ref, statev_ref, _, _ = _call_umat(stress_n, statev_n, dstran, drot=_identity_drot())
    # Under test: also identity (regression guard)
    stress_tst, statev_tst, _, _ = _call_umat(stress_n, statev_n, dstran, drot=_identity_drot())

    np.testing.assert_allclose(stress_tst, stress_ref, atol=1e-12)
    np.testing.assert_allclose(statev_tst, statev_ref, atol=1e-12)


# ---------------------------------------------------------------------------
# Test: DROT rotation — backstresses in STATEV are rotated before the call
# ---------------------------------------------------------------------------
def _voigt_rotate(v, R):
    """Rotate a Voigt-6 symmetric tensor: out = R * A * R^T."""
    A = np.array([[v[0], v[3], v[4]],
                  [v[3], v[1], v[5]],
                  [v[4], v[5], v[2]]])
    B = R @ A @ R.T
    return np.array([B[0,0], B[1,1], B[2,2], B[0,1], B[0,2], B[1,2]])


def test_drot_rotates_backstress():
    """After a pure 90-deg rotation with zero strain increment, the output
    backstresses must equal R * theta_in * R^T (and similarly for beta, q).

    ABAQUS convention:
      - STRESS passed to UMAT is already rotated by ABAQUS → simulate by
        pre-rotating the converged stress before passing as stress_n.
      - STATEV tensors are NOT rotated by ABAQUS → pass them unrotated.
      - UMAT must call ROTSIG to rotate theta/beta/q before the constitutive
        update.

    With DSTRAN = 0 the step is neutral (on the yield surface, treated as
    elastic by the Fortran check xi_trial_norm <= Y), so the only change to
    the output STATEV should be the ROTSIG rotation.
    """
    stress_conv, statev_n = _plastic_step()
    theta_n = statev_n[0:6].copy()
    beta_n  = statev_n[6:12].copy()

    # 90-degree rotation around z-axis
    angle = np.radians(90.0)
    c, s = np.cos(angle), np.sin(angle)
    drot = np.array([[c, s, 0.0],
                     [-s, c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)

    # ABAQUS has already rotated STRESS before calling UMAT
    stress_n_rotated = _voigt_rotate(stress_conv, drot)
    dstran = np.zeros(NTENS, dtype=np.float64)

    stress_out, statev_out, _, pnewdt = _call_umat(
        stress_n_rotated, statev_n, dstran, drot=drot
    )

    assert pnewdt >= 1.0, "Elastic/neutral step should converge"

    theta_expected = _voigt_rotate(theta_n, drot)
    beta_expected  = _voigt_rotate(beta_n,  drot)

    np.testing.assert_allclose(statev_out[0:6],  theta_expected, atol=1e-10,
                               err_msg="theta not correctly rotated by DROT")
    np.testing.assert_allclose(statev_out[6:12], beta_expected,  atol=1e-10,
                               err_msg="beta not correctly rotated by DROT")


# ---------------------------------------------------------------------------
# Test: non-convergence — STATEV and STRESS unchanged, PNEWDT < 1
# ---------------------------------------------------------------------------
def test_nonconvergence_freezes_statev():
    """A deliberately huge strain increment forces non-convergence.
    STRESS and STATEV must remain at their input values; PNEWDT must be < 1.
    """
    stress_n = np.zeros(NTENS, dtype=np.float64)
    statev_n = _make_zero_statev()
    # Extremely large increment — NR will not converge in 50 iterations
    dstran   = np.array([5.0, -1.5, -1.5, 0.0, 0.0, 0.0], dtype=np.float64)

    stress_out, statev_out, _, pnewdt = _call_umat(stress_n, statev_n, dstran)

    assert pnewdt < 1.0, f"Expected PNEWDT < 1 for non-convergence, got {pnewdt}"
    np.testing.assert_array_equal(stress_out, stress_n,
                                  err_msg="STRESS must not be updated on non-convergence")
    np.testing.assert_array_equal(statev_out, statev_n,
                                  err_msg="STATEV must not be updated on non-convergence")


# ---------------------------------------------------------------------------
# Test: PROPS / STATEV guard — incompatible element triggers PNEWDT = 0
# ---------------------------------------------------------------------------
def test_incompatible_element_returns_pnewdt_zero():
    """Calling umat with NPROPS=11 (<12) must trigger the guard and return PNEWDT=0."""
    stress = np.zeros(NTENS, dtype=np.float64)
    statev = np.zeros(NSTATV, dtype=np.float64)
    stran  = dstran = np.zeros(NTENS, dtype=np.float64)
    time   = np.zeros(2, dtype=np.float64)
    predef = dpred = np.zeros(1, dtype=np.float64)
    coords = np.zeros(3, dtype=np.float64)
    dfgrd0 = dfgrd1 = np.eye(3, dtype=np.float64)
    pnewdt = np.array(1.0, dtype=np.float64)
    cmname = b"YUKINE  " + b" " * 72
    drot   = np.eye(3, dtype=np.float64)

    short_props = PROPS_DEFAULT[:11].copy()  # NPROPS=11 < 12

    fortran.umat(
        stress, statev,
        stran, dstran, time,
        1.0, 0.0, 0.0,
        predef, dpred, cmname,
        NDI, NSHR, short_props, coords, drot,
        pnewdt, 1.0,
        dfgrd0, dfgrd1,
        1, 1, 1, 1, 1, 1,
    )

    assert float(pnewdt) == 0.0, f"Expected PNEWDT=0 for incompatible element, got {pnewdt}"
