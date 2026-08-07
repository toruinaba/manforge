"""Absolute-time measurement for the YUKinematic3D Fortran UMAT.

Run inside the Docker image (the host has no gfortran):
    make docker-bench-yu-3d

Timing happens inside Fortran (yu_3d_bench.f90) so the f2py conversion overhead
is excluded.  The 3-D system has 19 unknowns and yu_calc_ddsdde inverts it at
NRHS=19 while using only the leading 6 columns, so the right-hand-side
reduction matters far more here than in plane stress.
"""

import numpy as np
from _common import plastic_point, props

from manforge.models import YUKinematic3D
from manforge.simulation.integrator import FortranModule

N_REPEAT = 100_000


def main():
    bench = FortranModule("yu_3d_bench")
    model, state, dstran, res = plastic_point(YUKinematic3D, ntens=6)
    p = props(model)

    def s(k):
        return np.asarray(state[k], dtype=float)

    state_args = (
        s("stress"), s("theta"), s("beta"), float(state["R"]),
        s("q"), float(state["r"]), float(state["eps_eq"]), float(state["theta_max"]),
    )

    el, _sink, n_iter = bench.call(
        "yu_3d_bench_full", *p, *state_args, dstran, N_REPEAT)
    us_full = el / N_REPEAT * 1e6
    print(f"n_iter at bench point      : {int(n_iter)}")
    print(f"full stress_update         : {us_full:8.3f} us/call")

    st = res.state
    # theta_max_new, not theta_max_n: the 3-D Jacobian takes the updated value.
    jac_args = (
        np.asarray(st["stress"]), np.asarray(st["theta"]), np.asarray(st["beta"]),
        float(st["R"]), float(st["eps_eq"]),
        float(st["theta_max"]), float(state["R"]), float(res.dlambda),
    )
    el, _ = bench.call("yu_3d_bench_jac", *p, *jac_args, N_REPEAT)
    print(f"  calc_jacobian            : {el / N_REPEAT * 1e6:8.3f} us/call")

    resid_args = (
        np.asarray(st["stress"]), np.asarray(st["theta"]), np.asarray(st["beta"]),
        float(st["R"]), float(st["eps_eq"]),
        s("theta"), s("beta"), float(state["theta_max"]),
        np.asarray(res.stress_trial), float(res.dlambda),
    )
    el, _ = bench.call("yu_3d_bench_resid", *p, *resid_args, N_REPEAT)
    print(f"  calc_residual            : {el / N_REPEAT * 1e6:8.3f} us/call")

    dd_args = jac_args + (float(state["eps_eq"]),)
    el, _ = bench.call("yu_3d_bench_ddsdde", *p, *dd_args, N_REPEAT)
    us_dd = el / N_REPEAT * 1e6
    print(f"  calc_ddsdde              : {us_dd:8.3f} us/call")

    el, _ = bench.call("yu_3d_bench_ddsdde_fast", *p, *dd_args, N_REPEAT)
    us_fast = el / N_REPEAT * 1e6
    print(f"  ddsdde_fast (NRHS=6)     : {us_fast:8.3f} us/call   [-{us_dd - us_fast:.3f}]")

    jac = np.asarray(bench.call("yu_calc_jacobian", *p, *jac_args))
    for nrhs in (1, 6, 19):
        el, _ = bench.call("yu_3d_bench_dgesv", jac, nrhs, N_REPEAT)
        print(f"  dgesv 19x19 (NRHS={nrhs:2d})    : {el / N_REPEAT * 1e6:8.3f} us/call")

    dd_ref = np.asarray(bench.call("yu_calc_ddsdde", *p, *dd_args))
    dd_new = np.asarray(bench.call("yu_3d_ddsdde_fast", *p, *dd_args))
    rel = np.abs(dd_new - dd_ref).max() / np.abs(dd_ref).max()
    print(f"  ddsdde_fast vs ref       : max rel diff {rel:.3e}")

    print()
    print(f"ddsdde share of full       : {us_dd / us_full * 100:5.1f} %")
    proj = us_full - (us_dd - us_fast)
    print(f"projected full after fix   : {proj:8.3f} us  ({us_full / proj:.2f}x)")


if __name__ == "__main__":
    main()
