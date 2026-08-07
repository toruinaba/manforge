"""Absolute-time measurement for the YUKinematicProjPS Fortran UMAT.

Run inside the Docker image (the host has no gfortran):
    make docker-bench-yu-proj-ps

Timing happens inside Fortran (yu_projps_bench.f90) so the f2py conversion
overhead is excluded.  Reports the full update plus each internal stage, so an
optimisation target can be set against measured numbers rather than guessed
ratios -- the guess here was wrong, the cost sits in calc_ddsdde and the linear
solve rather than in Jacobian assembly.
"""

import numpy as np
from _common import plastic_point, props

from manforge.models import YUKinematicProjPS
from manforge.simulation.integrator import FortranModule

N_REPEAT = 200_000


def main():
    # One module holds both the implementation and the timing wrappers; they are
    # separate compilation units, so the callee bodies stay opaque to the
    # optimiser and the repeat loops cannot be hoisted away.
    bench = FortranModule("yu_projps_bench")
    model, state, dstran, res = plastic_point(YUKinematicProjPS, ntens=3)
    p = props(model)

    def s(k):
        return np.asarray(state[k], dtype=float)

    state_args = (
        s("stress"), s("theta"), s("beta"), float(state["R"]),
        s("q"), float(state["r"]), float(state["eps_eq"]), float(state["theta_max"]),
    )

    # Full update -- the quantity that matters for ABAQUS runtime.
    el, _sink, n_iter = bench.call(
        "yu_projps_bench_full", *p, *state_args, dstran, N_REPEAT)
    us_full = el / N_REPEAT * 1e6
    n = int(n_iter)
    print(f"n_iter at bench point      : {n}")
    print(f"full stress_update         : {us_full:8.3f} us/call")

    # Internal stages, timed at the converged point of that same step.
    st = res.state
    jac_args = (
        np.asarray(st["stress"]), np.asarray(st["theta"]), np.asarray(st["beta"]),
        float(st["R"]), float(st["eps_eq"]),
        float(state["theta_max"]), float(state["R"]), float(res.dlambda),
    )
    el, _ = bench.call("yu_projps_bench_jac", *p, *jac_args, N_REPEAT)
    us_jac = el / N_REPEAT * 1e6
    print(f"  calc_jacobian            : {us_jac:8.3f} us/call")

    resid_args = (
        np.asarray(st["stress"]), np.asarray(st["theta"]), np.asarray(st["beta"]),
        float(st["R"]), float(st["eps_eq"]),
        s("theta"), s("beta"), float(state["theta_max"]),
        np.asarray(res.stress_trial), float(res.dlambda),
    )
    el, _ = bench.call("yu_projps_bench_resid", *p, *resid_args, N_REPEAT)
    us_res = el / N_REPEAT * 1e6
    print(f"  calc_residual            : {us_res:8.3f} us/call")

    dd_args = jac_args + (float(state["eps_eq"]),)
    el, _ = bench.call("yu_projps_bench_ddsdde", *p, *dd_args, N_REPEAT)
    us_dd = el / N_REPEAT * 1e6
    print(f"  calc_ddsdde              : {us_dd:8.3f} us/call")

    el, _ = bench.call("yu_projps_bench_ddsdde_fast", *p, *dd_args, N_REPEAT)
    us_fast = el / N_REPEAT * 1e6
    print(f"  ddsdde_fast (NRHS=3)     : {us_fast:8.3f} us/call   [-{us_dd - us_fast:.3f}]")

    dd_ref = np.asarray(bench.call("yu_projps_calc_ddsdde", *p, *dd_args))
    dd_new = np.asarray(bench.call("yu_projps_ddsdde_fast", *p, *dd_args))
    rel = np.abs(dd_new - dd_ref).max() / np.abs(dd_ref).max()
    print(f"  ddsdde_fast vs ref       : max rel diff {rel:.3e}")

    # At 10x10 the hand-rolled elimination beats dgesv by ~3x -- LAPACK's
    # blocking and argument checking cost more than the work itself at this
    # size.  (The 3-D file is 19x19, where dgesv is the right choice.)
    jac = np.asarray(bench.call("yu_projps_calc_jacobian", *p, *jac_args))
    us_lu = None
    for nrhs in (1, 3, 10):
        el, _ = bench.call("yu_projps_bench_solve", jac, nrhs, N_REPEAT)
        us = el / N_REPEAT * 1e6
        if nrhs == 1:
            us_lu = us
        print(f"  solve 10x10 (NRHS={nrhs:2d})    : {us:8.3f} us/call")
    for nrhs in (1, 3, 10):
        el, _ = bench.call("yu_projps_bench_dgesv", jac, nrhs, N_REPEAT)
        print(f"  dgesv 10x10 (NRHS={nrhs:2d})    : {el / N_REPEAT * 1e6:8.3f} us/call")

    print()
    print(f"accounted: {n}*(resid+jac+lu) + ddsdde = "
          f"{n * (us_res + us_jac + us_lu) + us_dd:.3f} us  vs  {us_full:.3f} us measured")
    print(f"ddsdde share of full       : {us_dd / us_full * 100:5.1f} %")
    proj = us_full - (us_dd - us_fast)
    print(f"projected full after fix   : {proj:8.3f} us  ({us_full / proj:.2f}x)")


if __name__ == "__main__":
    main()
