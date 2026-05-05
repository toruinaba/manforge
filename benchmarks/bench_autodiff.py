"""Benchmark: JAX baseline vs autograd replacement.

3 scenarios:
  1. driver_step   – StrainDriver, J2Isotropic3D, N=40 steps
  2. vector_nr     – OWKinematic3D, 200 small-increment steps (implicit_state_names)
  3. fd_tangent    – check_tangent for J2Isotropic3D (1 + 12 stress_update calls)

Run:
    uv run python benchmarks/bench_autodiff.py

Output: median / min / p95 wall time for each scenario, plus a comparison
table if a previous result file is present.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

RESULTS_FILE = Path(__file__).parent / "RESULTS.json"
WARMUP = 1
N_REPS = 5


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _time(fn, n_reps: int = N_REPS, warmup: int = WARMUP):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def _stats(times):
    s = sorted(times)
    return {
        "median": statistics.median(s),
        "min": s[0],
        "p95": s[int(len(s) * 0.95)] if len(s) > 1 else s[-1],
    }


# ---------------------------------------------------------------------------
# scenario 1: driver step  (J2 3D, N=40)
# ---------------------------------------------------------------------------

def _make_driver_step():
    import numpy as np
    from manforge.models.j2_isotropic import J2Isotropic3D
    from manforge.simulation.driver import StrainDriver
    from manforge.simulation.integrator import PythonIntegrator
    from manforge.simulation.types import FieldHistory, FieldType

    model = J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
    strain_history = np.zeros((40, 6))
    strain_history[:, 0] = np.linspace(0.0, 5e-3, 40)

    integrator = PythonIntegrator(model)
    load = FieldHistory(FieldType.STRAIN, "eps", strain_history)

    def run():
        StrainDriver(integrator).run(load)

    return run


# ---------------------------------------------------------------------------
# scenario 2: vector NR  (OW 3D, 200 increments, implicit_state_names=["alpha","ep"])
# ---------------------------------------------------------------------------

def _make_vector_nr():
    import numpy as np
    from manforge.models.ow_kinematic import OWKinematic3D
    from manforge.simulation.integrator import PythonIntegrator

    model = OWKinematic3D(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=1.0)
    integrator = PythonIntegrator(model)
    deps = np.zeros(6)
    deps[0] = 5e-4

    def run():
        stress_n = np.zeros(6)
        state_n = model.initial_state()
        for _ in range(200):
            r = integrator.stress_update(deps, stress_n, state_n)
            stress_n = np.asarray(r.stress)
            state_n = {k: np.asarray(v) for k, v in r.state.items()}

    return run


# ---------------------------------------------------------------------------
# scenario 3: FD tangent check  (J2 3D, 1 plastic step)
# ---------------------------------------------------------------------------

def _make_fd_tangent():
    import numpy as np
    from manforge.models.j2_isotropic import J2Isotropic3D
    from manforge.simulation.integrator import PythonIntegrator
    from manforge.verification.fd_check import check_tangent

    model = J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
    integrator = PythonIntegrator(model)
    stress0 = np.zeros(6)
    state0 = model.initial_state()
    deps_plastic = np.zeros(6)
    deps_plastic[0] = 2e-3
    r = integrator.stress_update(deps_plastic, stress0, state0)
    stress1 = np.asarray(r.stress)
    state1 = {k: np.asarray(v) for k, v in r.state.items()}

    deps2 = np.zeros(6)
    deps2[0] = 5e-4

    def run():
        check_tangent(integrator, stress1, state1, deps2)

    return run


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

SCENARIOS = [
    ("driver_step",   _make_driver_step),
    ("vector_nr",     _make_vector_nr),
    ("fd_tangent",    _make_fd_tangent),
]

COL_W = 14


def _header():
    cols = ["scenario", "median (s)", "min (s)", "p95 (s)"]
    print("  ".join(f"{c:<{COL_W}}" for c in cols))
    print("  ".join("-" * COL_W for _ in cols))


def main():
    # detect backend label from installed packages
    try:
        import jax  # noqa: F401
        backend = "jax"
    except ImportError:
        try:
            import autograd  # noqa: F401
            backend = "autograd"
        except ImportError:
            backend = "unknown"

    print(f"\n=== bench_autodiff  backend={backend}  reps={N_REPS} ===\n")
    _header()

    results: dict[str, dict] = {}
    for name, factory in SCENARIOS:
        fn = factory()
        times = _time(fn)
        st = _stats(times)
        results[name] = st
        print("  ".join([
            f"{name:<{COL_W}}",
            f"{st['median']:<{COL_W}.4f}",
            f"{st['min']:<{COL_W}.4f}",
            f"{st['p95']:<{COL_W}.4f}",
        ]))

    # load previous results for comparison
    previous: dict = {}
    if RESULTS_FILE.exists():
        data = json.loads(RESULTS_FILE.read_text())
        other = [k for k in data if k != backend]
        if other:
            previous = data[other[0]]
            prev_backend = other[0]

    if previous:
        print(f"\n=== comparison  {prev_backend} → {backend} (median ratio) ===\n")
        cols2 = ["scenario", f"{prev_backend} (s)", f"{backend} (s)", "ratio"]
        print("  ".join(f"{c:<{COL_W}}" for c in cols2))
        print("  ".join("-" * COL_W for _ in cols2))
        for name in results:
            if name in previous:
                prev_m = previous[name]["median"]
                curr_m = results[name]["median"]
                ratio = curr_m / prev_m
                print("  ".join([
                    f"{name:<{COL_W}}",
                    f"{prev_m:<{COL_W}.4f}",
                    f"{curr_m:<{COL_W}.4f}",
                    f"{ratio:<{COL_W}.3f}",
                ]))

    # persist results
    existing: dict = {}
    if RESULTS_FILE.exists():
        existing = json.loads(RESULTS_FILE.read_text())
    existing[backend] = results
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(existing, indent=2))
    print(f"\nResults saved → {RESULTS_FILE}\n")


if __name__ == "__main__":
    main()
