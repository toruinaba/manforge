"""Service contract layer: registry → describe → run_simulation.

Demonstrates the workflow a web API / job worker builds on:
- Building a ModelRegistry and listing what it exposes
- describe() — the GET /models payload (param metadata for UI forms)
- run_simulation with plain inputs/outputs and a progress callback
- Non-convergence returned as a value (partial history), not an exception

Usage
-----
    uv run python examples/example_service.py
"""

import numpy as np

import manforge  # noqa: F401 — enables float64
from manforge.models import OWKinematic1D
from manforge.service import default_registry, run_simulation

# ---------------------------------------------------------------------------
# 1. Registry — explicit name → class mapping (what a deployment exposes)
# ---------------------------------------------------------------------------
reg = default_registry()

print("=" * 60)
print("  Registered models (GET /models)")
print("=" * 60)
for name in reg.names():
    d = reg.describe(name)
    print(f"\n  {name}  ({d.class_name}, ntens={d.ntens})")
    print(f"    {d.doc}")
    for pname in d.param_names:
        m = d.param_meta[pname]
        unit = f" [{m.unit}]" if m.unit else ""
        print(f"    {pname:10s}{unit:7s} default={m.default:<10g} bounds={m.bounds}  — {m.doc}")

# A private-model package would plug in the same way:
#     from manforge_internal import MODELS
#     for name, cls in MODELS.items():
#         reg.register(name, cls)

# ---------------------------------------------------------------------------
# 2. Run — defaults from the descriptor form a runnable parameter set
#    (the WebUI golden path: auto-filled form → submit)
# ---------------------------------------------------------------------------
name = "j2_isotropic"
d = reg.describe(name)
params = {k: m.default for k, m in d.param_meta.items()}

progress_log = []
out = run_simulation(
    reg.get(name),
    params,
    load=np.linspace(0.0, 5e-3, 200),        # strain history, raw list/array
    include_fields=["ep"],                    # state output is opt-in
    progress=lambda done, total: progress_log.append((done, total)),
)

print()
print("=" * 60)
print(f"  run_simulation: {name} (defaults)")
print("=" * 60)
print(f"  converged        : {out.converged}")
print(f"  n_steps          : {out.n_steps}")
print(f"  final sigma11    : {out.stress[-1, 0]:.1f} MPa")
print(f"  final ep         : {out.fields['ep'][-1]:.4e}")
print(f"  progress calls   : {len(progress_log)} (last: {progress_log[-1]})")

# ---------------------------------------------------------------------------
# 3. Non-convergence is a value, not an exception
#    A user exploring parameters will submit unsolvable values; the service
#    returns the partial history so the UI can plot it with a marker.
# ---------------------------------------------------------------------------
bad_params = dict(E=206e3, nu=0.3, sigma_y0=250.0, C_k=1e9, gamma=1e12)
out_bad = run_simulation(OWKinematic1D, bad_params, np.linspace(0.0, 0.05, 10))

print()
print("=" * 60)
print("  Unsolvable parameters (OW, absurd recovery)")
print("=" * 60)
print(f"  converged        : {out_bad.converged}")
print(f"  failed_step      : {out_bad.failed_step}")
print(f"  partial history  : {out_bad.n_steps} steps, stress shape {out_bad.stress.shape}")

# Caller bugs DO raise (ValueError → HTTP 400 in the API layer):
try:
    run_simulation(reg.get(name), {"E": 206e3}, np.linspace(0, 1e-3, 5))
except ValueError as e:
    print(f"\n  Caller bug raises: {e}")
