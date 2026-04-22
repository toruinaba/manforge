# Benchmark Results: JAX → autograd migration

Measured on 2026-04-22.  Platform: AWS Linux, CPU only.
Warmup: 1 run.  Measurement: median of 5 runs.

## Setup

- JAX  v0.9.2  (cpu, jax_enable_x64=True, no JIT)
- autograd 1.8.0 (numpy backend, no JIT)

## Results

| Scenario | JAX median (s) | autograd median (s) | Speedup |
|---|---:|---:|---:|
| driver_step — J2Isotropic3D, N=40 steps | 0.418 | 0.005 | **83×** |
| augmented_nr — OWKinematic3D, 200 steps | 267.8 | 18.4 | **15×** |
| fd_tangent — J2Isotropic3D check_tangent | 0.082 | 0.002 | **44×** |

## Interpretation

**driver_step (83×)**: J2 uses `user_defined_return_mapping` (closed-form radial return) so no
NR iteration runs. The speedup is almost entirely JAX's per-call Python↔XLA dispatch overhead,
which dominates when the computation itself is trivial (a few vector ops on 6-element arrays).
autograd executes the same numpy operations with near-zero tracing cost.

**augmented_nr (15×)**: OW 3D runs the (14×14) augmented Newton-Raphson for 200 steps.
Each step calls `autograd.jacobian` on the residual function — this is real autodiff work,
so the improvement is smaller than the other two. JAX's XLA dispatch overhead still
dominates over the actual linear-algebra cost at this scale.

**fd_tangent (44×)**: 1 + 12 = 13 `stress_update` calls (central differences).
Proportional to driver_step since J2 uses analytical hooks.

## Decision

All three scenarios show substantial improvement well above the 1.5× threshold.
Migration to autograd is confirmed. JAX dependency removed from `pyproject.toml`.
