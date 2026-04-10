# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (package manager: uv)
uv sync --extra dev          # Python tests
uv sync --all-extras         # All extras including matplotlib, meson/ninja

# Test
make test                    # All tests except slow fitting tests
make test-all                # Full suite including slow fitting tests
uv run pytest tests/test_j2_elastic.py -v  # Single test file

# Fortran compilation (requires gfortran)
make fortran-build           # Compile test_basic.f90 via f2py
make fortran-build-umat      # Compile abaqus_stubs.f90 + umat_j2.f90 via f2py
make fortran-test            # Run Fortran basic tests
make fortran-test-umat       # Run Fortran UMAT cross-validation tests

# Docker (reproducible gfortran environment)
make docker-build && make docker-test

# Cleanup compiled Fortran artifacts
make clean
```

There is no linter configured. No CI/CD workflows exist.

## Architecture

manforge is a framework for validating Fortran UMAT (Abaqus user material) constitutive models. Users define a material model in Python; the framework handles return mapping, consistent tangent computation (via JAX autodiff), parameter fitting, and cross-validation against compiled Fortran UMAT subroutines.

### Three-layer design

**1. Material model layer** — `src/manforge/core/material.py`

Users subclass `MaterialModel` and implement exactly 3 methods:
- `elastic_stiffness(params)` → 6×6 Voigt stiffness tensor
- `yield_function(stress, state, params)` → scalar (≤0 = elastic)
- `hardening_increment(dlambda, state, params)` → updated state dict

The reference implementation is `src/manforge/models/j2_isotropic.py` (J2 plasticity with linear isotropic hardening).

**2. Solver layer** — `src/manforge/core/`

- `return_mapping.py`: Elastic trial → yield check → scalar Newton-Raphson on plastic multiplier Δλ (max 50 iter, tol=1e-10)
- `tangent.py`: Consistent tangent via implicit differentiation of the 7×7 return-mapping residual system (does NOT differentiate through the NR iterations)

JAX autodiff computes yield function gradients and the Hessian needed for the tangent. Float64 is enabled globally in `src/manforge/__init__.py`.

**3. Application layer**

- `simulation/driver.py`: `UniaxialDriver`, `BiaxialDriver`, `GeneralDriver` — step through strain histories, accumulate stress/state
- `fitting/optimizer.py`: `fit_params()` wraps scipy.optimize (L-BFGS-B, Nelder-Mead, differential_evolution); loss defined in `fitting/objective.py`; uses drivers from `simulation/`
- `verification/fd_check.py`: Compares AD tangent vs central finite differences
- `verification/fortran_bridge.py`: f2py interface; calls compiled UMAT and compares output element-wise to Python (stress tol: 1e-6, tangent tol: 1e-5)

### Voigt convention

Stress/strain vectors are 6-component: `[11, 22, 33, 12, 13, 23]` with physical shear (not engineering shear). When computing norms or equivalences, Mandel scaling (×√2 on shear components) is applied internally. Helpers in `utils/voigt.py`.

### State variables

State is `dict[str, jnp.ndarray]`, e.g. `{"ep": 0.05}` for equivalent plastic strain. This is a JAX pytree and flows through JIT boundaries cleanly (though JIT is not used in the current version).

### Fortran UMAT

`fortran/umat_j2.f90` implements the same J2 algorithm as the Python reference. It uses the full ABAQUS UMAT signature; `fortran/abaqus_stubs.f90` provides mock implementations of ABAQUS internals (SINV, SPRINC, ROTSIG) for standalone compilation. Compiled via f2py to `.so` files in `fortran/`. The `Dockerfile` provides a reproducible gfortran + Python 3.12 environment for Fortran builds.
