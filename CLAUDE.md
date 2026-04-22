# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (package manager: uv)
uv sync --extra dev          # Python tests
uv sync --all-extras         # All extras including matplotlib, meson/ninja

# Test
make test                    # Fast tests: unit + integration, excluding slow/fortran
make test-unit               # Unit tests only (fastest)
make test-integration        # Integration tests excluding slow
make test-slow               # Slow tests (FD tangent, fitting, long loops)
make test-fortran            # Fortran cross-validation (requires compiled .so)
make test-all                # Full suite including slow and fortran
uv run pytest tests/integration/test_j2_elastic.py -v  # Single test file
uv run pytest tests/ --durations=20                    # Show 20 slowest tests

# Fortran compilation (requires gfortran)
# Option A: CLI (recommended for Python users)
uv run manforge build fortran/test_basic.f90 --name manforge_test_basic
uv run manforge build fortran/abaqus_stubs.f90 fortran/j2_isotropic_3d.f90 --name j2_isotropic_3d
uv run manforge list              # List compiled modules
uv run manforge clean             # Remove compiled artifacts
uv run manforge clean --dry-run   # Preview what would be removed

# Option B: Makefile (lower-level)
make fortran-build           # Compile test_basic.f90 via f2py
make fortran-build-umat      # Compile abaqus_stubs.f90 + j2_isotropic_3d.f90 via f2py → j2_isotropic_3d module
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

`MaterialModel` is the internal ABC. Users subclass one of the stress-state base classes and implement the required material-physics methods based on `hardening_type`:
- `elastic_stiffness()` → (ntens, ntens) Voigt stiffness tensor (always required)
- `yield_function(stress, state)` → scalar (≤0 = elastic) (always required)
- For **reduced** hardening (`hardening_type = "reduced"`, default): `update_state(dlambda, stress, state)` → updated state dict `q_{n+1}`. State is substituted into the residual each NR iteration; scalar NR on Δλ only (1D reduced system).
- For **augmented** hardening (`hardening_type = "augmented"`): `state_residual(state_new, dlambda, stress, state_n)` → residual dict (zero at convergence). State is an independent unknown; full (ntens+1+n_state) vector NR (augmented system). The two are related by `state_residual = q_{n+1} − update_state(...)`, and the base class provides `state_residual` automatically from `update_state`.

Material parameters are passed at construction time (`model = J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)`) and stored as instance attributes (`self.E`, `self.nu`, etc.). The `params` property auto-generates a dict from `param_names` for internal use.

`__init_subclass__` validates the required method is implemented at class definition time.

Stress-state base classes (choose the appropriate one):
- `MaterialModel3D` — SOLID_3D (ntens=6) and PLANE_STRAIN (ntens=4); requires `ndi == ndi_phys`
- `MaterialModelPS` — PLANE_STRESS (ntens=3); applies Schur condensation in `isotropic_C`
- `MaterialModel1D` — UNIAXIAL_1D (ntens=1); used for uniaxial fitting

Each base provides branch-free operator methods (`_dev`, `_vonmises`, `isotropic_C`, `_I_vol`, `_I_dev`) tailored to its stress state. The `_vonmises` in `MaterialModelPS` and `MaterialModel1D` includes the missing-component correction (n_missing × p²).

Optional hooks for user-supplied implementations: `user_defined_corrector(stress_trial, C, state_n)` → 3-tuple `(stress, state, dlambda)` or 5-tuple `(..., n_iterations, residual_history)`; `user_defined_tangent(stress, state, dlambda, C, state_n)` → `(ntens, ntens)` array. Both default to `None` (framework falls back to autodiff/NR).

The reference implementation is `src/manforge/models/j2_isotropic.py` (J2Isotropic3D, J2IsotropicPS, J2Isotropic1D).

**2. Solver layer** — `src/manforge/core/`

- `stress_update.py`: Two-level API:
  - `stress_update(model, deps, stress_n, state_n, method="auto")` → `StressUpdateResult` — full constitutive integration (elastic trial → yield check → return mapping → consistent tangent). Equivalent to one UMAT call.
  - `return_mapping(model, stress_trial, C, state_n, method="auto")` → `ReturnMappingResult` — plastic correction only (closest point projection). Dispatches on `model.hardening_type`: `"reduced"` → scalar NR on Δλ; `"augmented"` → (ntens+1+n_state) vector NR (max 50 iter, tol=1e-10).
  - `method` values: `"auto"` (use `user_defined_corrector` if present, else `"numerical_newton"`), `"numerical_newton"` (framework NR), `"user_defined"` (requires model to implement `user_defined_corrector`).
  - `ReturnMappingResult` fields: `stress`, `state`, `dlambda`, `n_iterations`, `residual_history`.
  - `StressUpdateResult` fields: `return_mapping` (None for elastic), `ddsdde`, `stress_trial`, `is_plastic`. Convenience properties `stress`, `state`, `dlambda`, `n_iterations`, `residual_history` delegate to `return_mapping` (or elastic defaults).
- `jacobian.py`: `JacobianBlocks` dataclass and `ad_jacobian_blocks(model, result, state_n)` — computes the residual Jacobian at the converged point via `jax.jacobian` and decomposes it into named blocks (`dstress_dsigma`, `dyield_dsigma`, `dstate_dstate`, etc.). For augmented models, state blocks are keyed by variable name (e.g. `jac.dstate_dsigma["alpha"]`). Accepts both `StressUpdateResult` and `ReturnMappingResult`. Used for step-by-step verification of analytical derivatives.
- `tangent.py`: Consistent tangent via implicit differentiation — reduced models use (ntens+1)×(ntens+1) system; augmented models use (ntens+1+n_state) system (does NOT differentiate through NR iterations)
- `residual.py`: Residual builders for both paths: `make_reduced_residual` (reduced) and `make_augmented_residual` (augmented), plus `select_residual_builder` dispatch

JAX autodiff computes yield function gradients and the Hessian needed for the tangent. Float64 is enabled globally in `src/manforge/__init__.py`.

**3. Application layer**

- `simulation/driver.py`: `StrainDriver`, `StressDriver` (+ aliases `UniaxialDriver`, `GeneralDriver`) — step through strain/stress histories. Returns `DriverResult` with `step_results: list[StressUpdateResult]` as primary data; `stress`, `strain`, `fields` are derived properties computed from `step_results`.
- `fitting/optimizer.py`: `fit_params()` wraps scipy.optimize (L-BFGS-B, Nelder-Mead, differential_evolution); loss defined in `fitting/objective.py`; uses drivers from `simulation/`
- `verification/fd_check.py`: Compares AD tangent vs central finite differences
- `verification/fortran_bridge.py`: f2py interface; calls compiled UMAT and compares output element-wise to Python (stress tol: 1e-6, tangent tol: 1e-5)

### StressState and dimensionality

`StressState` (`src/manforge/core/stress_state.py`) is a frozen dataclass that encapsulates the element dimensionality (ABAQUS NTENS convention). Four pre-built instances: `SOLID_3D` (ntens=6), `PLANE_STRAIN` (ntens=4), `PLANE_STRESS` (ntens=3), `UNIAXIAL_1D` (ntens=1). The model's `stress_state` attribute drives the size of all stress/strain arrays and the condensation of the elastic stiffness.

### Voigt convention

For 3D solid elements, stress/strain vectors are 6-component: `[11, 22, 33, 12, 13, 23]` with physical shear (not engineering shear). For other element types the component count is `ntens` per the associated `StressState`. When computing norms or equivalences, Mandel scaling (×√2 on shear components) is applied internally. Helpers in `utils/voigt.py`.

### State variables

State is `dict[str, jnp.ndarray]`, e.g. `{"ep": 0.05}` for equivalent plastic strain. This is a JAX pytree and flows through JIT boundaries cleanly (though JIT is not used in the current version).

### Fortran UMAT

`fortran/j2_isotropic_3d.f90` implements the same J2 algorithm as the Python reference. Subroutine names match Python class names: `j2_isotropic_3d` (core logic, f2py callable), `umat` (ABAQUS entry point), `j2_isotropic_3d_elastic_stiffness` (component check). `fortran/abaqus_stubs.f90` provides mock implementations of ABAQUS internals (SINV, SPRINC, ROTSIG) for standalone compilation. Compiled via f2py into the `j2_isotropic_3d` module. The `Dockerfile` provides a reproducible gfortran + Python 3.12 environment for Fortran builds.

`FortranUMAT` (`verification/fortran_bridge.py`) is a thin f2py wrapper whose only job is float64 type conversion. All subroutines are called via `fortran.call(name, *args)` — the same pattern for the full routine and for sub-components.
