# fortran/ — Fortran UMAT build guide

This directory contains Fortran source files for implementing and validating
ABAQUS UMAT constitutive subroutines that can be cross-validated against the
Python `manforge` implementation.

## Files

| File | Description |
|------|-------------|
| `j2_isotropic_3d.f90` | J2 isotropic hardening UMAT — contains `j2_isotropic_3d` (f2py interface), `umat` (ABAQUS entry point), and `j2_isotropic_3d_elastic_stiffness` (component check) |
| `abaqus_stubs.f90` | Stubs for ABAQUS internal functions (SINV, SPRINC, ROTSIG) — linked for symbol resolution; future UMATs may call them directly |
| `test_basic.f90` | Simple elastic subroutine for f2py smoke test |

---

## Prerequisites

- **gfortran** ≥ 9
- **Python** ≥ 3.10 with numpy (f2py is bundled with numpy)
- **meson** + **ninja** — required by f2py on Python 3.12+ (`uv sync --extra fortran`)
- **Docker** (recommended for a reproducible environment — see below)

Check availability:

```bash
gfortran --version
python -m numpy.f2py --version
```

---

## Build

### Option A: `manforge build` CLI (recommended)

```bash
# Install build tools (meson/ninja) if not yet done
uv sync --extra fortran

# Compile the J2 UMAT
uv run manforge build fortran/abaqus_stubs.f90 fortran/j2_isotropic_3d.f90 --name j2_isotropic_3d

# Smoke-test module only
uv run manforge build fortran/test_basic.f90 --name manforge_test_basic

# List compiled modules
uv run manforge list

# Remove compiled artifacts
uv run manforge clean
```

### Option B: Makefile

```bash
make fortran-build-umat   # J2 UMAT
make fortran-build        # smoke-test module
```

### Option C: f2py directly

```bash
cd fortran/
python -m numpy.f2py -c abaqus_stubs.f90 j2_isotropic_3d.f90 -m j2_isotropic_3d
```

All three options produce `j2_isotropic_3d.cpython-*.so` (Linux) or `.pyd` (Windows) in the `fortran/` directory.

### Subroutine naming convention

| Subroutine | Description |
|------------|-------------|
| `j2_isotropic_3d` | Core logic — matches Python `J2Isotropic3D` class |
| `umat` | ABAQUS entry point — required name for ABAQUS solver integration |
| `j2_isotropic_3d_elastic_stiffness` | Standalone elastic stiffness for component checks |

f2py module name = Python model name: `j2_isotropic_3d`. This pattern extends
naturally to future models: `FortranModule("j2_kinematic_3d")`, etc.

---

## Docker build (recommended)

A reproducible gfortran + Python 3.12 environment is provided via `Dockerfile`
at the project root:

```bash
# Build the image
docker build -t manforge-fortran .

# Compile and run Fortran tests inside the container
docker run --rm -v $(pwd):/workspace -w /workspace manforge-fortran \
    bash -c "make fortran-build-umat && make fortran-test-umat"
```

---

## Material parameters convention

`PROPS` array order (matches `J2Isotropic3D.param_names`):

| Index (1-based) | Name | Unit | Description |
|-----------------|------|------|-------------|
| 1 | E | MPa | Young's modulus |
| 2 | nu | — | Poisson's ratio |
| 3 | sigma_y0 | MPa | Initial yield stress |
| 4 | H | MPa | Linear isotropic hardening modulus |

`STATEV` array:

| Index (1-based) | Name | Description |
|-----------------|------|-------------|
| 1 | ep | Equivalent plastic strain |

---

## Voigt notation convention

Stress/strain components follow the ABAQUS convention (same as `manforge`):

```
STRESS / STRAN = [s11, s22, s33, s12, s13, s23]
                  (3 normal,  3 engineering shear: gamma = 2*eps_shear)
```

NTENS = 6 for 3D full stress state (NDI=3, NSHR=3).

---

## Cross-validation with Python

`FortranModule` wraps the f2py module with automatic float64 conversion.
All subroutines — the full routine and any sub-component — are called through
the same `call(name, *args)` pattern.

```python
import numpy as np
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.simulation import PythonIntegrator
from manforge.verification import FortranModule

model   = J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
fortran = FortranModule("j2_isotropic_3d")
```

### Full routine comparison

```python
dstran = np.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])

# Fortran
stress_f, ep_f, ddsdde_f = fortran.call(
    "j2_isotropic_3d",
    model.E, model.nu, model.sigma_y0, model.H,
    np.zeros(6), 0.0,   # stress_in, ep_in
    dstran,
)

# Python
result_py = stress_update(
    model, dstran, np.zeros(6), model.initial_state()
)

# Compare
np.testing.assert_allclose(np.array(result_py.stress), stress_f, rtol=1e-6)
np.testing.assert_allclose(np.array(result_py.ddsdde), np.array(ddsdde_f), rtol=1e-5)
```

### Component-level comparison

When the full routine results differ, isolate the faulty sub-component.
`j2_isotropic_3d.f90` already includes `j2_isotropic_3d_elastic_stiffness`
as a reference example:

```fortran
! In j2_isotropic_3d.f90 — already present
subroutine j2_isotropic_3d_elastic_stiffness(E, nu, C)
    double precision, intent(in)  :: E, nu
    double precision, intent(out) :: C(6,6)
    ! ... identical to the C assembly inside j2_isotropic_3d
end subroutine
```

After compiling (`manforge build` or `make fortran-build-umat`), the call pattern is identical:

```python
# Fortran sub-component
C_f = fortran.call("j2_isotropic_3d_elastic_stiffness", model.E, model.nu)

# Python reference
C_py = model.elastic_stiffness()

np.testing.assert_allclose(np.array(C_py), np.array(C_f), rtol=1e-12)
```

The same pattern applies to any internal quantity — yield function, flow
direction, tangent sub-block, etc.  The elastic stiffness is the best first
check because a mismatch here propagates into every downstream calculation.

To add a new component check: write a standalone subroutine in the `.f90` file,
recompile, then call it via `fortran.call("subroutine_name", *args)` and compare
with the corresponding Python calculation using `np.testing.assert_allclose`.
