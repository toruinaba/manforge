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
- **Docker** (recommended for a reproducible environment — see below)

Check availability:

```bash
gfortran --version
python -m numpy.f2py --version
```

---

## Build with f2py

### Compile the J2 UMAT into a Python extension module

```bash
cd fortran/
python -m numpy.f2py -c abaqus_stubs.f90 j2_isotropic_3d.f90 -m j2_isotropic_3d
```

Or via the Makefile from the project root:

```bash
make fortran-build-umat
```

This produces `j2_isotropic_3d.cpython-*.so` (Linux) or `.pyd` (Windows).

### Subroutine naming convention

| Subroutine | Description |
|------------|-------------|
| `j2_isotropic_3d` | Core logic — matches Python `J2Isotropic3D` class |
| `umat` | ABAQUS entry point — required name for ABAQUS solver integration |
| `j2_isotropic_3d_elastic_stiffness` | Standalone elastic stiffness for component checks |

f2py module name = Python model name: `j2_isotropic_3d`. This pattern extends
naturally to future models: `FortranUMAT("j2_kinematic_3d")`, etc.

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

`FortranUMAT` wraps the f2py module with automatic float64 conversion.
All subroutines — the full routine and any sub-component — are called through
the same `call(name, *args)` pattern.

```python
import numpy as np
import jax.numpy as jnp
import manforge  # enables JAX float64
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.core.return_mapping import return_mapping
from manforge.verification import FortranUMAT

model  = J2Isotropic3D()
params = {"E": 210000.0, "nu": 0.3, "sigma_y0": 250.0, "H": 1000.0}

fortran = FortranUMAT("j2_isotropic_3d")
```

### Full routine comparison

```python
dstran = np.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])

# Fortran
stress_f, ep_f, ddsdde_f = fortran.call(
    "j2_isotropic_3d",
    params["E"], params["nu"], params["sigma_y0"], params["H"],
    np.zeros(6), 0.0,   # stress_in, ep_in
    dstran,
)

# Python
stress_py, state_py, ddsdde_py = return_mapping(
    model, jnp.array(dstran), jnp.zeros(6), model.initial_state(), params
)

# Compare
np.testing.assert_allclose(np.array(stress_py), stress_f, rtol=1e-6)
np.testing.assert_allclose(np.array(ddsdde_py), np.array(ddsdde_f), rtol=1e-5)
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

After compiling (`make fortran-build-umat`), the call pattern is identical:

```python
# Fortran sub-component
C_f = fortran.call("j2_isotropic_3d_elastic_stiffness", params["E"], params["nu"])

# Python reference
C_py = model.elastic_stiffness(params)

np.testing.assert_allclose(np.array(C_py), np.array(C_f), rtol=1e-12)
```

The same pattern applies to any internal quantity — yield function, flow
direction, tangent sub-block, etc.  The elastic stiffness is the best first
check because a mismatch here propagates into every downstream calculation.

To add a new component check: write a standalone subroutine in the `.f90` file,
recompile, then call it via `fortran.call("subroutine_name", *args)` and compare
with the corresponding Python calculation using `np.testing.assert_allclose`.
