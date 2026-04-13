# fortran/ — Fortran UMAT build guide

This directory contains Fortran source files for implementing and validating
ABAQUS UMAT constitutive subroutines that can be cross-validated against the
Python `manforge` implementation.

## Files

| File | Description |
|------|-------------|
| `umat_j2.f90` | Full J2 isotropic hardening UMAT (`umat_j2_run` + ABAQUS interface + `umat_j2_elastic_stiffness` for component-level checks) |
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
python -m numpy.f2py -c abaqus_stubs.f90 umat_j2.f90 -m manforge_umat
```

Or via the Makefile from the project root:

```bash
make fortran-build-umat
```

This produces `manforge_umat.cpython-*.so` (Linux) or `.pyd` (Windows).

### Call directly from Python

```python
import numpy as np
import manforge_umat   # the compiled module

stress_out, ep_out, ddsdde = manforge_umat.umat_j2_run(
    210000.0, 0.3, 250.0, 1000.0,  # E, nu, sigma_y0, H
    np.zeros(6),                    # stress_in
    0.0,                            # ep_in
    np.array([2e-3, 0, 0, 0, 0, 0], dtype=np.float64),  # dstran
)

print("sigma11 =", stress_out[0])
print("DDSDDE[0,0] =", ddsdde[0, 0])
```

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

The primary workflow uses `FortranUMAT` to call Fortran subroutines with
automatic float64 conversion, then compares results explicitly against the
Python reference.  This approach works identically for `_run` and for any
individual sub-component.

```python
import numpy as np
import jax.numpy as jnp
import manforge  # enables JAX float64
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.core.return_mapping import return_mapping
from manforge.verification import FortranUMAT

model  = J2Isotropic3D()
params = {"E": 210000.0, "nu": 0.3, "sigma_y0": 250.0, "H": 1000.0}

fortran = FortranUMAT("manforge_umat")
```

### Full routine comparison (`_run`)

```python
dstran = np.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])

# Fortran
stress_f, ep_f, ddsdde_f = fortran.run(
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

When `_run` results differ, narrow down which sub-component is wrong.
Add a standalone subroutine to the `.f90` file, recompile, and compare using
the same pattern.  `umat_j2.f90` already includes `umat_j2_elastic_stiffness`
as a reference example:

```fortran
! In umat_j2.f90 — already present
subroutine umat_j2_elastic_stiffness(E, nu, C)
    double precision, intent(in)  :: E, nu
    double precision, intent(out) :: C(6,6)
    ! ... identical to the C assembly inside umat_j2_run
end subroutine
```

After recompiling (`make fortran-build-umat`), the call pattern is identical:

```python
# Fortran sub-component
C_f = fortran.call("umat_j2_elastic_stiffness", params["E"], params["nu"])

# Python reference
C_py = model.elastic_stiffness(params)

np.testing.assert_allclose(np.array(C_py), np.array(C_f), rtol=1e-12)
```

The same pattern applies to any internal quantity — yield function, flow
direction, tangent sub-block, etc.  The elastic stiffness is the best first
check because a mismatch here propagates into every downstream calculation.

---

## Batch verification (convenience utility)

`UMATVerifier` auto-generates 5 single-step test cases and a 35-step
tension-unload-compression history, then runs all comparisons at once.
Useful for a quick overall pass/fail check, but the comparison logic is
opaque — prefer the explicit `FortranUMAT` approach when debugging.

```python
from manforge.verification import UMATVerifier

verifier = UMATVerifier(model, "manforge_umat")
result   = verifier.run(params)
print(result.summary())
```

For a custom strain history:

```python
import numpy as np
strain = np.linspace(0.0, 5e-3, 100)   # shape (N,) uniaxial
result = verifier.run(params, strain_history=strain)
```
