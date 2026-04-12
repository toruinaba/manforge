# fortran/ — Fortran UMAT templates and build guide

This directory contains Fortran source files for implementing and validating
ABAQUS UMAT constitutive subroutines that can be cross-validated against the
Python `manforge` implementation.

## Files

| File | Description |
|------|-------------|
| `wrapper.f90` | UMAT skeleton + ISO_C_BINDING wrapper template |
| `test_basic.f90` | *(Step 9)* Simple elastic subroutine for f2py smoke test |
| `abaqus_stubs.f90` | *(Step 10)* Stubs for ABAQUS internal functions (SINV, SPRINC, …) |
| `umat_j2.f90` | *(Step 10)* Full J2 isotropic hardening UMAT implementation |

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

### Compile `wrapper.f90` into a Python extension module

```bash
cd fortran/
python -m numpy.f2py -c wrapper.f90 -m manforge_umat
```

This produces `manforge_umat.cpython-*.so` (Linux) or `.pyd` (Windows).

### Import and call from Python

```python
import numpy as np
import manforge_umat   # the compiled module

ntens  = 6
nstatv = 1
nprops = 4

stress  = np.zeros(ntens)
statev  = np.zeros(nstatv)
stran   = np.zeros(ntens)
dstran  = np.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
props   = np.array([210000.0, 0.3, 250.0, 1000.0])  # E, nu, sigma_y0, H
dtime   = 1.0

ddsdde = np.zeros((ntens, ntens))
sse = spd = scd = 0.0

manforge_umat.umat_j2_c(
    stress, statev, ddsdde,
    sse, spd, scd,
    stran, dstran, dtime,
    props, nprops, ntens, nstatv,
)

print("sigma11 =", stress[0])
print("DDSDDE[0,0] =", ddsdde[0, 0])
```

---

## Build with ctypes

An alternative to f2py is to compile a shared library and call it via `ctypes`:

```bash
gfortran -shared -fPIC -O2 -o libmanforge_umat.so wrapper.f90
```

```python
import ctypes
import numpy as np

lib = ctypes.CDLL("./libmanforge_umat.so")
# (argument types must be declared manually — see fortran_bridge.py in Step 10)
```

---

## Docker build (recommended)

A reproducible gfortran + Python 3.12 environment will be set up in **Step 9**.

Once `Dockerfile` is available at the project root:

```bash
# Build the image
docker build -t manforge-fortran .

# Compile and run Fortran tests inside the container
docker run --rm -v $(pwd):/workspace -w /workspace manforge-fortran \
    bash -c "cd fortran && python -m numpy.f2py -c wrapper.f90 -m manforge_umat \
             && python -m pytest tests/test_fortran_basic.py -v"
```

---

## Material parameters convention

`PROPS` array order (must match `manforge.models.j2_isotropic.J2Isotropic3D.param_names`):

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

After Step 10, use `manforge.verification.fortran_bridge.compare_with_fortran`
to verify that the Fortran UMAT matches the Python return mapping:

```python
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.verification.fortran_bridge import FortranUMAT, compare_with_fortran

model = J2Isotropic3D()
umat  = FortranUMAT("manforge_umat", subroutine_name="umat_j2")

test_cases = [
    {"strain_inc": [2e-3, 0, 0, 0, 0, 0], "stress_n": [0]*6,
     "state_n": {"ep": 0.0}, "params": {...}},
]
result = compare_with_fortran(model, umat, test_cases)
print("All cases passed:", result.passed)
```
