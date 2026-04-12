# fortran/ — Fortran UMAT build guide

This directory contains Fortran source files for implementing and validating
ABAQUS UMAT constitutive subroutines that can be cross-validated against the
Python `manforge` implementation.

## Files

| File | Description |
|------|-------------|
| `umat_j2.f90` | Full J2 isotropic hardening UMAT (`umat_j2_run` + ABAQUS interface) |
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

Use `FortranUMAT` from `manforge.verification.fortran_bridge` to wrap the
compiled module, then pass it to `compare_solvers`:

```python
import numpy as np
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.verification.fortran_bridge import FortranUMAT
from manforge.verification.compare import compare_solvers

model = J2Isotropic3D()
umat  = FortranUMAT("manforge_umat", model)

params = {"E": 210000.0, "nu": 0.3, "sigma_y0": 250.0, "H": 1000.0}
test_cases = [
    {
        "strain_inc": np.array([2e-3, 0, 0, 0, 0, 0]),
        "stress_n":   np.zeros(6),
        "state_n":    {"ep": 0.0},
        "params":     params,
    },
]

result = compare_solvers(umat.make_python_solver(), umat.call, test_cases)
print("All cases passed:", result.passed)
```

`FortranUMAT.call` follows the standard solver protocol
`(strain_inc, stress_n, state_n, params) -> (stress_new, state_new, ddsdde)`,
so it can be used anywhere a solver callable is expected.
