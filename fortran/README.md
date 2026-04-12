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

Use `UMATVerifier` for the standard workflow — it auto-generates test cases
and runs a multi-step strain history comparison:

```python
import manforge  # enables JAX float64
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.verification import UMATVerifier

model  = J2Isotropic3D()
params = {"E": 210000.0, "nu": 0.3, "sigma_y0": 250.0, "H": 1000.0}

verifier = UMATVerifier(model, "manforge_umat")
result   = verifier.run(params)
print(result.summary())
```

`run` performs two phases automatically:

1. **Single-step** — estimates the yield strain from the model's yield
   surface, generates 5 test cases (elastic, plastic uniaxial, multiaxial,
   shear, pre-stressed), and compares with
   :func:`~manforge.verification.compare.compare_solvers`.
2. **Multi-step** — runs a tension-unload-compression cycle through both
   Python and Fortran with independent state propagation, comparing stress,
   tangent, and state at every step.

For custom loading, pass your own strain history:

```python
import numpy as np
strain = np.linspace(0.0, 5e-3, 100)          # shape (N,) uniaxial
result = verifier.run(params, strain_history=strain)
```

For lower-level access, `FortranUMAT.call` follows the standard solver
protocol `(strain_inc, stress_n, state_n, params) -> (stress_new, state_new, ddsdde)`
and can be used directly with `compare_solvers`.

---

## Component-level verification

`UMATVerifier` is a black-box acceptance test — it compares `_run` inputs and
outputs end-to-end.  When it reports a failure, the next step is to isolate
*which sub-component* is wrong.

### Pattern: expose internal subroutines for direct comparison

Add a standalone f2py-callable subroutine for each component you want to
inspect, recompile, then compare against the matching Python method.

`umat_j2.f90` already includes `umat_j2_elastic_stiffness` as a reference
example:

```fortran
! In umat_j2.f90 — already present
subroutine umat_j2_elastic_stiffness(E, nu, C)
    double precision, intent(in)  :: E, nu
    double precision, intent(out) :: C(6,6)
    ! ... identical to the C assembly inside umat_j2_run
end subroutine
```

After recompiling (`make fortran-build-umat`), call it from Python:

```python
import numpy as np
import manforge_umat
import manforge  # enables JAX float64
from manforge.models.j2_isotropic import J2Isotropic3D

model  = J2Isotropic3D()
params = {"E": 210000.0, "nu": 0.3, "sigma_y0": 250.0, "H": 1000.0}

# Fortran sub-component
C_fortran = np.array(manforge_umat.umat_j2_elastic_stiffness(params["E"], params["nu"]))

# Python reference
C_python = np.array(model.elastic_stiffness(params))

np.testing.assert_allclose(C_fortran, C_python, rtol=1e-12)
```

### Extending the pattern

The same approach works for any internal quantity — yield function value, flow
direction, consistent tangent sub-block, etc.  Add a subroutine that outputs
the quantity of interest, recompile, and compare against the Python model's
corresponding method (`yield_function`, `hardening_increment`, …).

The elastic stiffness is the best first check because it is the simplest
component and a mismatch here propagates into every downstream calculation.
