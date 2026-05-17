# tests/benchmarks/

**Definition**: A benchmark here is a numerical-equivalence harness — it asserts that
two independent implementations of the same constitutive physics produce results
within specified tolerances.

## Structure

```
benchmarks/<model>/
    conftest.py                      # scenario factory
    test_analytical_vs_numerical.py  # Path A: analytical vs framework NR
    test_numerical_vs_fortran.py     # Path B: Python NR vs compiled Fortran UMAT
```

## What does NOT belong here

Fortran-integration contract tests (binding registries, param-order contracts, f2py
sanity) are **not** numerical equivalence harnesses.  They live in `tests/integration/`
at the mirrored position of their source module.

## Tolerance policy

| quantity | tolerance |
|----------|-----------|
| stress   | atol=1e-6 (absolute, MPa-scale) |
| Δλ, ep   | atol=1e-10 (absolute) |
| tangent  | max_rel_err < 1e-5 (relative) |

## Running

```bash
make test-benchmarks              # Path A (no Fortran required)
make fortran-build-umat && make test-benchmarks-fortran  # Path B
```

## Adding a new model benchmark

1. Create `benchmarks/<model>/`.
2. Add a `conftest.py` with scenario fixtures.
3. Implement `test_analytical_vs_numerical.py` and/or `test_numerical_vs_fortran.py`.
4. Use `@pytest.mark.fortran` on any test that requires a compiled `.so`.
