# mymodels — a model family built on returnmap

The layout a repository of your own models takes, complete and runnable:
J2 plasticity with Voce hardening, from the Python model to a verified UMAT.
The guide walks through it step by step (in Japanese):
<https://toruinaba.github.io/returnmap/guide/adding-a-model.html>.

```
pyproject.toml                  returnmap pinned to a release tag
conftest.py                     only for running from a returnmap checkout
src/mymodels/voce/
  README.md                     the formulation
  model.py                      J2Voce (the physics), J2Voce1D, J2Voce3D (+ hand-written solver)
  verification.py               what is checked: MODELS, UMATS
  fortran/voce_3d.f90           the UMAT: the port of J2Voce3D
  tests/test_verification.py    runs the common checks
fortran/                        where `returnmap build` puts compiled modules (created on build)
```

Copy the directory out of the checkout and it is a project of its own:

```bash
uv sync --extra dev
uv run pytest                                  # 2 passed, 1 skipped (the UMAT is not built)

uv sync --extra dev --extra fortran            # needs gfortran
uv run returnmap build --family mymodels       # compiles what verification.py declares
uv run pytest                                  # 3 passed
```
