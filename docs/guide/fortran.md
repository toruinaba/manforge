# Fortran UMAT クロス検証

**Build → Load → Integrate → Crosscheck**:

```bash
uv run manforge build fortran/abaqus_stubs.f90 fortran/j2_isotropic_3d.f90 --name j2_isotropic_3d
uv run manforge list   # 確認
```

```python
from manforge.simulation import FortranIntegrator, FortranModule
from manforge.verification import CrosscheckStrainDriver

fortran = FortranModule("j2_isotropic_3d")
fc_int = FortranIntegrator.from_model(fortran, "j2_isotropic_3d", model)

cc = CrosscheckStrainDriver(PythonIntegrator(model), fc_int)
cr = cc.run(load)
assert cr.passed, f"max stress rel err: {cr.max_stress_rel_err:.2e}"
```

ビルド手順の詳細は `fortran/README.md`、通しの実例は `examples/example_fortran_crosscheck.py`。
