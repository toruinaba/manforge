# Fortran UMAT クロス検証

**Build → Load → Integrate → Crosscheck**:

```bash
uv run manforge build fortran/abaqus_stubs.f90 fortran/j2_isotropic_3d.f90 --name j2_isotropic_3d
uv run manforge list   # 確認
```

```python
from manforge.simulation import FortranIntegrator, FortranModule
from manforge.verification import crosscheck

fortran = FortranModule("j2_isotropic_3d")
fc_int = FortranIntegrator.from_model(fortran, "j2_isotropic_3d", model)

result = crosscheck(PythonIntegrator(model), fc_int, load)
assert result.passed, result.summary()
```

UMAT のサブルーチンが自分の収束フラグを返す場合は、`parse_umat_failure` で読ませる。指定しないと UMAT 側の非収束は見えず、収束しなかった反復値が解として比較されてしまう:

```python
fc_int = FortranIntegrator.from_model(
    fortran, "my_model_core", model,
    param_fn=lambda: (model.E, model.nu, ...),          # PROPS の順が param_names と違う場合
    parse_umat_failure=lambda ret: None if ret[12] else f"fail_code={int(ret[14])}",
)
```

ビルド手順の詳細は `fortran/README.md`、通しの実例は `examples/example_fortran_crosscheck.py`。
