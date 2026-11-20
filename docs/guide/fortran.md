# Fortran UMAT クロス検証

**Build → Load → Integrate → Crosscheck**:

```bash
uv run returnmap build --abaqus-stubs src/returnmap/models/j2_isotropic/fortran/j2_isotropic_3d.f90 --name j2_isotropic_3d
uv run returnmap list   # 確認
```

| 指定 | 意味 |
|---|---|
| `--abaqus-stubs` | ABAQUS のユーティリティルーチン (SINV、SPRINC、ROTSIG) の代わりを一緒にコンパイルする。UMAT のソースを ABAQUS の外でリンクするのに要る。スタブはパッケージに入っていて、`returnmap stubs` がそのパスを出す (f2py やコンパイラを直接呼ぶとき用) |
| `--link lapack --link blas` | リンクするライブラリ (LAPACK を呼ぶ UMAT など)。繰り返し指定できる |
| `--output-dir` | 出力先。既定はプロジェクト直下の `fortran/` |

**自分のリポジトリで使うとき** — returnmap を入れてあれば、UMAT のソースとテスト以外に用意するものは無い。スタブは上のとおりパッケージから取れる。テストの側は、returnmap が pytest のプラグインを持っていて (入れてあれば自動で効く)、`fortran` / `slow` マーカーを登録し、`fortran/` を `sys.path` に足す。`conftest.py` にそのための記述は要らない:

```python
import pytest

pytest.importorskip("my_umat", reason="build it with: returnmap build --abaqus-stubs ... --name my_umat")

@pytest.mark.fortran
def test_python_matches_umat():
    ...
```

```python
from returnmap.simulation import FortranIntegrator, FortranModule
from returnmap.verification import crosscheck

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
