# Fortran UMAT クロス検証

## モデルに UMAT を宣言する

UMAT まで行ったモデルは、ファミリの `verification.py` に UMAT を宣言する。ビルドも検査も、この宣言を読む。

```python
# src/returnmap/models/yu_kinematic/verification.py
from returnmap.verification import Umat

UMATS = [
    Umat(YUKinematic3D, module="yu_kinematic_3d",
         sources=["fortran/yu_kinematic_3d.f90"],        # モデルのモジュールがあるディレクトリから
         libraries=["lapack", "blas"],
         integrator_options={"parse_umat_failure": ...}), # UMAT 自身の収束フラグを読む
]
# PROPS の順が param_names と違う、または一部しか受け取らない UMAT は props=["E", "nu", ...] で宣言する
```

```bash
uv run returnmap build --family returnmap.models.yu_kinematic    # そのファミリの UMAT を全部
uv run returnmap build --family returnmap.models                 # 全ファミリ
```

```python
from returnmap.verification import check_model

report = check_model(YUKinematic3D.at_defaults())
print(report.summary())
# ...
#   ok   UMAT   stress 2.5e-15, state 1.9e-15 (tol 1e-06), tangent 6.5e-15 (tol 1e-05) over 3 histories (yu_kinematic_3d)
```

`check_model` は、Python のモデルを検査したのと同じ載荷履歴で UMAT を走らせ、全ステップで応力・状態・接線を比べる。ビルドしていなければ `not built` と出るだけで、不合格にはならない。引数の規約は `FortranIntegrator.from_model` と同じ: パラメータを `param_names` の順、応力、状態変数を宣言の順、ひずみ増分。

## 1 つずつ行う

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
