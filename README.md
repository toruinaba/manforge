# manforge

Fortran UMAT (Abaqus) の検証・フィッティング用 Python フレームワーク。降伏関数と硬化則を実装するだけで、return mapping / consistent tangent (autograd 自動微分) / パラメータフィッティング / Fortran クロス検証がすべて使える。

- autograd による **consistent tangent 自動計算** — 手動でアルゴリズム接線を導出する必要なし
- **汎用 NR ソルバ** — スカラー Δλ / ベクトル `[σ, Δλ, q]` の 2 経路を自動選択
- **多次元対応** — 3D ソリッド / 平面ひずみ / 平面応力 / 1D を `StressDimension` で統一管理
- **組み込みモデル** — J2 等方硬化 / Armstrong-Frederick / Ohno-Wang を同梱

---

## Installation

```bash
pip install manforge
pip install "manforge[dev]"      # pytest 含む
pip install "manforge[examples]" # matplotlib 含む
pip install "manforge[fortran]"  # meson/ninja 含む (Fortran ビルド用)
```

開発環境 (ソースから):

```bash
uv sync --extra dev          # テスト依存含む
uv sync --all-extras         # 全 extras

# Fortran モジュールのビルド (gfortran 必須)
uv sync --extra fortran
uv run manforge build fortran/abaqus_stubs.f90 fortran/j2_isotropic_3d.f90 --name j2_isotropic_3d
```

---

## Quick Start

```python
import numpy as np
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.simulation import StrainDriver, PythonIntegrator
from manforge.simulation.types import FieldHistory, FieldType

model = J2Isotropic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)

strain_history = np.zeros((100, 6))
strain_history[:, 0] = np.linspace(0.0, 5e-3, 100)  # ε11 を増加
load = FieldHistory(FieldType.STRAIN, "Strain", strain_history)

result = StrainDriver(PythonIntegrator(model)).run(load)
print(f"Final σ11: {result.stress[-1, 0]:.1f} MPa")
```

詳しい使用例は `examples/` を参照。

---

## Concepts

```
MaterialModel  ──defines──▶  yield_function / update_state / state_residual
      │
      ▼
  Integrator   ──wraps──▶  stress_update(strain_inc, stress_n, state_n) → StressUpdateResult
      │                    return_mapping(stress_trial, state_n)         → ReturnMappingResult
      ▼
    Driver     ──steps──▶  run(load) → DriverResult
                            iter_run(load) → Iterator[DriverStep]
```

**Integrator の種類**

| クラス | 内部ソルバ | 用途 |
|--------|----------|------|
| `PythonIntegrator(model)` | auto (user_defined があれば使用、なければ NR) | 通常用途 |
| `PythonNumericalIntegrator(model)` | numerical_newton 固定 | NR 強制 |
| `PythonAnalyticalIntegrator(model)` | user_defined 固定 | 閉形式解と比較する場合 |
| `FortranIntegrator.from_model(fortran, subroutine, model)` | Fortran f2py | UMAT クロス検証 |

**Voigt 規約** — 3D ソリッドは `[σ11, σ22, σ33, σ12, σ13, σ23]` (6 成分)。剪断成分には Mandel スケーリング (×√2) を内部適用。

---

## User Guide

### 5.1 モデルを定義する

応力状態に合わせた基底クラスを選択する:

| 基底クラス | 対応 StressDimension | NTENS |
|-----------|----------------|-------|
| `MaterialModel3D` | `SOLID_3D`, `PLANE_STRAIN` | 6, 4 |
| `MaterialModelPS` | `PLANE_STRESS` | 3 |
| `MaterialModel1D` | `UNIAXIAL_1D` | 1 |

**Explicit-state (J2 型)** — 状態変数を `Explicit` として宣言し `update_state` に closed-form 更新を書く。NR はスカラー Δλ のみ解く。

```python
import autograd.numpy as anp
from manforge.core import Implicit, Explicit, NTENS, SCALAR
from manforge.core.material import MaterialModel3D

class MyModel(MaterialModel3D):
    param_names = ["E", "nu", "sigma_y0", "H"]
    ep = Explicit(shape=SCALAR, doc="equivalent plastic strain")

    def __init__(self, *, E, nu, sigma_y0, H):
        super().__init__()
        self.E = E; self.nu = nu; self.sigma_y0 = sigma_y0; self.H = H

    def yield_function(self, state):
        return self._vonmises(state["stress"]) - (self.sigma_y0 + self.H * state["ep"])

    def update_state(self, dlambda, state_n, state_trial):
        return [self.ep(state_n["ep"] + dlambda)]
```

`elastic_stiffness` は `self.E` と `self.nu` から自動生成。状態依存の場合のみ `elastic_stiffness(self, state=None)` を override する。参照実装: `src/manforge/models/j2_isotropic.py`

**要素タイプの切り替え**

```python
from manforge.core.dimension import PLANE_STRAIN, PLANE_STRESS, UNIAXIAL_1D

model_pe = J2Isotropic3D(PLANE_STRAIN, E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)
model_ps = J2IsotropicPS(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)
model_1d = J2Isotropic1D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)
```

| インスタンス | NTENS | 想定要素 |
|------------|-------|---------|
| `SOLID_3D` | 6 | C3D8 等 3D ソリッド |
| `PLANE_STRAIN` | 4 | CPE4 / CAX4 |
| `PLANE_STRESS` | 3 | CPS4 / シェル |
| `UNIAXIAL_1D` | 1 | 1D トラス / フィッティング用 |

**Implicit-state (Ohno-Wang 型)** — `update_state` が closed-form で解けない場合は状態変数を `Implicit` として宣言し `state_residual` に後方 Euler 残差を書く。`stress = Implicit(shape=NTENS)` を宣言すると σ も NR 未知数になる。

```python
class MyImplicitModel(MaterialModel3D):
    param_names = ["E", "nu", "sigma_y0", "C_k", "gamma"]
    stress = Implicit(shape=NTENS, doc="Cauchy stress (NR unknown)")
    alpha  = Implicit(shape=NTENS, doc="backstress tensor")
    ep     = Implicit(shape=SCALAR, doc="equivalent plastic strain")

    def yield_function(self, state):
        xi = state["stress"] - state["alpha"]
        return self._vonmises(xi) - self.sigma_y0

    def state_residual(self, state_new, dlambda, state_n, state_trial):
        R_alpha = ...         # shape (ntens,)
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        return [self.alpha(R_alpha), self.ep(R_ep)]
        # self.stress を省略すると framework が associative R_stress を自動装着
```

参照実装: `src/manforge/models/ow_kinematic.py`

**Δλ 行残差のカスタム指定 (粘塑性)** — `state_residual` に `self.dlambda(R_dl)` を含めると NR 残差の Δλ 行を差し替えられる。省略時は `yield_function(state) = 0`。

**user_defined フック** — 閉形式の return mapping が存在する場合は `user_defined_return_mapping(self, stress_trial, C, state_n)` を実装すると `method="auto"` で自動使用される。`None` を返すと汎用 NR にフォールバック。`J2Isotropic3D` は両フック実装済み。

**組み込みモデル**

```python
from manforge.models import AFKinematic3D, OWKinematic3D

model_af = AFKinematic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, C_k=8_000.0, gamma=80.0)
model_ow = OWKinematic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, C_k=8_000.0, gamma=0.05)
```

---

### 5.2 シミュレーションを実行する

```python
import numpy as np
from manforge.simulation import StrainDriver, StressDriver, PythonIntegrator
from manforge.simulation.types import FieldHistory, FieldType

# ひずみ制御
strain_history = np.zeros((100, 6))
strain_history[:, 0] = np.linspace(0, 5e-3, 100)
load = FieldHistory(FieldType.STRAIN, "Strain", strain_history)
result = StrainDriver(PythonIntegrator(model)).run(load)
result.stress    # np.ndarray (N, ntens)
result.strain    # np.ndarray (N, ntens)

# 状態変数を収集する
result = StrainDriver(PythonIntegrator(model)).run(
    load,
    collect_state={"ep": FieldType.STRAIN}
)
result.fields["ep"].data   # np.ndarray (N,)

# 応力制御
stress_target = np.zeros((100, 6))
stress_target[:, 0] = np.linspace(0, 300.0, 100)
result = StressDriver(PythonIntegrator(model)).run(
    FieldHistory(FieldType.STRESS, "Stress", stress_target)
)
```

`DriverResult.step_results` は `list[StressUpdateResult]`。各ステップの詳細にアクセスできる:

```python
step = result.step_results[15]
print(step.dlambda, step.n_iterations)
```

**iter_run** — ステップごとに `DriverStep` を受け取るジェネレータ。途中で分岐・打切りができる:

```python
driver = StrainDriver(PythonIntegrator(model))
for step in driver.iter_run(load):
    if step.result.is_plastic:
        print(f"Plasticity onset at step {step.i}, ep={step.result.state['ep']:.4e}")
        break
```

`StressDriver` では `step.n_outer_iter` / `step.residual_inf` で外側 NR の収束状況を確認できる。非収束時は既定 `RuntimeError`; `raise_on_nonconverged=False` で `step.converged=False` の step を yield して consumer が判断することもできる。

---

### 5.3 パラメータフィッティング

```python
from manforge.simulation.driver import StrainDriver
from manforge.fitting import fit_params

fit_config = {
    "sigma_y0": (200.0, (100.0, 500.0)),  # (初期値, (下限, 上限))
    "H":        (500.0, (0.0,  5000.0)),
}
result = fit_params(
    model,
    driver_factory=lambda i: StrainDriver(i),
    exp_data=exp_data,
    fit_config=fit_config,
    fixed_params={"E": 210_000.0, "nu": 0.3},
    method="L-BFGS-B",
)
print(result.params, result.residual)
```

`exp_data` は `{"strain": ..., "stress": ...}` の dict。詳細: `examples/example_fit_j2.py`

---

### 5.4 検証

新しいモデルを実装するときの推奨ワークフロー:

1. **数値解で収束確認** — `PythonNumericalIntegrator` で残差が 0 に落ちることを確認
2. **AD 値との照合** — `JacobianBlocks` でヤコビアン各ブロックを要素ごとに検証
3. **FD tangent チェック** — `check_tangent` で consistent tangent の精度を確認
4. **Fortran への移植** — 検証済みブロック構造をそのままサブルーチン構成に対応

**JacobianBlocks**

```python
import numpy.testing as npt
from manforge.simulation.integrator import PythonIntegrator
from manforge.verification import ad_jacobian_blocks

result = PythonIntegrator(model).stress_update(deps, stress_n, state_n)
jac = ad_jacobian_blocks(model, result, state_n)

npt.assert_allclose(jac.dyield_dsigma, my_n, rtol=1e-8)
npt.assert_allclose(float(jac.dyield_ddlambda), -H, rtol=1e-8)

# implicit state 付きモデル (例: Ohno-Wang)
jac.dstate_dsigma["alpha"]        # ∂R_alpha/∂σ
jac.dstate_ddlambda["ep"]         # ∂R_ep/∂Δλ
```

**check_tangent**

```python
from manforge.verification.fd_check import check_tangent

result = check_tangent(PythonIntegrator(model), stress_n, state_n, strain_inc)
print(result.passed, result.max_rel_err)
```

**CrosscheckStrainDriver**

```python
from manforge.simulation import PythonNumericalIntegrator, PythonAnalyticalIntegrator
from manforge.simulation.types import FieldHistory, FieldType
from manforge.verification import CrosscheckStrainDriver, generate_strain_history

history = generate_strain_history(model)
load = FieldHistory(FieldType.STRAIN, "eps", history)
cc = CrosscheckStrainDriver(
    PythonNumericalIntegrator(model),
    PythonAnalyticalIntegrator(model),
)
result = cc.run(load)
print(f"Passed: {result.passed}  max_stress_err={result.max_stress_rel_err:.2e}")
```

`iter_run` で失敗ステップを詳細デバッグできる:

```python
from manforge.verification import compare_jacobians

for cr in cc.iter_run(load):
    if not cr.passed:
        jac = compare_jacobians(model, cr.result_a, cr.result_b, cr.state_n)
        print(jac.blocks)
        break
```

---

### 5.5 Fortran UMAT クロス検証

**Build → Load → Integrate → Crosscheck** の 4 ステップ:

```bash
# Build
uv sync --extra fortran
uv run manforge build fortran/abaqus_stubs.f90 fortran/j2_isotropic_3d.f90 --name j2_isotropic_3d
uv run manforge list   # 確認
```

```python
import numpy as np
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.simulation import FortranIntegrator, StrainDriver
from manforge.simulation.integrator import PythonIntegrator
from manforge.simulation.types import FieldHistory, FieldType
from manforge.verification import FortranModule, CrosscheckStrainDriver

model   = J2Isotropic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)
fortran = FortranModule("j2_isotropic_3d")

# Fortran UMAT を StrainDriver で実行
fc_int = FortranIntegrator.from_model(fortran, "j2_isotropic_3d", model)
strain_data = np.zeros((50, model.ntens))
strain_data[:, 0] = np.linspace(0.0, 5e-3, 50)
load = FieldHistory(FieldType.STRAIN, "eps", strain_data)
result = StrainDriver(fc_int).run(load)

# Python 実装との照合
cc = CrosscheckStrainDriver(PythonIntegrator(model), fc_int)
cr = cc.run(load)
assert cr.passed, f"max stress rel err: {cr.max_stress_rel_err:.2e}"
```

詳細は `fortran/README.md` を参照。

---

## API Reference

| シンボル | import パス |
|---------|------------|
| `MaterialModel3D`, `MaterialModelPS`, `MaterialModel1D` | `manforge.core.material` |
| `Implicit`, `Explicit`, `NTENS` | `manforge.core` |
| `SOLID_3D`, `PLANE_STRAIN`, `PLANE_STRESS`, `UNIAXIAL_1D` | `manforge.core.dimension` |
| `ReturnMappingResult`, `StressUpdateResult` | `manforge.core` |
| `PythonIntegrator`, `PythonNumericalIntegrator`, `PythonAnalyticalIntegrator`, `FortranIntegrator` | `manforge.simulation` |
| `StrainDriver`, `StressDriver` | `manforge.simulation` |
| `FieldHistory`, `FieldType` | `manforge.simulation.types` |
| `fit_params` | `manforge.fitting` |
| `check_tangent` | `manforge.verification.fd_check` |
| `ad_jacobian_blocks`, `JacobianBlocks` | `manforge.verification` |
| `CrosscheckStrainDriver`, `CrosscheckStressDriver` | `manforge.verification` |
| `compare_jacobians` | `manforge.verification` |
| `FortranModule` | `manforge.verification` |
| `generate_strain_history` | `manforge.verification` |
| `J2Isotropic3D`, `J2IsotropicPS`, `J2Isotropic1D` | `manforge.models.j2_isotropic` |
| `AFKinematic3D`, `OWKinematic3D` | `manforge.models` |

---

## Project Layout

```
src/manforge/
├── core/
│   ├── dimension.py       # StressDimension (SOLID_3D / PLANE_STRAIN / PLANE_STRESS / UNIAXIAL_1D)
│   ├── material/
│   │   ├── base.py        # MaterialModel (ABC)
│   │   └── bases.py       # MaterialModel3D / MaterialModelPS / MaterialModel1D
│   ├── result.py          # ReturnMappingResult / StressUpdateResult
│   └── state.py           # Implicit / Explicit / StateField / State
├── autodiff/
│   ├── backend.py         # autograd バックエンド
│   └── operators.py       # dev, vonmises, hydrostatic, norm_mandel 等
├── models/
│   ├── j2_isotropic.py    # J2 等方硬化 (参照実装)
│   ├── af_kinematic.py    # Armstrong-Frederick 移動硬化
│   └── ow_kinematic.py    # Ohno-Wang 修正 AF
├── simulation/
│   ├── types.py           # FieldType / FieldHistory / DriverResult
│   ├── driver.py          # StrainDriver / StressDriver
│   ├── _residual.py       # make_nr_residual / make_tangent_residual
│   └── integrator/
│       ├── base.py        # _PythonIntegratorBase (NR + consistent tangent)
│       ├── python.py      # PythonIntegrator / PythonNumericalIntegrator / PythonAnalyticalIntegrator
│       └── fortran.py     # FortranIntegrator
├── fitting/
│   ├── objective.py       # 残差二乗和
│   └── optimizer.py       # fit_params() + FitResult
├── verification/
│   ├── fd_check.py        # check_tangent()
│   ├── jacobian.py        # JacobianBlocks / ad_jacobian_blocks
│   ├── jacobian_compare.py# compare_jacobians
│   ├── crosscheck_driver.py # CrosscheckStrainDriver / CrosscheckStressDriver
│   ├── fortran_bridge.py  # FortranModule
│   └── test_cases.py      # generate_strain_history 等
└── utils/
    ├── voigt.py           # Voigt ↔ full tensor, Mandel 変換
    ├── tensor.py          # 4 階テンソル演算
    └── smooth.py          # smooth_sqrt / abs / norm / macaulay / direction
```

---

## Development

```bash
# テスト
make test              # 高速テスト: unit + integration (slow・fortran 除く)
make test-unit         # unit のみ
make test-slow         # FD tangent, フィッティング, 長い cyclic ループ
make test-fortran      # Fortran クロス検証 (コンパイル済み .so 必須)
make test-all          # 全スイート

# CLI
uv run manforge build fortran/abaqus_stubs.f90 fortran/j2_isotropic_3d.f90 --name j2_isotropic_3d
uv run manforge list
uv run manforge clean

# Docker (再現可能な gfortran 環境)
make docker-build && make docker-test
```

**Examples**

| スクリプト | 内容 |
|-----------|------|
| `examples/example_j2_uniaxial.py` | 単軸引張 + tangent 検証 + プロット |
| `examples/example_fit_j2.py` | J2 パラメータフィッティング |
| `examples/example_ow_convergence.py` | NR 残差履歴で二次収束検証 |
| `examples/example_verify_j2_analytical.py` | analytical return mapping vs AD jacobian |
| `examples/example_fortran_uniaxial.py` | Fortran UMAT を StrainDriver で実行 (要コンパイル) |
| `examples/crosscheck_umat_external.py` | Fortran UMAT crosscheck 最小例 |
| `examples/crosscheck_umat_advanced.py` | Fortran UMAT crosscheck 全機能ツアー |

---

## License

[MIT License](LICENSE)
