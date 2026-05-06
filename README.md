# manforge

Fortran UMAT (Abaqus) の検証・フィッティング用 Python フレームワーク。降伏関数と硬化則を実装するだけで、return mapping / consistent tangent (autograd 自動微分) / パラメータフィッティング / Fortran クロス検証がすべて使える。

---

## Features

- **autograd 自動微分による consistent tangent** — 手動でアルゴリズム接線を導出する必要なし
- **汎用 Newton-Raphson ソルバ** — Reduced (スカラー Δλ) / Augmented ((ntens+1+n_state) 系) の 2 経路を自動選択
- **多次元対応** — 3D ソリッド / 平面ひずみ / 平面応力 / 1D (NTENS=6/4/3/1) に `StressState` で統一対応
- **統一ドライバ** — `StrainDriver` / `StressDriver` で任意の荷重経路をシミュレート
- **組み込みモデル** — J2 等方硬化 / Armstrong-Frederick / Ohno-Wang 修正 AF モデルを同梱
- **検証ツール群** — FD tangent チェック / ソルバ間比較 / JacobianBlocks / Fortran UMAT クロス検証
- **パラメータフィッティング** — L-BFGS-B / Nelder-Mead / differential_evolution を scipy 経由

---

## Requirements

- Python >= 3.10
- `autograd >= 1.8.0`
- `numpy`
- `scipy`
- (optional) `gfortran` — Fortran クロス検証用
- (optional) `matplotlib` — examples のプロット用

---

## Installation

```bash
# pip
pip install manforge
pip install "manforge[dev]"       # pytest 含む
pip install "manforge[examples]"  # matplotlib 含む
pip install "manforge[fortran]"   # meson/ninja 含む (Fortran ビルド用)
```

開発環境 (ソースから):

```bash
uv sync --extra dev          # テスト依存含む
uv sync --all-extras         # 全 extras

# Fortran モジュールのビルド (gfortran 必須)
uv sync --extra fortran
uv run manforge build fortran/abaqus_stubs.f90 fortran/j2_isotropic_3d.f90 --name j2_isotropic_3d

# Docker 内でフルビルド + テスト
make docker-build && make docker-test
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
strain_history[:, 0] = np.linspace(0.0, 5e-3, 100)   # ε11 を増加
load = FieldHistory(FieldType.STRAIN, "Strain", strain_history)

result = StrainDriver(PythonIntegrator(model)).run(load)
print(f"Final σ11: {result.stress[-1, 0]:.1f} MPa")
```

詳しい使用例は `examples/` を参照。

---

## Architecture

```
manforge
├── Material layer  (core/material/)           定義: 降伏関数・硬化則・弾性テンソル
├── Solver layer    (simulation/integrator.py …) 実行: return mapping / consistent tangent
└── Application     (simulation/ fitting/ …)   使用: ドライバ / フィッティング / 検証
```

```
src/manforge/
├── core/
│   ├── stress_state.py    # StressState (SOLID_3D / PLANE_STRAIN / PLANE_STRESS / UNIAXIAL_1D)
│   ├── material/          # MaterialModel ABC + MaterialModel3D/PS/1D
│   │   ├── base.py        #   MaterialModel (ABC, フレームワーク基底)
│   │   └── stress_states.py #   MaterialModel3D / MaterialModelPS / MaterialModel1D
│   ├── stress_update.py   # ReturnMappingResult / StressUpdateResult — 戻り値 dataclass
│   ├── solver.py          # _numerical_newton (scalar / vector NR を自動選択)
│   ├── tangent.py         # consistent tangent (implicit differentiation)
│   ├── residual.py        # make_nr_residual / make_tangent_residual
│   └── jacobian.py        # JacobianBlocks — AD ヤコビアンブロック分解
├── autodiff/
│   ├── backend.py         # autograd バックエンド設定
│   └── operators.py       # dev, vonmises, hydrostatic, norm_mandel 等
├── models/
│   ├── j2_isotropic.py    # J2 等方硬化 (参照実装)
│   ├── af_kinematic.py    # Armstrong-Frederick 移動硬化
│   └── ow_kinematic.py    # Ohno-Wang 修正 AF (implicit_state_names の参照実装)
├── simulation/
│   ├── types.py           # FieldType / FieldHistory / DriverResult
│   ├── driver.py          # StrainDriver / StressDriver
│   └── integrator.py      # PythonIntegrator / PythonNumericalIntegrator / PythonAnalyticalIntegrator / FortranIntegrator — アルゴリズム本体を保持
├── fitting/
│   ├── objective.py       # 残差二乗和
│   └── optimizer.py       # fit_params() + FitResult
├── verification/
│   ├── comparator_base.py # Comparator ABC + 共有 rel-err ヘルパー
│   ├── jacobian_compare.py# compare_jacobians / JacobianComparisonResult
│   ├── crosscheck_driver.py # CrosscheckStrainDriver / CrosscheckStressDriver
│   ├── fortran_registry.py# @verified_against_fortran デコレータ
│   ├── fd_check.py        # check_tangent()
│   ├── fortran_bridge.py  # FortranModule — f2py ラッパー
│   └── test_cases.py      # estimate_yield_strain / generate_*
└── utils/
    ├── voigt.py           # Voigt ↔ full tensor, Mandel 変換
    ├── tensor.py          # 4 階テンソル操作
    └── smooth.py          # smooth_sqrt / abs / norm / macaulay / direction
```

---

## Core API

### Material Model

応力状態に合わせた基底クラスを継承し、必要なメソッドを実装する:

| 基底クラス | 対応 StressState | NTENS |
|-----------|----------------|-------|
| `MaterialModel3D` | `SOLID_3D`, `PLANE_STRAIN` | 6, 4 |
| `MaterialModelPS` | `PLANE_STRESS` | 3 |
| `MaterialModel1D` | `UNIAXIAL_1D` | 1 |

状態変数は `Implicit` / `Explicit` フィールドとして宣言し、σ も含めて対等に扱う:

| フィールド宣言 | 必須メソッド | NR 未知数 |
|--------------|-------------|----------|
| 全 state が `Explicit` (デフォルト) | `yield_function`, `update_state` | スカラー Δλ |
| 一部 state が `Implicit` (σ は Explicit) | `yield_function`, `update_state`, `state_residual` の両方 | `[Δλ, q_implicit]` |
| 全 state が `Implicit`、`stress = Implicit` | `yield_function`, `state_residual` | `[σ, Δλ, q]` |

`update_state` は **陽更新 state のみ** のキーを返す。`state_residual` は **陰 state のみ** のキーを返し、`self.stress(R_stress)` を含めると σ 残差をカスタム指定できる。さらに `self.dlambda(R_dl)` を含めると Δλ 行の残差を差し替えられる (粘塑性 Perzyna 型など)。省略時は `yield_function(state) = 0` がデフォルト。全メソッドは `state["stress"]` 経由で応力にアクセスする。

`elastic_stiffness(self, state=None)` は `MaterialModel3D` / `MaterialModelPS` / `MaterialModel1D` 基底クラスに等方性デフォルト実装が用意されており、`self.E` と `self.nu` を持つモデルなら実装不要。状態依存の弾性剛性 (損傷塑性など) を持つモデルのみ override する。

`__init_subclass__` がクラス定義時に `yield_function` の実装を検証するため、抜け漏れは `TypeError` として即座に報告される。

共通の演算子メソッド (`_vonmises`, `_dev`, `isotropic_C`, `_I_vol`, `_I_dev`) は基底クラスが応力状態に合わせた分岐なし実装を提供する。

### Stress Update

`integrator.stress_update` が完全な構成積分 (elastic trial → yield check → return mapping → consistent tangent) を実行する。`integrator.return_mapping` は tangent を含まない plastic correction のみの API。

```python
from manforge.simulation.integrator import PythonIntegrator

integrator = PythonIntegrator(model)

# 完全な構成積分 (tangent 含む)
result = integrator.stress_update(strain_inc, stress_n, state_n)
result.stress         # 収束応力 σ_{n+1}
result.ddsdde         # consistent tangent dσ/dΔε
result.stress_trial   # 試行応力 σ_trial = σ_n + C Δε  (FortranIntegrator では None)
result.dlambda        # 塑性乗数 Δλ (弾性ステップでは 0)
result.is_plastic     # True = 塑性ステップ
result.n_iterations   # NR 反復数
result.residual_history  # 各反復の残差ノルム

# 塑性補正のみ (tangent を計算しない)
rm = integrator.return_mapping(stress_trial, C, state_n)
rm.stress    # 収束応力
rm.state     # 収束状態
rm.dlambda   # Δλ
```

ソルバ経路は Integrator クラスで選択する (§Driver の表を参照)。`PythonIntegrator` が通常用途 (`user_defined` があれば使用、なければ NR)、`PythonNumericalIntegrator` が NR 固定、`PythonAnalyticalIntegrator` が閉形式解固定。

### Driver

`StrainDriver` と `StressDriver` はそれぞれひずみ制御・応力制御のシミュレーションを実行する。

```python
import numpy as np
from manforge.simulation import StrainDriver, StressDriver, PythonIntegrator
from manforge.simulation.types import FieldHistory, FieldType

# ひずみ制御
strain_history = np.zeros((100, 6))
strain_history[:, 0] = np.linspace(0, 5e-3, 100)
load = FieldHistory(FieldType.STRAIN, "Strain", strain_history)
result = StrainDriver(PythonIntegrator(model)).run(load)
result.stress    # np.ndarray (N, ntens) — 各ステップの σ_{n+1}
result.strain    # np.ndarray (N, ntens)

# 状態変数収集
result = StrainDriver(PythonIntegrator(model)).run(
    load,
    collect_state={"ep": FieldType.STRAIN}
)
result.fields["ep"].data   # np.ndarray (N,) — 等価塑性ひずみ履歴

# 応力制御 — 目標応力列から逆算
stress_target = np.zeros((100, 6))
stress_target[:, 0] = np.linspace(0, 300.0, 100)
stress_load = FieldHistory(FieldType.STRESS, "Stress", stress_target)
result = StressDriver(PythonIntegrator(model)).run(stress_load)
```

Driver は必ず `StressIntegrator` を受け取る。bare `MaterialModel` を渡すと `TypeError` になる。Integrator クラスは 3 種類:

| クラス | 内部ソルバ | 用途 |
|--------|----------|------|
| `PythonIntegrator(model)` | auto (user_defined あれば使用、なければ NR) | 通常用途 |
| `PythonNumericalIntegrator(model)` | numerical_newton 固定 | NR 強制 |
| `PythonAnalyticalIntegrator(model)` | user_defined 固定 | 閉形式解と比較する場合 |

`DriverResult.step_results` は `list[StressUpdateResult]` で、各ステップの詳細にアクセスできる:

```python
step = result.step_results[15]   # 15ステップ目の StressUpdateResult
print(step.dlambda, step.n_iterations)
```

`iter_run` を使うとステップごとに `DriverStep` を受け取るジェネレータになる。途中で分岐・打切り・対話的検査が可能:

```python
from manforge.simulation.driver import StrainDriver
from manforge.simulation.integrator import PythonIntegrator
from manforge.simulation.types import FieldHistory, FieldType
import numpy as np

load = FieldHistory(FieldType.STRAIN, "Strain", np.linspace(0, 5e-3, 100))
driver = StrainDriver(PythonIntegrator(model))

# 塑性開始ステップで break
for step in driver.iter_run(load):
    # step.i           — 0-based ステップ番号
    # step.strain      — 累積ひずみ (ntens,)
    # step.result      — StressUpdateResult (stress, state, ddsdde, dlambda, ...)
    if step.result.is_plastic:
        print(f"Plasticity onset at step {step.i}, ep={step.result.state['ep']:.4e}")
        break
```

`StressDriver` では `step.n_outer_iter` と `step.residual_inf` で外側 NR の収束状況を確認できる。非収束時の既定動作は `RuntimeError` だが、`raise_on_nonconverged=False` で `step.converged=False` の step を yield して consumer が判断することもできる。既存の `run` はそのまま使える（後方互換）。

### Fitting

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

## Adding a New Model

### Explicit-state template (J2-like)

状態変数を `Explicit` として宣言し、`update_state` に closed-form 更新を書く。NR はスカラー Δλ のみ解く。σ は自動装着されフレームワークが関連流れ則で更新する。

```python
import autograd.numpy as anp
from manforge.core import Implicit, Explicit, NTENS
from manforge.core.material import MaterialModel3D

class MyModel(MaterialModel3D):
    param_names = ["E", "nu", "sigma_y0", "H"]
    ep = Explicit(shape=(), doc="equivalent plastic strain")

    def __init__(self, *, E, nu, sigma_y0, H):
        super().__init__()   # デフォルト SOLID_3D
        self.E = E; self.nu = nu; self.sigma_y0 = sigma_y0; self.H = H

    def yield_function(self, state):
        return self._vonmises(state["stress"]) - (self.sigma_y0 + self.H * state["ep"])

    def update_state(self, dlambda, state_n, state_trial):
        return [self.ep(state_n["ep"] + dlambda)]
```

`elastic_stiffness` は `self.E` と `self.nu` から自動生成されるため省略可能。状態依存の弾性剛性を持つ場合は `elastic_stiffness(self, state=None)` を override する。

参照実装: `src/manforge/models/j2_isotropic.py`

#### 要素タイプ

```python
from manforge.core.stress_state import PLANE_STRAIN, PLANE_STRESS, UNIAXIAL_1D
from manforge.models.j2_isotropic import J2Isotropic3D, J2IsotropicPS, J2Isotropic1D

# 第1引数に StressState を渡すと要素タイプを切り替えられる
model_pe = J2Isotropic3D(PLANE_STRAIN, E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)
model_ps = J2IsotropicPS(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)
model_1d = J2Isotropic1D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)
```

利用可能なプリセット:

| インスタンス | NTENS | 想定要素 |
|------------|-------|---------|
| `SOLID_3D` | 6 | C3D8 等 3D ソリッド |
| `PLANE_STRAIN` | 4 | CPE4 / CAX4 |
| `PLANE_STRESS` | 3 | CPS4 / シェル |
| `UNIAXIAL_1D` | 1 | 1D トラス / フィッティング用 |

### Implicit-state template (Ohno-Wang-like)

`update_state` が closed-form で解けない場合は状態変数を `Implicit` として宣言し、`state_residual` に後方 Euler 残差を書く。`stress = Implicit(shape=NTENS)` を宣言すると σ も NR 未知数になる。

```python
import autograd.numpy as anp
from manforge.core import Implicit, Explicit, NTENS
from manforge.core.material import MaterialModel3D

class MyImplicitModel(MaterialModel3D):
    param_names = ["E", "nu", "sigma_y0", "C_k", "gamma"]
    stress = Implicit(shape=NTENS, doc="Cauchy stress (NR unknown)")
    alpha  = Implicit(shape=NTENS, doc="backstress tensor")
    ep     = Implicit(shape=(),    doc="equivalent plastic strain")

    def __init__(self, *, E, nu, sigma_y0, C_k, gamma):
        super().__init__()
        self.E = E; self.nu = nu; self.sigma_y0 = sigma_y0
        self.C_k = C_k; self.gamma = gamma

    def yield_function(self, state):
        xi = state["stress"] - state["alpha"]
        return self._vonmises(xi) - self.sigma_y0

    def state_residual(self, state_new, dlambda, state_n, state_trial):
        """後方 Euler 残差。implicit state のキーのみ返す。収束時にゼロ。"""
        # ...モデル固有の残差を定義...
        R_alpha = ...         # shape (ntens,)
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        # stress を省略すると framework が associative R_stress を自動装着する
        return [self.alpha(R_alpha), self.ep(R_ep)]
```

参照実装: `src/manforge/models/ow_kinematic.py`

#### Δλ 行残差のカスタム指定 (粘塑性・非標準 consistency condition)

`state_residual` に `self.dlambda(R_dl)` を含めると NR 残差の Δλ 行を差し替えられる。省略時は `yield_function(state) = 0` (標準弾塑性)。

```python
def state_residual(self, state_new, dlambda, state_n, state_trial):
    # Perzyna 型粘塑性: f − η·Δλ/Δt = 0
    f = self.yield_function(state_new)
    R_dl = f - self.eta * dlambda / self.dt
    R_alpha = ...
    return [self.dlambda(R_dl), self.alpha(R_alpha)]
```

`self.dlambda(R_dl)` は scalar-NR / vector NR / consistent tangent の全経路で機能する。`update_state` で使うと `TypeError`、リスト内に 2 個含めると `ValueError`。

### Kinematic / backstress (AF・OW の既製クラス)

backstress 型 (Armstrong-Frederick / Ohno-Wang) を使う場合は組み込みクラスが利用できる:

```python
import numpy as np
from manforge.models import AFKinematic3D, OWKinematic3D
from manforge.simulation import StrainDriver, PythonIntegrator
from manforge.simulation.types import FieldHistory, FieldType

# Armstrong-Frederick (全 state 陽更新、スカラー NR)
model_af = AFKinematic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, C_k=8_000.0, gamma=80.0)

# Ohno-Wang 修正 AF (全 state 陰、stress=Implicit、ベクトル NR)
model_ow = OWKinematic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, C_k=8_000.0, gamma=0.05)

strain_history = np.zeros((150, 6))
strain_history[:50, 0]  = np.linspace(0, 1e-2, 50)
strain_history[50:, 0]  = np.linspace(1e-2, -1e-2, 100)
load = FieldHistory(FieldType.STRAIN, "Strain", strain_history)
result = StrainDriver(PythonIntegrator(model_af)).run(load)
```

飽和 backstress の比較:

| モデル | 飽和 ‖α‖ | γ の単位 |
|--------|---------|---------|
| `AFKinematic` | `C_k / γ` | 無次元 |
| `OWKinematic` | `√(C_k / γ)` | 1/MPa |

同じ飽和振幅を得るには `γ_OW = C_k / ‖α_sat‖²`、`γ_AF = C_k / ‖α_sat‖` とする。

### Advanced — user_defined フック

閉形式の return mapping が存在する場合、`user_defined_return_mapping` をオーバーライドすると `method="auto"` (デフォルト) で自動的に使用される。`None` を返すと汎用 NR にフォールバックする。

```python
from manforge.core.stress_update import ReturnMappingResult

class MyModel(MaterialModel3D):
    # ... 必須メソッド ...

    def user_defined_return_mapping(self, stress_trial, C, state_n):
        """None を返すと汎用 NR にフォールバック。"""
        # ...閉形式で stress_new, state_new, dlambda を計算...
        return ReturnMappingResult(stress=stress_new, state=state_new, dlambda=dlambda)

    def user_defined_tangent(self, stress, state, dlambda, C, state_n):
        """None を返すと autodiff にフォールバック。"""
        # ...閉形式で ddsdde を計算...
        return ddsdde
```

`J2Isotropic3D` は両フックを実装済み (J2 radial return 閉形式 + 既知の `D^ep` 式)。

---

## Verification

新しいモデルを実装するときの推奨ワークフロー:

1. **残差の立式** — 応力更新式と硬化則残差を定義
2. **数値解で動作確認** — `method="auto"` (汎用 NR) で収束を確認
3. **AD 値との照合** — `JacobianBlocks` でヤコビアンの各ブロックを要素ごとに検証
4. **FD tangent チェック** — `check_tangent` で consistent tangent の精度を確認
5. **Fortran への移植** — 検証済みブロック構造をそのままサブルーチン構成に対応させる

### JacobianBlocks で AD 値との照合

`ad_jacobian_blocks` は収束点の残差ヤコビアンを autograd で計算し、物理量・状態変数名をキーとするブロックに分解する:

```python
import numpy.testing as npt
from manforge.simulation.integrator import PythonIntegrator
from manforge.core.jacobian import ad_jacobian_blocks

result = PythonIntegrator(model).stress_update(deps, stress_n, state_n)
jac = ad_jacobian_blocks(model, result, state_n)

# フロー方向 ∂f/∂σ を自分の解析式と照合
my_n = (3/2) * dev_stress / sigma_vm
npt.assert_allclose(jac.dyield_dsigma, my_n, rtol=1e-8)

npt.assert_allclose(float(jac.dyield_ddlambda), -H, rtol=1e-8)
print(jac.dstress_dsigma)   # shape (ntens, ntens)
print(jac.full)             # フラットな全体行列
```

implicit_state_names を持つモデル (例: Ohno-Wang) では状態変数名をキーとするブロックも取得できる:

```python
jac.dstate_dsigma["alpha"]        # ∂R_alpha/∂σ,  shape (ntens, ntens)
jac.dstate_ddlambda["ep"]         # ∂R_ep/∂Δλ,    shape (1,)
jac.dstate_dstate["alpha"]["ep"]  # ∂R_alpha/∂ep, shape (ntens, 1)
```

### check_tangent で FD 照合

```python
from manforge.verification.fd_check import check_tangent
from manforge.simulation.integrator import PythonIntegrator, PythonAnalyticalIntegrator

# method="auto" 相当 (user_defined があれば使用、なければ NR)
result = check_tangent(PythonIntegrator(model), stress_n, state_n, strain_inc)
# 解析接線を FD と比較する場合
result = check_tangent(PythonAnalyticalIntegrator(model), stress_n, state_n, strain_inc)
print(result.passed, result.max_rel_err)
```

### CrosscheckStrainDriver でソルバ同士の比較

`CrosscheckStrainDriver` は 2 つの `StressIntegrator` を同じロード履歴で走らせて各ステップを比較する。単一ステップ比較には 1 行の `FieldHistory` と `initial_stress`/`initial_state` kwarg を使う:

```python
import numpy as np
from manforge.simulation import PythonNumericalIntegrator, PythonAnalyticalIntegrator
from manforge.simulation.types import FieldHistory, FieldType
from manforge.verification import CrosscheckStrainDriver, generate_strain_history

# Multi-step comparison
history = generate_strain_history(model)
load = FieldHistory(FieldType.STRAIN, "eps", history)
cc = CrosscheckStrainDriver(
    PythonNumericalIntegrator(model),
    PythonAnalyticalIntegrator(model),
)
result = cc.run(load)
print(f"Passed: {result.passed}  max_stress_err={result.max_stress_rel_err:.2e}")
```

`iter_run` を使うとステップごとに `CrosscheckCaseResult` を受け取るジェネレータになる。失敗ステップで打ち切り、`compare_jacobians` で Jacobian ブロックレベルの詳細デバッグが可能:

```python
from manforge.verification import compare_jacobians

for cr in cc.iter_run(load):
    if not cr.passed:
        print(f"Step {cr.index} failed: stress_err={cr.stress_rel_err:.2e}")
        jac = compare_jacobians(model, cr.result_a, cr.result_b, cr.state_n)
        print(jac.blocks)
        break
```

### Fortran UMAT クロス検証

**Fortran ワークフロー**:

1. **Build** — `.f90` を f2py でコンパイルして importable モジュールを生成
2. **Load** — `FortranModule("module_name")` でロード (`uv run manforge list` で確認)
3. **Integrate** — `FortranIntegrator.from_model(fortran, subroutine, model)` でドライバ互換に
4. **Crosscheck** — `CrosscheckStrainDriver(py_int, fc_int)` で Python 実装と step 毎に照合

```python
import numpy as np
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.simulation import FortranIntegrator, StrainDriver
from manforge.simulation.integrator import PythonIntegrator
from manforge.simulation.types import FieldHistory, FieldType
from manforge.verification import FortranModule, CrosscheckStrainDriver

model   = J2Isotropic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)
fortran = FortranModule("j2_isotropic_3d")  # importable module name, not a file path

# --- Production: run the UMAT via StrainDriver ---
fc_int = FortranIntegrator.from_model(fortran, "j2_isotropic_3d", model)
strain_data = np.zeros((50, model.ntens))
strain_data[:, 0] = np.linspace(0.0, 5e-3, 50)
load = FieldHistory(FieldType.STRAIN, "eps", strain_data)
result = StrainDriver(fc_int).run(load)

# --- Crosscheck: validate against Python implementation ---
py_int = PythonIntegrator(model)
cc = CrosscheckStrainDriver(py_int, fc_int)
cr = cc.run(load)
assert cr.passed, f"max stress rel err: {cr.max_stress_rel_err:.2e}"
```

コンポーネント単位での比較 (`FortranModule.call` を直接使う場合):

```python
from manforge.verification import FortranModule

fortran = FortranModule("j2_isotropic_3d")
dstran  = np.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
stress_f, ep_f, ddsdde_f = fortran.call(
    "j2_isotropic_3d",
    model.E, model.nu, model.sigma_y0, model.H,
    np.zeros(6), 0.0, dstran,
)
C_f = fortran.call("j2_isotropic_3d_elastic_stiffness", model.E, model.nu)
```

ビルド:

```bash
uv sync --extra fortran
uv run manforge build fortran/abaqus_stubs.f90 fortran/j2_isotropic_3d.f90 --name j2_isotropic_3d
uv run manforge list   # 確認
# または: make fortran-build-umat  /  make docker-test
```

詳細: `fortran/README.md`

---

## CLI

```bash
# Fortran ソースをビルドして Python 拡張モジュールを生成
uv run manforge build fortran/abaqus_stubs.f90 fortran/j2_isotropic_3d.f90 --name j2_isotropic_3d

# コンパイル済みモジュール一覧
uv run manforge list

# コンパイル済みアーティファクトを削除
uv run manforge clean
uv run manforge clean --dry-run   # プレビュー (削除しない)
```

---

## Utilities

### smooth.py — 微分可能なスムーズ近似

ゼロ点・キンク点を含む全定義域で C∞ 微分可能な近似関数 (`src/manforge/utils/smooth.py`):

| 関数 | 近似対象 | 特記事項 |
|------|---------|---------|
| `smooth_sqrt(x, eps)` | `√x` | 基底プリミティブ。`√(x + ε²)` |
| `smooth_abs(x, eps)` | `\|x\|` | `smooth_sqrt(x²)` |
| `smooth_norm(v, eps)` | `‖v‖` | `smooth_sqrt(v·v)` |
| `smooth_macaulay(x, eps)` | `max(x, 0)` | ランプ関数の滑らか版 |
| `smooth_direction(v, eps)` | `v / ‖v‖` | ゼロベクトルで NaN にならない |

デフォルト `eps=1e-30`。`_vonmises` は内部で `smooth_sqrt` を使用しているため、呼び出し側で重ねてラップする必要はない。

```python
from manforge.utils.smooth import smooth_direction, smooth_norm

n_hat = smooth_direction(deviatoric_stress)   # ゼロ応力でも安全
```

### その他のユーティリティ

- `utils/voigt.py` — Voigt ↔ full tensor 変換、Mandel スケーリング
- `utils/tensor.py` — 4 階テンソル演算 (`ddot44`, `ddot42`, `symmetrize4`)
- `autodiff/operators.py` — StressState 対応の `dev`, `vonmises`, `hydrostatic`, `norm_mandel`

---

## Testing

```bash
make test              # 高速テスト: unit + integration (slow・fortran 除く)
make test-unit         # unit のみ (最速)
make test-integration  # integration (slow 除く)
make test-slow         # FD tangent, フィッティング, 長い cyclic ループ
make test-fortran      # Fortran クロス検証 (コンパイル済み .so 必須)
make test-all          # 全スイート
```

---

## Examples

| スクリプト | 内容 |
|-----------|------|
| `examples/example_j2_uniaxial.py` | 単軸引張シミュレーション + tangent 検証 + stress-strain プロット |
| `examples/example_fit_j2.py` | 合成データへの J2 パラメータフィッティング + 収束履歴プロット |
| `examples/example_ow_convergence.py` | J2 / OW kinematic の NR 残差履歴で二次収束検証 |
| `examples/example_verify_j2_analytical.py` | analytical return mapping vs AD jacobian ブロック照合 |
| `examples/example_fortran_uniaxial.py` | **Fortran UMAT** を StrainDriver で回す最小 production 例 (要コンパイル) |
| `examples/crosscheck_umat_external.py` | **Fortran UMAT** crosscheck 最小例 — 外部ユーザ視点 |
| `examples/crosscheck_umat_advanced.py` | **Fortran UMAT** crosscheck 全機能ツアー (strain/stress driver, iter_run, jacobian) |

```bash
uv run python examples/example_j2_uniaxial.py
uv run python examples/example_fit_j2.py
uv run python examples/example_ow_convergence.py
uv run python examples/example_verify_j2_analytical.py

# Fortran examples (require: uv run manforge build ...)
uv run python examples/example_fortran_uniaxial.py
uv run python examples/crosscheck_umat_external.py
uv run python examples/crosscheck_umat_advanced.py
```

---

## Technical Notes

- **Voigt 記法**: 3D ソリッドは `[σ11, σ22, σ33, σ12, σ13, σ23]` (6成分)。次元は `StressState` で制御。剪断成分には内部演算で Mandel スケーリング (×√2) を適用
- **state 型**: `dict[str, anp.ndarray]` で統一
- **精度**: numpy のデフォルト float64 をそのまま使用 (追加設定不要)
- **バージョン確認**: `import manforge; print(manforge.__version__)`
- **jit 互換**: 現バージョンは正確性優先。jit 対応は将来課題

---

## License

[MIT License](LICENSE)
