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
from manforge.simulation.driver import StrainDriver

model = J2Isotropic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)

strain_history = np.zeros((100, 6))
strain_history[:, 0] = np.linspace(0.0, 5e-3, 100)   # ε11 を増加

result = StrainDriver().run(model, strain_history)
print(f"Final σ11: {result.stress[-1, 0]:.1f} MPa")
```

詳しい使用例は `examples/` を参照。

---

## Architecture

```
manforge
├── Material layer  (core/material.py)         定義: 降伏関数・硬化則・弾性テンソル
├── Solver layer    (core/stress_update.py …)  実行: return mapping / consistent tangent
└── Application     (simulation/ fitting/ …)   使用: ドライバ / フィッティング / 検証
```

```
src/manforge/
├── core/
│   ├── stress_state.py    # StressState (SOLID_3D / PLANE_STRAIN / PLANE_STRESS / UNIAXIAL_1D)
│   ├── material.py        # MaterialModel ABC + MaterialModel3D/PS/1D
│   ├── stress_update.py   # stress_update / return_mapping — NR ソルバ統合
│   ├── solver.py          # _reduced_nr / _augmented_nr
│   ├── tangent.py         # consistent tangent (implicit differentiation)
│   ├── residual.py        # make_reduced_residual / make_augmented_residual
│   └── jacobian.py        # JacobianBlocks — AD ヤコビアンブロック分解
├── autodiff/
│   ├── backend.py         # autograd バックエンド設定
│   └── operators.py       # dev, vonmises, hydrostatic, norm_mandel 等
├── models/
│   ├── j2_isotropic.py    # J2 等方硬化 (参照実装)
│   ├── af_kinematic.py    # Armstrong-Frederick 移動硬化
│   └── ow_kinematic.py    # Ohno-Wang 修正 AF (augmented path の参照実装)
├── simulation/
│   ├── types.py           # FieldType / FieldHistory / DriverResult
│   └── driver.py          # StrainDriver / StressDriver
├── fitting/
│   ├── objective.py       # 残差二乗和
│   └── optimizer.py       # fit_params() + FitResult
├── verification/
│   ├── compare.py         # compare_solvers()
│   ├── fd_check.py        # check_tangent()
│   ├── fortran_bridge.py  # FortranUMAT — f2py ラッパー
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

必須メソッドは `hardening_type` によって異なる:

| `hardening_type` | 必須メソッド |
|-----------------|-------------|
| `"reduced"` (デフォルト) | `elastic_stiffness`, `yield_function`, `update_state` |
| `"augmented"` | `elastic_stiffness`, `yield_function`, `state_residual` |

`__init_subclass__` がクラス定義時にこれを検証するため、抜け漏れは `TypeError` として即座に報告される。

共通の演算子メソッド (`_vonmises`, `_dev`, `isotropic_C`, `_I_vol`, `_I_dev`) は基底クラスが応力状態に合わせた分岐なし実装を提供する。

### Stress Update

`stress_update` が完全な構成積分 (elastic trial → yield check → return mapping → consistent tangent) を実行する。`return_mapping` は tangent を含まない plastic correction のみの API。

```python
from manforge.core.stress_update import stress_update, return_mapping

# 完全な構成積分 (tangent 含む)
result = stress_update(model, strain_inc, stress_n, state_n)
result.stress         # 収束応力 σ_{n+1}
result.ddsdde         # consistent tangent dσ/dΔε
result.stress_trial   # 試行応力 σ_trial = σ_n + C Δε
result.dlambda        # 塑性乗数 Δλ (弾性ステップでは 0)
result.is_plastic     # True = 塑性ステップ
result.n_iterations   # NR 反復数
result.residual_history  # 各反復の残差ノルム

# 塑性補正のみ (tangent を計算しない)
rm = return_mapping(model, stress_trial, C, state_n)
rm.stress    # 収束応力
rm.state     # 収束状態
rm.dlambda   # Δλ
```

`method` 引数でソルバ経路を選択できる:

| `method` | return mapping | consistent tangent |
|----------|---------------|-------------------|
| `"auto"` (デフォルト) | `user_defined_return_mapping` があれば使用、なければ NR | `user_defined_tangent` があれば使用、なければ autodiff |
| `"numerical_newton"` | 常に NR+autodiff | 常に autodiff |
| `"user_defined"` | `user_defined_return_mapping` が必須 (なければ `NotImplementedError`) | `user_defined_tangent` が必須 |

### Driver

`StrainDriver` と `StressDriver` はそれぞれひずみ制御・応力制御のシミュレーションを実行する。

```python
import numpy as np
from manforge.simulation.driver import StrainDriver, StressDriver
from manforge.simulation.types import FieldHistory, FieldType

# ひずみ制御 — N×ntens の配列を直接渡す
strain_history = np.zeros((100, 6))
strain_history[:, 0] = np.linspace(0, 5e-3, 100)
result = StrainDriver().run(model, strain_history)
result.stress    # np.ndarray (N, ntens) — 各ステップの σ_{n+1}
result.strain    # np.ndarray (N, ntens)

# FieldHistory を使う場合 (状態変数収集時に必要)
load = FieldHistory(FieldType.STRAIN, "Strain", strain_history)
result = StrainDriver().run(
    model, load,
    collect_state={"ep": FieldType.STRAIN}
)
result.fields["ep"].data   # np.ndarray (N,) — 等価塑性ひずみ履歴

# 応力制御 — 目標応力列から逆算
stress_target = np.zeros((100, 6))
stress_target[:, 0] = np.linspace(0, 300.0, 100)
result = StressDriver().run(model, stress_target)
```

`UniaxialDriver` / `GeneralDriver` は `StrainDriver` への後方互換エイリアス。

`DriverResult.step_results` は `list[StressUpdateResult]` で、各ステップの詳細にアクセスできる:

```python
step = result.step_results[15]   # 15ステップ目の StressUpdateResult
print(step.dlambda, step.n_iterations)
```

### Fitting

```python
from manforge.simulation.driver import StrainDriver
from manforge.fitting import fit_params

fit_config = {
    "sigma_y0": (200.0, (100.0, 500.0)),  # (初期値, (下限, 上限))
    "H":        (500.0, (0.0,  5000.0)),
}
result = fit_params(
    model, StrainDriver(), exp_data,
    fit_config=fit_config,
    fixed_params={"E": 210_000.0, "nu": 0.3},
    method="L-BFGS-B",
)
print(result.params, result.residual)
```

`exp_data` は `{"strain": ..., "stress": ...}` の dict。詳細: `examples/example_fit_j2.py`

---

## Adding a New Model

### Reduced template (J2-like)

`hardening_type = "reduced"` (デフォルト) の場合、状態更新を closed-form で `update_state` に記述する。NR はスカラー Δλ のみ解く。

```python
import autograd.numpy as anp
from manforge.core.material import MaterialModel3D

class MyModel(MaterialModel3D):
    param_names = ["E", "nu", "sigma_y0", "H"]
    state_names = ["ep"]

    def __init__(self, *, E, nu, sigma_y0, H):
        super().__init__()   # デフォルト SOLID_3D
        self.E = E; self.nu = nu; self.sigma_y0 = sigma_y0; self.H = H

    def elastic_stiffness(self):
        mu = self.E / (2.0 * (1.0 + self.nu))
        lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        return self.isotropic_C(lam, mu)

    def yield_function(self, stress, state):
        return self._vonmises(stress) - (self.sigma_y0 + self.H * state["ep"])

    def update_state(self, dlambda, stress, state):
        return {"ep": state["ep"] + dlambda}
```

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

### Augmented template (Ohno-Wang-like)

`update_state` が closed-form で解けない場合 (後方 Euler 離散化に状態変数が非線形に現れる場合)、`hardening_type = "augmented"` を宣言して `state_residual` を実装する。NR は (ntens+1+n_state) の拡張残差系を解く。

```python
import autograd.numpy as anp
from manforge.core.material import MaterialModel3D

class MyImplicitModel(MaterialModel3D):
    hardening_type = "augmented"
    param_names = ["E", "nu", "sigma_y0", "C_k", "gamma"]
    state_names = ["alpha", "ep"]

    def __init__(self, *, E, nu, sigma_y0, C_k, gamma):
        super().__init__()
        self.E = E; self.nu = nu; self.sigma_y0 = sigma_y0
        self.C_k = C_k; self.gamma = gamma

    def initial_state(self):
        return {"alpha": anp.zeros(self.ntens), "ep": anp.array(0.0)}

    def elastic_stiffness(self):
        mu = self.E / (2.0 * (1.0 + self.nu))
        lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        return self.isotropic_C(lam, mu)

    def yield_function(self, stress, state):
        xi = stress - state["alpha"]
        return self._vonmises(xi) - self.sigma_y0

    def state_residual(self, state_new, dlambda, stress, state_n):
        """後方 Euler 残差。収束時にゼロになる dict を返す。"""
        # ...モデル固有の残差を定義...
        R_alpha = ...         # shape (ntens,)
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        return {"alpha": R_alpha, "ep": R_ep}
```

参照実装: `src/manforge/models/ow_kinematic.py`

### Kinematic / backstress (AF・OW の既製クラス)

backstress 型 (Armstrong-Frederick / Ohno-Wang) を使う場合は組み込みクラスが利用できる:

```python
import numpy as np
from manforge.models import AFKinematic3D, OWKinematic3D
from manforge.simulation.driver import StrainDriver

# Armstrong-Frederick (reduced path)
model_af = AFKinematic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, C_k=8_000.0, gamma=80.0)

# Ohno-Wang 修正 AF (augmented path)
model_ow = OWKinematic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, C_k=8_000.0, gamma=0.05)

strain_history = np.zeros((150, 6))
strain_history[:50, 0]  = np.linspace(0, 1e-2, 50)
strain_history[50:, 0]  = np.linspace(1e-2, -1e-2, 100)
result = StrainDriver().run(model_af, strain_history)
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
from manforge.core.stress_update import stress_update
from manforge.core.jacobian import ad_jacobian_blocks

result = stress_update(model, deps, stress_n, state_n)
jac = ad_jacobian_blocks(model, result, state_n)

# フロー方向 ∂f/∂σ を自分の解析式と照合
my_n = (3/2) * dev_stress / sigma_vm
npt.assert_allclose(jac.dyield_dsigma, my_n, rtol=1e-8)

npt.assert_allclose(float(jac.dyield_ddlambda), -H, rtol=1e-8)
print(jac.dstress_dsigma)   # shape (ntens, ntens)
print(jac.full)             # フラットな全体行列
```

augmented モデル (例: Ohno-Wang) では状態変数名をキーとするブロックも取得できる:

```python
jac.dstate_dsigma["alpha"]        # ∂R_alpha/∂σ,  shape (ntens, ntens)
jac.dstate_ddlambda["ep"]         # ∂R_ep/∂Δλ,    shape (1,)
jac.dstate_dstate["alpha"]["ep"]  # ∂R_alpha/∂ep, shape (ntens, 1)
```

### check_tangent で FD 照合

```python
from manforge.verification.fd_check import check_tangent

result = check_tangent(
    model, stress_n, state_n,
    strain_inc,
    method="auto",   # "numerical_newton" / "user_defined" も指定可
)
print(result.passed, result.max_rel_err)
```

### compare_solvers でソルバ同士の比較

```python
import numpy as np
from manforge.core.stress_update import stress_update
from manforge.verification.compare import compare_solvers
from manforge.verification.test_cases import generate_single_step_cases

def make_solver(method):
    def _solve(strain_inc, stress_n, state_n):
        r = stress_update(model, strain_inc, stress_n, state_n, method=method)
        return r.stress, r.state, r.ddsdde
    return _solve

# 弾性・塑性・多軸・剪断ケースを自動生成
cases = generate_single_step_cases(model)
result = compare_solvers(
    solver_a=make_solver("numerical_newton"),
    solver_b=make_solver("user_defined"),
    test_cases=cases,
)
print(f"Passed: {result.passed}  max_stress_err={result.max_stress_rel_err:.2e}")
```

`generate_single_step_cases` は `estimate_yield_strain` で降伏ひずみを自動推定し、5 ケース (弾性・塑性・多軸・剪断・プレストレス) を生成する。

### Fortran UMAT クロス検証

```python
import numpy as np
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.core.stress_update import stress_update
from manforge.verification import FortranUMAT

model   = J2Isotropic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)
dstran  = np.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
fortran = FortranUMAT("j2_isotropic_3d")

stress_f, ep_f, ddsdde_f = fortran.call(
    "j2_isotropic_3d",
    model.E, model.nu, model.sigma_y0, model.H,
    np.zeros(6), 0.0, dstran,
)
result_py = stress_update(model, dstran, np.zeros(6), model.initial_state())
np.testing.assert_allclose(np.array(result_py.stress), stress_f, rtol=1e-6)
```

コンポーネント単位での比較も同じパターン:

```python
C_f  = fortran.call("j2_isotropic_3d_elastic_stiffness", model.E, model.nu)
C_py = model.elastic_stiffness()
np.testing.assert_allclose(np.array(C_py), np.array(C_f), rtol=1e-12)
```

ビルド:

```bash
uv sync --extra fortran
uv run manforge build fortran/abaqus_stubs.f90 fortran/j2_isotropic_3d.f90 --name j2_isotropic_3d
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

```bash
uv run python examples/example_j2_uniaxial.py
uv run python examples/example_fit_j2.py
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
