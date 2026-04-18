# manforge

Fortran UMAT (Abaqus) の検証・フィッティング用 Python フレームワーク。

降伏関数・硬化則・弾性テンソルを定義するだけで、return mapping / consistent tangent (自動微分) / パラメータフィッティング / Fortran クロス検証がすべて動く設計。

---

## Features

- **JAX 自動微分による consistent tangent** — 手動でアルゴリズム接線を導出する必要なし
- **汎用 return mapping ソルバ** — Closest point projection + Newton-Raphson (最大 50 反復、tol=1e-10)
- **多次元対応 (StressState)** — 3D ソリッド (NTENS=6) / 平面ひずみ・軸対称 (NTENS=4) / 平面応力 (NTENS=3) / 1D トラス (NTENS=1) に対応。既存コードは後方互換
- **統一ドライバ I/O** — `FieldHistory` / `DriverResult` による型付きデータモデル。ひずみ制御 (`StrainDriver`) と応力制御 (`StressDriver`) を統一 API で提供
- **解析解 vs 自動微分の Python 同士比較** — 閉形式解が存在するモデルでは `method="analytical"` / `"autodiff"` を切り替えてクロス検証
- **パラメータフィッティング** — L-BFGS-B / Nelder-Mead / differential_evolution を scipy 経由で実行
- **Fortran UMAT クロス検証** — f2py 経由で Python 実装と Fortran UMAT の出力を自動比較
- **汎用ソルバ比較** — `compare_solvers()` で任意の 2 つのソルバ出力 (stress, DDSDDE) を自動照合
- **有限差分 tangent チェック** — AD / 解析解のどちらも中心差分で自動照合
- **応力状態基底クラス** — `MaterialModel3D` / `MaterialModelPS` / `MaterialModel1D` を継承し、`elastic_stiffness` / `yield_function` と硬化則メソッド 1 つを実装するだけで新規モデルに対応。演算子 (`_vonmises`, `_dev`, `isotropic_C` 等) は基底が提供
- **移動硬化モデル (Armstrong-Frederick)** — `AFKinematic3D` / `AFKinematicPS` / `AFKinematic1D` として実装済み。`manforge.models` から直接インポート可能
- **暗黙的硬化則サポート (augmented residual)** — `hardening_type = "implicit"` を宣言して `hardening_residual` を実装すると (ntens+1+n_state) 拡張残差系が自動で有効になり、closed-form では解けない硬化則を持つモデルも記述可能。`__init_subclass__` がクラス定義時に必須メソッドの有無を検証する
- **Ohno-Wang 修正 AF モデル** — `OWKinematic3D` / `OWKinematicPS` / `OWKinematic1D` として実装済み。後方 Euler 離散化が genuinely implicit な実例
- **JAX 安全な微分ユーティリティ** — `utils/smooth.py` が `smooth_sqrt` / `smooth_abs` / `smooth_norm` / `smooth_macaulay` / `smooth_direction` を提供。ゼロ点を含め C∞ 全微分可能で、高次微分が必要なモデルで利用

---

## Requirements

- Python >= 3.10
- `jax[cpu]` >= 0.4.20
- `numpy`
- `scipy`
- (optional) `gfortran` — Fortran クロス検証用
- (optional) `matplotlib` — examples のプロット用

---

## Installation

```bash
# pip でインストール (PyPI / wheel)
pip install manforge

# Fortran ビルドツール (meson/ninja) 含む
pip install manforge[fortran]

# テスト依存含む
pip install manforge[dev]

# matplotlib 含む
pip install manforge[examples]
```

開発環境 (ソースから):

```bash
# 基本インストール
uv sync

# テスト依存含む
uv sync --extra dev

# matplotlib 含む
uv sync --extra examples

# Fortran モジュールのビルド (gfortran 必須)
uv sync --extra fortran           # meson/ninja のインストール
uv run manforge build fortran/abaqus_stubs.f90 fortran/j2_isotropic_3d.f90 --name j2_isotropic_3d
# pip install 環境では以下でも同等
# python -m manforge build fortran/abaqus_stubs.f90 fortran/j2_isotropic_3d.f90 --name j2_isotropic_3d

# Docker 内でフルビルド + テスト
make docker-build && make docker-test
```

---

## Quick Start

```python
import manforge  # JAX float64 モードを有効化
import numpy as np
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.simulation.driver import StrainDriver
from manforge.simulation.types import FieldHistory, FieldType

model = J2Isotropic3D(
    E=210_000.0,    # Young's modulus [MPa]
    nu=0.3,         # Poisson's ratio
    sigma_y0=250.0, # Initial yield stress [MPa]
    H=1_000.0,      # Linear isotropic hardening modulus [MPa]
)
driver = StrainDriver()

strain_history = np.linspace(0.0, 5e-3, 100)       # cumulative ε11
load = FieldHistory(FieldType.STRAIN, "Strain", strain_history)
result = driver.run(model, load)

print(f"Final stress: {result.stress[-1, 0]:.1f} MPa")  # σ11
```

より詳しい使用例は `examples/` ディレクトリを参照。

---

## Architecture

```
src/manforge/
├── core/
│   ├── stress_state.py    # StressState — 次元記述子 (SOLID_3D / PLANE_STRAIN / PLANE_STRESS / UNIAXIAL_1D)
│   ├── material.py        # MaterialModel ABC + MaterialModel3D/PS/1D (応力状態基底)
│   ├── return_mapping.py  # Closest point projection + Newton-Raphson (scalar and augmented)
│   ├── tangent.py         # Consistent tangent (暗黙微分 / augmented)
│   └── residual.py        # make_augmented_residual() — 拡張残差 factory (implicit hardening 用)
├── autodiff/
│   ├── backend.py         # JAX バックエンド設定
│   └── operators.py       # dev, vonmises, hydrostatic, norm_mandel 等 (StressState 対応)
├── models/
│   ├── j2_isotropic.py    # J2 等方硬化 (参照実装)
│   ├── af_kinematic.py    # J2 + Armstrong-Frederick 移動硬化 (AFKinematic3D/PS/1D)
│   └── ow_kinematic.py    # J2 + Ohno-Wang 修正 AF 移動硬化 (OWKinematic3D/PS/1D)
├── simulation/
│   ├── types.py           # FieldType enum / FieldHistory / DriverResult — ドライバ I/O データモデル
│   └── driver.py          # DriverBase ABC / StrainDriver / StressDriver (UniaxialDriver・GeneralDriver は後方互換エイリアス)
├── fitting/
│   ├── objective.py       # 残差二乗和
│   └── optimizer.py       # fit_params() + FitResult
├── verification/
│   ├── compare.py         # compare_solvers() — 汎用ソルバ比較
│   ├── fd_check.py        # FD vs AD/解析解 tangent 比較
│   ├── fortran_bridge.py  # FortranUMAT — f2py 薄ラッパー（型変換のみ）
│   └── test_cases.py      # テストケース生成（estimate_yield_strain, generate_*）
└── utils/
    ├── voigt.py           # Voigt ↔ full tensor, Mandel 変換 (StressState 対応)
    ├── tensor.py          # 4 階テンソル操作
    └── smooth.py          # JAX 安全なスムーズ近似 (smooth_sqrt/abs/norm/macaulay/direction)

fortran/
├── j2_isotropic_3d.f90    # J2 UMAT 本体 (radial return + 7x7 tangent)
├── abaqus_stubs.f90       # SINV, SPRINC, ROTSIG モック (リンカ解決用)
├── test_basic.f90         # f2py 動作確認用
└── README.md              # ビルド手順
```

---

## Driver API

### データモデル

すべてのドライバは `FieldHistory` を入力に受け取り、`DriverResult` を返す。

| 型 | 役割 |
|----|------|
| `FieldType.STRESS` / `FieldType.STRAIN` | 物理空間（応力空間 / ひずみ空間）の分類 |
| `FieldHistory(type, name, data)` | 任意の物理量の時系列。`data` の shape は `(N, ntens)` (テンソル) または `(N,)` (スカラー) |
| `DriverResult` | `fields: dict[str, FieldHistory]` に全出力フィールドを保持。`.stress` / `.strain` プロパティで主要出力に直接アクセス |

### StrainDriver — ひずみ制御

```python
from manforge.simulation.driver import StrainDriver
from manforge.simulation.types import FieldHistory, FieldType
import numpy as np

# 単軸 (load.data.ndim == 1 → ε11 のみ処理、横方向は自動でゼロ)
load = FieldHistory(FieldType.STRAIN, "Strain", np.linspace(0, 5e-3, 100))
result = StrainDriver().run(model, load)
result.stress          # np.ndarray (N, ntens)
result.strain          # np.ndarray (N, ntens)

# 多軸 (load.data.ndim == 2 → 全成分を直接指定)
strain_6d = np.zeros((100, 6))
strain_6d[:, 0] = np.linspace(0, 5e-3, 100)   # ε11 のみ変化
load = FieldHistory(FieldType.STRAIN, "Strain", strain_6d)
result = StrainDriver().run(model, load)
```

`UniaxialDriver` / `GeneralDriver` は `StrainDriver` への後方互換エイリアス。

### StressDriver — 応力制御

目標応力列を指定し、ddsdde を使った Newton-Raphson でひずみ増分を逆解析する。

```python
from manforge.simulation.driver import StressDriver

stress_target = np.zeros((100, 6))
stress_target[:, 0] = np.linspace(0, 300.0, 100)  # σ11 を 300 MPa まで増加
load = FieldHistory(FieldType.STRESS, "Stress", stress_target)
result = StressDriver().run(model, load)
result.stress   # (N, ntens) — 収束後の応力 (目標値に一致)
result.strain   # (N, ntens) — 対応するひずみ
```

### 状態変数の収集

```python
result = StrainDriver().run(
    model, load,
    collect_state={"ep": FieldType.STRAIN}
)
result.fields["ep"].data   # np.ndarray (N,) — 等価塑性ひずみ履歴
```

---

## Adding a New Model

応力状態に対応した基底クラスを継承して必要なメソッドを実装するだけ。
return mapping / consistent tangent / fitting / verification はすべて自動で利用可能になる。

必須メソッドは `hardening_type` によって異なる:

| `hardening_type` | 必須メソッド |
|-----------------|-------------|
| `"explicit"` (デフォルト) | `elastic_stiffness`, `yield_function`, `hardening_increment` |
| `"implicit"` | `elastic_stiffness`, `yield_function`, `hardening_residual` |

`__init_subclass__` がクラス定義時にこれらの制約を検証する（実行時エラーではなく定義時 `TypeError`）。

```python
from manforge.core.material import MaterialModel3D  # または MaterialModelPS / MaterialModel1D
import jax.numpy as jnp

class MyModel(MaterialModel3D):
    param_names = ["E", "nu", "sigma_y0", "K"]
    state_names = ["ep"]

    def __init__(self, stress_state=None, *, E, nu, sigma_y0, K):
        from manforge.core.stress_state import SOLID_3D
        super().__init__(stress_state or SOLID_3D)
        self.E = E; self.nu = nu; self.sigma_y0 = sigma_y0; self.K = K

    def elastic_stiffness(self):
        mu = self.E / (2.0 * (1.0 + self.nu))
        lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        return self.isotropic_C(lam, mu)  # MaterialModel3D が提供する分岐なし実装

    def yield_function(self, stress, state):
        # self._vonmises() は MaterialModel ABC が smooth_sqrt ベースの実装を提供
        return self._vonmises(stress) - (self.sigma_y0 + ...)

    def hardening_increment(self, dlambda, stress, state):
        # 状態変数の更新。stress は現在の NR 反復点での応力テンソル。
        # 等方硬化のみなら stress を無視してよい。
        return {"ep": state["ep"] + dlambda}
```

参照実装: `src/manforge/models/j2_isotropic.py`

#### 要素タイプを指定して使う

```python
from manforge.core.stress_state import PLANE_STRAIN, PLANE_STRESS, UNIAXIAL_1D
from manforge.models.j2_isotropic import J2Isotropic3D, J2IsotropicPS, J2Isotropic1D

# 平面ひずみ / 軸対称 (NTENS=4) — full-rank, 解析解あり
model_pe = J2Isotropic3D(PLANE_STRAIN, E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)
# 平面応力 (NTENS=3) — Schur 縮約済み剛性, autodiff
model_ps = J2IsotropicPS(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)
# 単軸 (NTENS=1) — フィッティング用, 解析解あり
model_1d = J2Isotropic1D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)
```

利用可能なプリセット:

| インスタンス | NTENS | 想定要素 |
|------------|-------|---------|
| `SOLID_3D` | 6 | C3D8 等 3D ソリッド |
| `PLANE_STRAIN` | 4 | CPE4 / CAX4 (平面ひずみ・軸対称) |
| `PLANE_STRESS` | 3 | CPS4 / シェル (弾性剛性の静的凝縮済み) |
| `UNIAXIAL_1D` | 1 | 1D トラス / フィッティング用 |

### 解析解の追加 (オプション)

閉形式の return mapping が存在する場合、`plastic_corrector` と `analytical_tangent` をオーバーライドすると
`method="analytical"` での高速解析解パスが有効になる。

```python
class MyModel(MaterialModel3D):
    # ... 必須メソッド (elastic_stiffness, yield_function, hardening_increment) は省略 ...

    def plastic_corrector(self, stress_trial, C, state_n):
        """塑性補正の閉形式。None を返すと汎用 NR+autodiff にフォールバック。"""
        # stress_trial, C は return_mapping が計算済みで渡される
        # ...閉形式で stress_new, state_new, dlambda を計算...
        return stress_new, state_new, dlambda

    def analytical_tangent(self, stress, state, dlambda, C, state_n):
        """Consistent tangent の閉形式。None を返すと autodiff にフォールバック。"""
        # ...閉形式で ddsdde を計算...
        return ddsdde
```

`J2Isotropic3D` / `J2Isotropic1D` では両メソッドが実装済み (J2 radial return 閉形式 + `D^ep` の既知式)。

**dispatch 動作まとめ:**

| `method` | plastic correction | consistent tangent |
|----------|-------------------|-------------------|
| `"auto"` (デフォルト) | 解析解 (あれば) / NR+autodiff | 解析解 (あれば) / autodiff |
| `"autodiff"` | 常に NR+autodiff | 常に autodiff |
| `"analytical"` | 解析解 (なければ `NotImplementedError`) | 解析解 (なければ `NotImplementedError`) |

### 暗黙的硬化則 (implicit hardening)

`hardening_increment` が closed-form で解けない場合（後方 Euler 離散化に状態変数が非線形に現れる場合）、
`hardening_type = "implicit"` を宣言し `hardening_residual` を実装すると **augmented residual system** が自動的に有効になる。

`hardening_type == "implicit"` の場合、`return_mapping` は (ntens+1+n_state) の拡張残差系を Newton-Raphson で解き、
consistent tangent も `augmented_consistent_tangent()` で計算される。
**既存モデル (`J2Isotropic`, `AFKinematic`) は変更不要**（デフォルト `hardening_type = "explicit"` で動作する）。

`__init_subclass__` がクラス定義時に以下を強制する:
- `"explicit"` モデルが `hardening_increment` を未実装 → `TypeError`
- `"implicit"` モデルが `hardening_residual` を未実装 → `TypeError`
- 不正な `hardening_type` 値 → `TypeError`

```python
from manforge.core.material import MaterialModel3D
import jax.numpy as jnp

class MyImplicitModel(MaterialModel3D):
    hardening_type = "implicit"  # augmented NR パスを有効にする
    param_names = ["E", "nu", "sigma_y0", "C_k", "gamma"]
    state_names = ["alpha", "ep"]

    def __init__(self, stress_state=None, *, E, nu, sigma_y0, C_k, gamma):
        from manforge.core.stress_state import SOLID_3D
        super().__init__(stress_state or SOLID_3D)
        self.E = E; self.nu = nu; self.sigma_y0 = sigma_y0
        self.C_k = C_k; self.gamma = gamma

    def initial_state(self):
        return {"alpha": jnp.zeros(self.ntens), "ep": jnp.array(0.0)}

    def elastic_stiffness(self):
        mu = self.E / (2.0 * (1.0 + self.nu))
        lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        return self.isotropic_C(lam, mu)

    def yield_function(self, stress, state):
        xi = stress - state["alpha"]
        return self._vonmises(xi) - self.sigma_y0

    # hardening_increment は任意 — 初期値シードとして定義できる
    # def hardening_increment(self, dlambda, stress, state):
    #     return {"alpha": state["alpha"], "ep": state["ep"] + dlambda}

    def hardening_residual(self, state_new, dlambda, stress, state_n):
        """後方 Euler 残差。R_q(q_{n+1}, Δλ, σ, q_n) = 0 を定義する。
        zero at convergence.
        """
        alpha_new = state_new["alpha"]
        # ...モデル固有の残差を定義...
        R_alpha = ...  # shape (ntens,)
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        return {"alpha": R_alpha, "ep": R_ep}
```

`hardening_residual` の引数:

| 引数 | 説明 |
|------|------|
| `state_new` | 提案されたステップ n+1 の状態（独立変数） |
| `dlambda` | 塑性乗数増分 Δλ |
| `stress` | 現在の応力 σ_{n+1} |
| `state_n` | ステップ n 開始時の状態 |

戻り値は `state_new` と同じキー・形状を持つ残差 dict（収束時にゼロ）。

参照実装: `src/manforge/models/ow_kinematic.py`

### テンソル型状態変数 (移動硬化・backstress)

backstress のような 6 成分テンソル状態変数も `state` dict に格納できる。
`hardening_increment` が `stress` を受け取るため、flow direction `n = df/dσ` を内部で計算して
Armstrong-Frederick 型の移動硬化則を記述可能。

組み込みの `AFKinematic3D` / `AFKinematicPS` / `AFKinematic1D` を使う場合:

```python
import numpy as np
from manforge.models import AFKinematic3D
from manforge.simulation.driver import StrainDriver
from manforge.simulation.types import FieldHistory, FieldType

model = AFKinematic3D(
    E=210_000.0,    # Young's modulus [MPa]
    nu=0.3,
    sigma_y0=250.0, # Initial yield stress [MPa]
    C_k=8_000.0,    # Kinematic hardening modulus [MPa]
    gamma=80.0,     # Dynamic recovery (gamma=0 → Prager linear rule)
)

# 単軸引張-圧縮サイクル
strain_history = np.concatenate([np.linspace(0, 1e-2, 50), np.linspace(1e-2, -1e-2, 100)])
load = FieldHistory(FieldType.STRAIN, "Strain", strain_history)
result = StrainDriver().run(model, load)
```

飽和 backstress は `C_k / gamma`。`gamma=0` は Prager 線形移動硬化則と等価。
AF モデルは汎用 NR + JAX autodiff パス (解析フックなし) を使うため、
新規モデルの実装検証としても位置づけられる。

組み込みの `OWKinematic3D` / `OWKinematicPS` / `OWKinematic1D` (Ohno-Wang 修正 AF) を使う場合:

```python
from manforge.models import OWKinematic3D

model = OWKinematic3D(
    E=210_000.0,
    nu=0.3,
    sigma_y0=250.0,
    C_k=8_000.0,
    gamma=0.05,      # OW: γ ‖α‖ α (単位: 1/MPa)
)

result = StrainDriver().run(model, load)
```

Ohno-Wang モデルは `hardening_type = "implicit"` を宣言しており、
return mapping は自動的に (ntens+1+n_state) 拡張 NR を使用し、
consistent tangent も `augmented_consistent_tangent()` で計算される。

**AF と OW の飽和 backstress の違い:**

| モデル | 飽和 backstress 大きさ | γ の単位 |
|--------|----------------------|---------|
| `AFKinematic` | `C_k / γ` | 無次元 |
| `OWKinematic` | `√(C_k / γ)` | 1/MPa (backstress と同次元) |

同じ飽和振幅 `‖α_sat‖` を得るには `γ_OW = C_k / ‖α_sat‖²`、`γ_AF = C_k / ‖α_sat‖` とする。

カスタムで記述する場合のスケルトン:

```python
from manforge.core.material import MaterialModel3D
import jax.numpy as jnp

class MyKinematic(MaterialModel3D):
    param_names = ["E", "nu", "sigma_y0", "C_k", "gamma"]
    state_names = ["alpha", "ep"]

    def __init__(self, stress_state=None, *, E, nu, sigma_y0, C_k, gamma):
        from manforge.core.stress_state import SOLID_3D
        super().__init__(stress_state or SOLID_3D)
        self.E = E; self.nu = nu; self.sigma_y0 = sigma_y0
        self.C_k = C_k; self.gamma = gamma

    def initial_state(self):
        return {"alpha": jnp.zeros(self.ntens), "ep": jnp.array(0.0)}

    def yield_function(self, stress, state):
        xi = stress - state["alpha"]
        return self._vonmises(xi) - self.sigma_y0

    def hardening_increment(self, dlambda, stress, state):
        alpha_n = state["alpha"]
        xi = stress - alpha_n
        s_xi = self._dev(xi)
        # _vonmises は smooth_sqrt ベースで実装されているため、
        # ゼロ点でも微分可能 — smooth_abs でラップ不要
        vm_safe = self._vonmises(xi)
        n_hat = s_xi / vm_safe
        alpha_new = (alpha_n + self.C_k * dlambda * n_hat) / (1.0 + self.gamma * dlambda)
        return {"alpha": alpha_new, "ep": state["ep"] + dlambda}
```

同様の構造で Chaboche (複数 backstress)、Yoshida-Uemori (二面モデル)、
Lemaitre 損傷モデル等も記述可能。

---

## Smooth Utility Functions (`utils/smooth.py`)

`src/manforge/utils/smooth.py` は、ゼロ点・キンク点を含む全定義域で C∞ 微分可能な
スムーズ近似関数を提供する。内部実装は JAX プリミティブのみで構成されており、
`jax.grad` / `jax.jacobian` / 高次微分が安全に通る。

| 関数 | 近似対象 | 特記事項 |
|------|---------|---------|
| `smooth_sqrt(x, eps)` | `√x` | 基底プリミティブ。`√(x + ε²)` |
| `smooth_abs(x, eps)` | `\|x\|` | `smooth_sqrt(x²)` |
| `smooth_norm(v, eps)` | `‖v‖` | `smooth_sqrt(v·v)` |
| `smooth_macaulay(x, eps)` | `⟨x⟩ = max(x, 0)` | ランプ関数の滑らか版 |
| `smooth_direction(v, eps)` | `v / ‖v‖` | ゼロベクトルで NaN にならない |

デフォルト `eps = 1e-30`。float64 では `√(1e-30) ≈ 3e-16` なので MPa オーダーの計算に影響しない。

**なぜ `jnp.maximum(vm, eps)` ではないか**: `jnp.maximum` はクランプ境界でキンクを持つため、
`jax.jacobian` などの高次微分時に不連続な勾配を与える。`smooth_sqrt` は C∞ なのでこの問題がない。
`_vonmises` は内部で `smooth_sqrt` を使用しているため、呼び出し側で `smooth_abs` を重ねる必要はない。

```python
from manforge.utils.smooth import smooth_norm, smooth_direction

# ゼロベクトルでも安全な単位法線の計算
n_hat = smooth_direction(deviatoric_stress)

# 任意ベクトルのスムーズノルム
norm_val = smooth_norm(some_vector)
```

---

## Parameter Fitting

```python
from manforge.simulation.driver import StrainDriver
from manforge.fitting import fit_params  # または manforge.fitting.optimizer から

driver = StrainDriver()

fit_config = {
    "sigma_y0": (200.0, (100.0, 500.0)),  # (初期値, (下限, 上限))
    "H":        (500.0, (0.0,  5000.0)),
}
fixed_params = {"E": 210_000.0, "nu": 0.3}

result = fit_params(
    model, driver, exp_data,
    fit_config=fit_config,
    fixed_params=fixed_params,
    method="L-BFGS-B",
)
print(result.params, result.residual)
```

`exp_data` は `{"strain": ..., "stress": ...}` の dict。`strain` が 1-D array の場合は
`StrainDriver` が自動的に単軸モードで動作し、予測応力の σ11 成分と比較する。

詳細: `examples/example_fit_j2.py`

---

## Fortran Cross-Validation

`FortranUMAT` は f2py モジュールへの薄いラッパーで、型変換（float64）だけを担当する。
ルーチン全体もコンポーネントも同じ `call(name, *args)` パターンで呼べる:

```python
import numpy as np
import jax.numpy as jnp
import manforge
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.core.return_mapping import return_mapping
from manforge.verification import FortranUMAT

model   = J2Isotropic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)
dstran  = np.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
fortran = FortranUMAT("j2_isotropic_3d")

# Fortran ルーチンを呼ぶ
stress_f, ep_f, ddsdde_f = fortran.call(
    "j2_isotropic_3d",
    model.E, model.nu, model.sigma_y0, model.H,
    np.zeros(6), 0.0, dstran,
)

# Python 側と明示的に比較
stress_py, _, ddsdde_py = return_mapping(
    model, jnp.array(dstran), jnp.zeros(6), model.initial_state()
)
np.testing.assert_allclose(np.array(stress_py), stress_f, rtol=1e-6)
```

コンポーネント単位での比較も同じパターン（詳細: `fortran/README.md`）:

```python
C_f  = fortran.call("j2_isotropic_3d_elastic_stiffness", model.E, model.nu)
C_py = model.elastic_stiffness()
np.testing.assert_allclose(np.array(C_py), np.array(C_f), rtol=1e-12)
```

ビルド:

```bash
# CLI (推奨)
uv sync --extra fortran
uv run manforge build fortran/abaqus_stubs.f90 fortran/j2_isotropic_3d.f90 --name j2_isotropic_3d

# Makefile
make fortran-build-umat   # ホスト環境 (gfortran 必須)
make docker-test          # Docker 環境でビルド + テスト
```

詳細: `fortran/README.md`

---

## Analytical vs Autodiff Comparison (Python-to-Python)

Fortran UMAT がなくても、解析解と自動微分解をコード内で直接比較できる。

### 1 増分を直接比較

```python
from manforge.core.return_mapping import return_mapping
from manforge.models.j2_isotropic import J2Isotropic3D
import jax.numpy as jnp

model = J2Isotropic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)
strain_inc = jnp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])

s_ad, _, D_ad = return_mapping(model, strain_inc, jnp.zeros(6),
                               model.initial_state(), method="autodiff")
s_an, _, D_an = return_mapping(model, strain_inc, jnp.zeros(6),
                               model.initial_state(), method="analytical")
```

### 複数ケースを体系的に比較

```python
from manforge.verification.compare import compare_solvers

def make_solver(method):
    def _solve(strain_inc, stress_n, state_n):
        return return_mapping(model, jnp.asarray(strain_inc),
                              jnp.asarray(stress_n), state_n, method=method)
    return _solve

result = compare_solvers(
    solver_a=make_solver("autodiff"),
    solver_b=make_solver("analytical"),
    test_cases=[
        {"strain_inc": jnp.array([2e-3, 0, 0, 0, 0, 0]),
         "stress_n": jnp.zeros(6),
         "state_n": {"ep": jnp.array(0.0)}},
    ],
)
print(f"Passed: {result.passed}  max_stress_err={result.max_stress_rel_err:.2e}")
```

### 解析解 tangent の有限差分チェック

```python
from manforge.verification.fd_check import check_tangent

result = check_tangent(
    model, jnp.zeros(6), model.initial_state(),
    jnp.array([2e-3, 0, 0, 0, 0, 0]),
    method="analytical",
)
print(result.passed, result.max_rel_err)
```

---

## Testing

```bash
# Python テスト (Fortran テスト除く)
pytest tests/ -v

# Fortran クロス検証含む全テスト
make test-all

# Docker 内フルテスト
make docker-test
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

- **Voigt 記法**: 3D ソリッドは `[σ11, σ22, σ33, σ12, σ13, σ23]` (6成分)。次元は `StressState` で制御され、平面ひずみは `[σ11, σ22, σ33, σ12]` (4成分)、平面応力は `[σ11, σ22, σ12]` (3成分) 等に対応。内部演算では剪断成分に √2 を乗じた Mandel 表現を使用
- **StressState**: `ntens` / `ndi` / `nshr` / Mandel 係数をまとめた frozen dataclass。演算子 (`dev`, `vonmises`, `I_dev_voigt` 等) は `ss` 引数省略時に 3D 規定動作 (後方互換)
- **JAX float64**: `import manforge` 時に `jax.config.update("jax_enable_x64", True)` を自動実行
- **バージョン確認**: `import manforge; print(manforge.__version__)` で現在のバージョンを確認できる (`importlib.metadata` 経由)
- **jit 互換**: 現バージョンは正確性優先。jit 対応は将来課題 (`# TODO: jit` コメントで留保)
- **state 型**: `dict[str, jnp.ndarray]` で統一 (JAX pytree として自然に扱える)

---

## License

[MIT License](LICENSE)
