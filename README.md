# manforge

Fortran UMAT (Abaqus) の検証・フィッティング用 Python フレームワーク。

降伏関数・硬化則・弾性テンソルを定義するだけで、return mapping / consistent tangent (自動微分) / パラメータフィッティング / Fortran クロス検証がすべて動く設計。

---

## Features

- **JAX 自動微分による consistent tangent** — 手動でアルゴリズム接線を導出する必要なし
- **汎用 return mapping ソルバ** — Closest point projection + Newton-Raphson (最大 50 反復、tol=1e-10)
- **多次元対応 (StressState)** — 3D ソリッド (NTENS=6) / 平面ひずみ・軸対称 (NTENS=4) / 平面応力 (NTENS=3) / 1D トラス (NTENS=1) に対応。既存コードは後方互換
- **解析解 vs 自動微分の Python 同士比較** — 閉形式解が存在するモデルでは `method="analytical"` / `"autodiff"` を切り替えてクロス検証
- **パラメータフィッティング** — L-BFGS-B / Nelder-Mead / differential_evolution を scipy 経由で実行
- **Fortran UMAT クロス検証** — f2py 経由で Python 実装と Fortran UMAT の出力を自動比較
- **汎用ソルバ比較** — `compare_solvers()` で任意の 2 つのソルバ出力 (stress, DDSDDE) を自動照合
- **有限差分 tangent チェック** — AD / 解析解のどちらも中心差分で自動照合
- **応力状態基底クラス** — `MaterialModel3D` / `MaterialModelPS` / `MaterialModel1D` を継承して 3 メソッドを実装するだけで新規モデルに対応。演算子 (`_vonmises`, `_dev`, `isotropic_C` 等) は基底が提供

---

## Requirements

- Python >= 3.10
- `jax[cpu]` >= 0.4.20
- `scipy`
- (optional) `gfortran` — Fortran クロス検証用
- (optional) `matplotlib` — examples のプロット用

---

## Installation

```bash
# 基本インストール
uv sync

# テスト依存含む
uv sync --extra dev

# matplotlib 含む
uv sync --extra examples

# Fortran モジュールのビルド
make fortran-build-umat

# Docker 内でフルビルド + テスト
make docker-build && make docker-test
```

---

## Quick Start

```python
import manforge  # JAX float64 モードを有効化
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.simulation.driver import UniaxialDriver

params = {
    "E": 210_000.0,    # Young's modulus [MPa]
    "nu": 0.3,         # Poisson's ratio
    "sigma_y0": 250.0, # Initial yield stress [MPa]
    "H": 1_000.0,      # Linear isotropic hardening modulus [MPa]
}

model = J2Isotropic3D()
driver = UniaxialDriver()

import numpy as np
strain_history = np.linspace(0.0, 5e-3, 100)       # cumulative ε11
stress_history = driver.run(model, strain_history, params)  # σ11 history

print(f"Final stress: {stress_history[-1]:.1f} MPa")
```

より詳しい使用例は `examples/` ディレクトリを参照。

---

## Architecture

```
src/manforge/
├── core/
│   ├── stress_state.py    # StressState — 次元記述子 (SOLID_3D / PLANE_STRAIN / PLANE_STRESS / UNIAXIAL_1D)
│   ├── material.py        # MaterialModel ABC + MaterialModel3D/PS/1D (応力状態基底)
│   ├── return_mapping.py  # Closest point projection + Newton-Raphson
│   └── tangent.py         # Consistent tangent (暗黙微分)
├── autodiff/
│   ├── backend.py         # JAX バックエンド設定
│   └── operators.py       # dev, vonmises, hydrostatic, norm_mandel 等 (StressState 対応)
├── models/
│   └── j2_isotropic.py    # J2 等方硬化 (参照実装)
├── fitting/
│   ├── driver.py          # UniaxialDriver, GeneralDriver
│   ├── objective.py       # 残差二乗和
│   └── optimizer.py       # fit_params() + FitResult
├── verification/
│   ├── compare.py         # compare_solvers() — 汎用ソルバ比較
│   ├── fd_check.py        # FD vs AD/解析解 tangent 比較
│   └── fortran_bridge.py  # FortranUMAT — f2py ブリッジ
└── utils/
    ├── voigt.py           # Voigt ↔ full tensor, Mandel 変換 (StressState 対応)
    └── tensor.py          # 4 階テンソル操作

fortran/
├── umat_j2.f90            # J2 UMAT 本体 (radial return + 7x7 tangent)
├── abaqus_stubs.f90       # SINV, SPRINC, ROTSIG モック (リンカ解決用)
├── test_basic.f90         # f2py 動作確認用
└── README.md              # ビルド手順
```

---

## Adding a New Model

応力状態に対応した基底クラスを継承して 3 つのメソッドを実装するだけ。
return mapping / consistent tangent / fitting / verification はすべて自動で利用可能になる。

```python
from manforge.core.material import MaterialModel3D  # または MaterialModelPS / MaterialModel1D
import jax.numpy as jnp

class MyModel(MaterialModel3D):
    param_names = ["E", "nu", "sigma_y0", "K"]
    state_names = ["ep"]
    # __init__ 不要 — MaterialModel3D.__init__(SOLID_3D) が MRO 経由で呼ばれる
    # PLANE_STRAIN で使いたい場合は MyModel(PLANE_STRAIN) と渡せばよい

    def elastic_stiffness(self, params):
        E, nu = params["E"], params["nu"]
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return self.isotropic_C(lam, mu)  # MaterialModel3D が提供する分岐なし実装

    def yield_function(self, stress, state, params):
        # self._vonmises() は MaterialModel3D が正しい実装を提供
        return self._vonmises(stress) - (params["sigma_y0"] + ...)

    def hardening_increment(self, dlambda, stress, state, params):
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
model_pe = J2Isotropic3D(PLANE_STRAIN)
# 平面応力 (NTENS=3) — Schur 縮約済み剛性, autodiff
model_ps = J2IsotropicPS()
# 単軸 (NTENS=1) — フィッティング用, 解析解あり
model_1d = J2Isotropic1D()
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
    # ... 必須の 3 メソッドは省略 ...

    def plastic_corrector(self, stress_trial, C, state_n, params):
        """塑性補正の閉形式。None を返すと汎用 NR+autodiff にフォールバック。"""
        # stress_trial, C は return_mapping が計算済みで渡される
        # ...閉形式で stress_new, state_new, dlambda を計算...
        return stress_new, state_new, dlambda

    def analytical_tangent(self, stress, state, dlambda, C, state_n, params):
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

### テンソル型状態変数 (移動硬化・backstress)

backstress のような 6 成分テンソル状態変数も `state` dict に格納できる。
`hardening_increment` が `stress` を受け取るため、flow direction `n = df/dσ` を内部で計算して
Armstrong-Frederick 型の移動硬化則を記述可能。

```python
import jax
from manforge.core.material import MaterialModel3D
import jax.numpy as jnp

class J2KinematicHardening(MaterialModel3D):
    param_names = ["E", "nu", "sigma_y0", "H_iso", "C_kin", "gamma"]
    state_names = ["ep", "alpha"]
    # __init__ 不要 — MaterialModel3D.__init__(SOLID_3D) が MRO 経由で呼ばれる

    def initial_state(self):
        ntens = self.stress_state.ntens
        return {"ep": jnp.array(0.0), "alpha": jnp.zeros(ntens)}

    def yield_function(self, stress, state, params):
        xi = stress - state["alpha"]  # 移動した応力
        # self._vonmises() は MaterialModel3D が提供する分岐なし実装
        return self._vonmises(xi) - (params["sigma_y0"] + params["H_iso"] * state["ep"])

    def hardening_increment(self, dlambda, stress, state, params):
        # flow direction を JAX 自動微分で取得 (Mandel 重み付け自動適用)
        n = jax.grad(lambda s: self._vonmises(s - state["alpha"]))(stress)
        # Armstrong-Frederick: dα = (2/3)*C*Δλ*n − γ*Δλ*α
        alpha_new = state["alpha"] + (2/3)*params["C_kin"]*dlambda*n \
                    - params["gamma"]*dlambda*state["alpha"]
        return {"ep": state["ep"] + dlambda, "alpha": alpha_new}
```

同様の構造で Chaboche (複数 backstress)、Yoshida-Uemori (二面モデル)、
Lemaitre 損傷モデル等も記述可能。

---

## Parameter Fitting

```python
from manforge.fitting.optimizer import fit_params

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

詳細: `examples/example_fit_j2.py`

---

## Fortran Cross-Validation

```python
from manforge.verification import UMATVerifier

verifier = UMATVerifier(model, "manforge_umat")
result   = verifier.run(params)
print(result.summary())
```

`run` は2フェーズを自動実行する:
1. **単一ステップ比較** — 降伏ひずみを自動推定し、弾性・塑性・多軸・せん断の計5ケースを生成して `compare_solvers` で比較
2. **多ステップ比較** — 引張→除荷→圧縮のサイクル履歴を両ソルバーで独立に走らせ、ステップごとに応力・状態変数・tangentを比較

テストケース生成を単独で使うこともできる:

```python
from manforge.verification.test_cases import generate_single_step_cases
from manforge.verification import compare_solvers

cases  = generate_single_step_cases(model, params)
result = compare_solvers(solver_a, solver_b, cases)
```

ビルド:

```bash
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

model = J2Isotropic3D()
params = {"E": 210_000.0, "nu": 0.3, "sigma_y0": 250.0, "H": 1_000.0}
strain_inc = jnp.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0])

s_ad, _, D_ad = return_mapping(model, strain_inc, jnp.zeros(6),
                               model.initial_state(), params, method="autodiff")
s_an, _, D_an = return_mapping(model, strain_inc, jnp.zeros(6),
                               model.initial_state(), params, method="analytical")
```

### 複数ケースを体系的に比較

```python
from manforge.verification.compare import compare_solvers

def make_solver(method):
    def _solve(strain_inc, stress_n, state_n, params):
        return return_mapping(model, jnp.asarray(strain_inc),
                              jnp.asarray(stress_n), state_n, params, method=method)
    return _solve

result = compare_solvers(
    solver_a=make_solver("autodiff"),
    solver_b=make_solver("analytical"),
    test_cases=[
        {"strain_inc": jnp.array([2e-3, 0, 0, 0, 0, 0]),
         "stress_n": jnp.zeros(6),
         "state_n": {"ep": jnp.array(0.0)},
         "params": params},
    ],
)
print(f"Passed: {result.passed}  max_stress_err={result.max_stress_rel_err:.2e}")
```

### 解析解 tangent の有限差分チェック

```python
from manforge.verification.fd_check import check_tangent

result = check_tangent(
    model, jnp.zeros(6), model.initial_state(), params,
    jnp.array([2e-3, 0, 0, 0, 0, 0]),
    method="analytical",
)
print(result.passed, result.max_rel_err)
```

---

## Testing

```bash
# Python テスト (112 件、Fortran テスト除く)
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
- **jit 互換**: 現バージョンは正確性優先。jit 対応は将来課題 (`# TODO: jit` コメントで留保)
- **state 型**: `dict[str, jnp.ndarray]` で統一 (JAX pytree として自然に扱える)

---

## License

MIT License
