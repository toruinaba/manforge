# constitutive パッケージのスケルトン作成

## 概要

Fortran UMAT (Abaqus) の検証・フィッティング用Pythonパッケージ `constitutive` のたたきを作成せよ。
ユーザーは降伏関数・硬化則・弾性テンソルだけ定義すれば、return mapping、consistent tangent (自動微分)、パラメータフィッティング、Fortranクロス検証がすべて動く設計とする。

## 技術スタック

- Python >=3.10
- ビルド: hatchling + uv (pyproject.toml, src layout)
- 自動微分: JAX (jax[cpu]>=0.4.20), float64必須 (jax_enable_x64)
- 最適化: scipy.optimize
- テスト: pytest
- Fortran連携: f2py (optional, fortran/ ディレクトリに分離)

## ディレクトリ構成

constitutive/
├── pyproject.toml
├── README.md
├── src/
│ └── constitutive/
│ ├── **init**.py # jax_enable_x64 設定
│ ├── core/
│ │ ├── **init**.py
│ │ ├── material.py # MaterialModel ABC
│ │ ├── return_mapping.py # 汎用 return mapping ソルバ
│ │ └── tangent.py # consistent tangent (AD)
│ ├── autodiff/
│ │ ├── **init**.py
│ │ ├── backend.py # JAX バックエンド設定
│ │ └── operators.py # dev, vonmises, norm_mandel, I_dev, I_vol 等
│ ├── models/
│ │ ├── **init**.py
│ │ └── j2_isotropic.py # J2 + 等方硬化 (参照実装・使用例)
│ ├── fitting/
│ │ ├── **init**.py
│ │ ├── driver.py # UniaxialDriver, BiaxialDriver, GeneralDriver
│ │ ├── objective.py # 残差・目的関数構築
│ │ └── optimizer.py # fit_params() API
│ ├── verification/
│ │ ├── **init**.py
│ │ ├── fd_check.py # 数値微分 vs AD 自動比較
│ │ └── fortran_bridge.py # f2py/ctypes 経由 Fortran 呼び出し
│ └── utils/
│ ├── **init**.py
│ ├── voigt.py # Voigt記法ユーティリティ (6成分, 平面応力4成分対応)
│ └── tensor.py # 4階テンソル操作
├── fortran/
│ ├── wrapper.f90 # ISO_C_BINDING ラッパーの雛形
│ └── README.md # f2py ビルド手順
├── tests/
│ ├── conftest.py # 共通fixture (材料パラメータ等)
│ ├── test_operators.py # Voigt演算の単体テスト
│ ├── test_j2_elastic.py # 弾性域のテスト
│ ├── test_j2_plastic.py # 塑性域 return mapping テスト
│ ├── test_tangent_fd.py # AD tangent vs FD の整合テスト
│ └── test_fitting.py # フィッティングの煙テスト
└── examples/
├── example_j2_uniaxial.py # 単軸引張シミュレーション + プロット
└── example_fit_j2.py # J2パラメータフィッティング例

## 各モジュールの設計仕様

### 1. MaterialModel ABC (core/material.py)

from abc import ABC, abstractmethod

class MaterialModel(ABC):
param_names: list[str] # パラメータ名リスト
state_names: list[str] # 内部変数名リスト
ntens: int = 6 # 応力・ひずみ成分数 (default: 3D)

```
@abstractmethod
def elastic_stiffness(self, params: dict) -> jnp.ndarray:
"""(ntens, ntens) 弾性剛性テンソル (Voigt)"""

@abstractmethod
def yield_function(self, stress: jnp.ndarray, state: dict, params: dict) -> float:
"""降伏関数 f(σ, q). f<=0 で弾性域"""

@abstractmethod
def hardening_increment(self, dlambda: float, state: dict, params: dict) -> dict:
"""状態変数の更新 state_{n+1} = g(Δλ, state_n)"""

# --- フレームワーク側が提供するヘルパー (ABC内にデフォルト実装) ---
def isotropic_C(self, lam: float, mu: float) -> jnp.ndarray:
"""等方弾性テンソル生成"""

def initial_state(self) -> dict:
"""状態変数のゼロ初期化"""
```

### 2. return_mapping (core/return_mapping.py)

- 入力: model, strain_inc (ntens,), stress_n, state_n, params
- 出力: stress_{n+1}, state_{n+1}, ddsdde (ntens, ntens)
- cutting plane法ではなく、associative flow rule前提のclosest point projection
- Newton-Raphson反復 (最大50反復, tol=1e-10)
- 降伏関数の勾配 df/dσ は jax.grad で自動計算
- consistent tangent は jax.jacobian で σ_{n+1}(Δε) を微分して取得
- jax.jit でコンパイル可能にする (pure function 設計)

### 3. operators (autodiff/operators.py)

全て jax.numpy で実装し AD 対応にすること:

- dev(stress_voigt) → 偏差応力
- vonmises(stress_voigt) → von Mises 相当応力
- hydrostatic(stress_voigt) → 静水圧
- norm_mandel(t_voigt) → Mandel ノルム
- I_dev_voigt() → 偏差射影テンソル (6x6)
- I_vol_voigt() → 体積射影テンソル (6x6)
- identity_voigt() → δ_{ij} の Voigt 表現

Voigtの剪断成分は工学ひずみ(γ=2ε)ではなく、
応力側は σ = [σ11,σ22,σ33,σ12,σ13,σ23] の6成分で統一。
内部演算で Mandel 変換が必要な場合はユーティリティを用意する。

### 4. fitting (fitting/)

- UniaxialDriver: ひずみ増分列を受け取り、return_mappingをループして応力履歴を返す
- objective.py: 実験データ (strain, stress) と計算結果の残差二乗和
- optimizer.py:
- fit_params(model, driver, exp_data, fit_config, fixed_params, method) が主API
- fit_config = {“param_name”: (initial, (lb, ub)), …}
- 内部で scipy.optimize.minimize または differential_evolution を呼ぶ
- 戻り値は FitResult dataclass (params, residual, history, plot())

### 5. verification (verification/)

- fd_check.py:
- check_tangent(model, stress, state, params, strain_inc, eps=1e-7, tol=1e-5) → bool
- 中心差分でDDSDDEを計算し、AD結果と比較
- 相対誤差の最大値を返す
- fortran_bridge.py:
- FortranUMAT クラス: f2pyモジュールをロードし、Python側と同じI/Fで呼べるラッパー
- compare_with_fortran(model, fortran_umat, test_cases) → 比較レポート

### 6. テスト (tests/)

- conftest.py: steel_params fixture (E=210000, nu=0.3, sigma_y0=250, H=1000)
- test_operators.py: dev, vonmises の既知値テスト
- test_j2_elastic.py: 弾性域で DDSDDE == C_elastic であることの確認
- test_j2_plastic.py: 単軸引張で降伏後の応力が解析解と一致
- test_tangent_fd.py: AD tangent vs FD が rtol=1e-5 で一致
- test_fitting.py: 合成データに対してフィッティングが収束する煙テスト

## 実装上の注意

1. JAX の float64: **init**.py で jax.config.update(“jax_enable_x64”, True) 必須
2. JAX 配列は immutable: .at[i].set(v) を使う
3. jax.jit 互換にするため、Python の if/for ではなく jax.lax.cond / jax.lax.fori_loop を使うか、
まずは jit なしで正しく動くことを優先し、jit 対応は TODO コメントで残す
4. state は dict[str, jnp.ndarray] で統一。JAX の pytree として自然に扱える
5. Voigt 記法の剪断成分の係数 (√2 Mandel vs 工学ひずみ) は operators.py 内で明示的にコメントすること
6. docstring は numpy スタイルで書く

## 作業手順

1. pyproject.toml, README.md, **init**.py を作成
2. utils/ (voigt.py, tensor.py) → autodiff/ (operators.py) の順で基盤を実装
3. core/material.py (ABC) → models/j2_isotropic.py の順でモデル層を実装
4. core/return_mapping.py, core/tangent.py を実装
5. fitting/ (driver.py → objective.py → optimizer.py) を実装
6. verification/ (fd_check.py, fortran_bridge.py) を実装
7. tests/ を作成し、pytest で全テスト通過を確認
8. examples/ を作成
9. fortran/ の雛形を作成

各ステップ完了ごとに pytest を実行し、テストが通ることを確認してから次に進むこと。

