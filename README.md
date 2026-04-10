# manforge

Fortran UMAT (Abaqus) の検証・フィッティング用 Python フレームワーク。

降伏関数・硬化則・弾性テンソルを定義するだけで、return mapping / consistent tangent (自動微分) / パラメータフィッティング / Fortran クロス検証がすべて動く設計。

---

## Features

- **JAX 自動微分による consistent tangent** — 手動でアルゴリズム接線を導出する必要なし
- **汎用 return mapping ソルバ** — Closest point projection + Newton-Raphson (最大 50 反復、tol=1e-10)
- **パラメータフィッティング** — L-BFGS-B / Nelder-Mead / differential_evolution を scipy 経由で実行
- **Fortran UMAT クロス検証** — f2py 経由で Python 実装と Fortran UMAT の出力を自動比較
- **有限差分 tangent チェック** — AD と中心差分の DDSDDE を自動照合
- **MaterialModel ABC** — 3 メソッドを実装するだけで新規モデルに対応

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
from manforge.models.j2_isotropic import J2IsotropicHardening
from manforge.fitting.driver import UniaxialDriver

params = {
    "E": 210_000.0,    # Young's modulus [MPa]
    "nu": 0.3,         # Poisson's ratio
    "sigma_y0": 250.0, # Initial yield stress [MPa]
    "H": 1_000.0,      # Linear isotropic hardening modulus [MPa]
}

model = J2IsotropicHardening()
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
│   ├── material.py        # MaterialModel ABC
│   ├── return_mapping.py  # Closest point projection + Newton-Raphson
│   └── tangent.py         # Consistent tangent (暗黙微分)
├── autodiff/
│   ├── backend.py         # JAX バックエンド設定
│   └── operators.py       # dev, vonmises, hydrostatic, norm_mandel 等
├── models/
│   └── j2_isotropic.py    # J2 等方硬化 (参照実装)
├── fitting/
│   ├── driver.py          # UniaxialDriver, GeneralDriver
│   ├── objective.py       # 残差二乗和
│   └── optimizer.py       # fit_params() + FitResult
├── verification/
│   ├── fd_check.py        # FD vs AD tangent 比較
│   └── fortran_bridge.py  # FortranUMAT, compare_with_fortran()
└── utils/
    ├── voigt.py           # Voigt ↔ full tensor, Mandel 変換
    └── tensor.py          # 4 階テンソル操作

fortran/
├── umat_j2.f90            # J2 UMAT 本体 (radial return + 7x7 tangent)
├── abaqus_stubs.f90       # SINV, SPRINC, ROTSIG モック
├── wrapper.f90            # ISO_C_BINDING ラッパーテンプレート
├── test_basic.f90         # f2py 動作確認用
└── README.md              # ビルド手順
```

---

## Adding a New Model

`MaterialModel` ABC を継承して 3 つのメソッドを実装するだけ。
return mapping / consistent tangent / fitting / verification はすべて自動で利用可能になる。

```python
from manforge.core.material import MaterialModel
import jax.numpy as jnp

class MyModel(MaterialModel):
    param_names = ["E", "nu", "sigma_y0", "K"]
    state_names = ["ep"]

    def elastic_stiffness(self, params):
        # ntens x ntens の弾性剛性テンソル (Voigt)
        return self.isotropic_C(lam, mu)

    def yield_function(self, stress, state, params):
        # f(σ, q) <= 0 で弾性域
        return vonmises(stress) - (params["sigma_y0"] + ...)

    def hardening_increment(self, dlambda, state, params):
        # 状態変数の更新
        return {"ep": state["ep"] + dlambda}
```

参照実装: `src/manforge/models/j2_isotropic.py`

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
from manforge.verification.fortran_bridge import FortranUMAT, compare_with_fortran

fortran_umat = FortranUMAT("manforge_umat", model)

result = compare_with_fortran(model, fortran_umat, test_cases)
print(f"Passed: {result.passed}  ({result.n_passed}/{result.n_cases})")
print(f"Max stress rel err: {result.max_stress_rel_err:.2e}")
```

ビルド:

```bash
make fortran-build-umat   # ホスト環境 (gfortran 必須)
make docker-test          # Docker 環境でビルド + テスト
```

詳細: `fortran/README.md`

---

## Testing

```bash
# Python テスト (37 件)
pytest tests/ -v

# Fortran クロス検証含む全テスト (48 件)
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

- **Voigt 記法**: 応力/ひずみ成分順 `[σ11, σ22, σ33, σ12, σ13, σ23]`。内部演算では剪断成分に √2 を乗じた Mandel 表現を使用
- **JAX float64**: `import manforge` 時に `jax.config.update("jax_enable_x64", True)` を自動実行
- **jit 互換**: 現バージョンは正確性優先。jit 対応は将来課題 (`# TODO: jit` コメントで留保)
- **state 型**: `dict[str, jnp.ndarray]` で統一 (JAX pytree として自然に扱える)

---

## License

MIT License
