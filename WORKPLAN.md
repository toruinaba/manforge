# manforge パッケージ 作業計画

## 概要

Fortran UMAT (Abaqus) の検証・フィッティング用 Python パッケージ `manforge` を新規作成する。
ユーザーは降伏関数・硬化則・弾性テンソルのみ定義すれば、return mapping / consistent tangent (AD) /
パラメータフィッティング / Fortran クロス検証がすべて動く設計。
仕様詳細は `plan.md` を参照。

## 成果物ルートディレクトリ

```
/home/ec2-user/manforge/manforge/
```

---

## Step 1 — プロジェクト基盤 + 依存インストール

**作成ファイル:**
- `pyproject.toml` — hatchling + uv, src-layout, 依存: `jax[cpu]>=0.4.20`, `scipy`, dev extras に `pytest`
- `README.md` — 概要・インストール・クイックスタート
- `src/manforge/__init__.py` — `jax.config.update("jax_enable_x64", True)` を最初に実行
- 全サブパッケージの `__init__.py` (空ファイル):
  - `src/manforge/core/__init__.py`
  - `src/manforge/autodiff/__init__.py`
  - `src/manforge/models/__init__.py`
  - `src/manforge/fitting/__init__.py`
  - `src/manforge/verification/__init__.py`
  - `src/manforge/utils/__init__.py`

**検証:** `uv sync` → `python -c "import manforge"`

---

## Step 2 — utils/ と autodiff/ (基盤ライブラリ) + テスト

**作成ファイル:**
- `src/manforge/utils/voigt.py` — Voigt ↔ full tensor 変換, Mandel 変換係数を明示コメント
- `src/manforge/utils/tensor.py` — 4階テンソル操作
- `src/manforge/autodiff/backend.py` — JAX バックエンド設定 (float64 guard)
- `src/manforge/autodiff/operators.py` — `dev`, `vonmises`, `hydrostatic`, `norm_mandel`, `I_dev_voigt`, `I_vol_voigt`, `identity_voigt` (全て `jnp` で AD 対応)
- `tests/conftest.py` — `steel_params` fixture (E=210000, nu=0.3, sigma_y0=250, H=1000)
- `tests/test_operators.py` — `dev`, `vonmises` の既知値テスト

**検証:** `pytest tests/test_operators.py -v`

---

## Step 3 — core/material.py と models/j2_isotropic.py

**作成ファイル:**
- `src/manforge/core/material.py` — `MaterialModel` ABC
  - 抽象メソッド: `elastic_stiffness`, `yield_function`, `hardening_increment`
  - デフォルト実装: `isotropic_C(lam, mu)`, `initial_state()`
- `src/manforge/models/j2_isotropic.py` — J2 + 等方硬化の参照実装

**検証:** `pytest tests/ -v` (回帰確認)

---

## Step 4a — core/tangent.py

**作成ファイル:**
- `src/manforge/core/tangent.py` — `consistent_tangent(stress_fn, strain_inc)`: `jax.jacobian` で σ_{n+1}(Δε) を微分

---

## Step 4b — core/return_mapping.py + テスト

**作成ファイル:**
- `src/manforge/core/return_mapping.py`
  - 入力: `model`, `strain_inc (ntens,)`, `stress_n`, `state_n`, `params`
  - 出力: `(stress_new, state_new, ddsdde)`
  - closest point projection (associative flow rule)
  - Newton-Raphson 反復 (max=50, tol=1e-10)
  - `jax.grad` で df/dσ を自動計算、consistent tangent は tangent.py を使用
  - jit 互換は `# TODO: jit` コメントで留保
- `tests/test_j2_elastic.py` — 弾性域で DDSDDE == C_elastic
- `tests/test_j2_plastic.py` — 単軸引張, 降伏後応力が解析解と一致
- `tests/test_tangent_fd.py` — AD tangent vs 中心差分 FD (rtol=1e-5)

**検証:** `pytest tests/ -v`

---

## Step 5 — fitting/ + テスト

**作成ファイル:**
- `src/manforge/fitting/driver.py` — `UniaxialDriver`, `BiaxialDriver`, `GeneralDriver`
- `src/manforge/fitting/objective.py` — 実験データとの残差二乗和
- `src/manforge/fitting/optimizer.py` — `fit_params()` API, `FitResult` dataclass
  ```python
  fit_config = {"param_name": (initial, (lb, ub)), ...}
  result = fit_params(model, driver, exp_data, fit_config, fixed_params, method)
  ```
- `tests/test_fitting.py` — 合成データへのフィッティング収束 (煙テスト)

**検証:** `pytest tests/ -v`

---

## Step 6 — verification/

**作成ファイル:**
- `src/manforge/verification/fd_check.py`
  - `check_tangent(model, stress, state, params, strain_inc, eps=1e-7, tol=1e-5) -> bool`
- `src/manforge/verification/fortran_bridge.py`
  - `FortranUMAT` クラス (f2py / ctypes ロード)
  - `compare_with_fortran(model, fortran_umat, test_cases)` → 比較レポート

**検証:** `pytest tests/ -v` (回帰確認)

---

## Step 7 — examples/

**作成ファイル:**
- `examples/example_j2_uniaxial.py` — 単軸引張シミュレーション + matplotlib プロット
- `examples/example_fit_j2.py` — J2 パラメータフィッティング例

**検証:** `python examples/example_j2_uniaxial.py && python examples/example_fit_j2.py`

---

## Step 8 — fortran/ 雛形

**作成ファイル:**
- `fortran/wrapper.f90` — ISO_C_BINDING ラッパーテンプレート
- `fortran/README.md` — f2py ビルド手順

---

## 実装上の重要事項

| 項目 | 方針 |
|------|------|
| float64 | `__init__.py` で `jax.config.update("jax_enable_x64", True)` を最初に実行 |
| JAX immutability | `array.at[i].set(v)` を使用 |
| jit 互換 | 初期実装は jit なしで正確性を優先、jit 対応箇所は `# TODO: jit` コメントを残す |
| state 型 | `dict[str, jnp.ndarray]` で統一 (JAX pytree として自然に扱える) |
| Voigt 剪断係数 | operators.py 内で Mandel 係数 (√2) vs 工学ひずみ の違いを明示コメント |
| docstring | NumPy スタイルで記述 |

---

## 最終検証

```bash
cd /home/ec2-user/manforge/manforge
pytest tests/ -v
python examples/example_j2_uniaxial.py
python examples/example_fit_j2.py
```
