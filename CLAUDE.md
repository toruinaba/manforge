# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**API 契約の正は各モジュールの docstring。** この文書はコマンド・テスト規約・コードから導けない横断契約のみを持つ。設計意図の詳細は該当モジュール冒頭や該当メソッドの docstring を読むこと(例: `elastic_check` の分岐設計は `core/material/base.py`、fitting の失敗ペナルティ設計は `fitting/objective.py`、PCM の自前実装理由は `fitting/metrics.py`)。

## Commands

```bash
# Install (package manager: uv)
uv sync --extra dev          # Python tests
uv sync --all-extras         # All extras including matplotlib, meson/ninja

# Test
make test                    # Fast tests: unit + integration, excluding slow/fortran
make test-all                # Full suite including slow and fortran
# その他のターゲット (unit/integration/e2e/slow/benchmarks/docker) は Makefile を参照
uv run pytest tests/benchmarks/j2_isotropic/ -m "not fortran" -v  # Single benchmark dir

# Fortran compilation (requires gfortran; this host has none — use Docker)
uv run manforge build fortran/abaqus_stubs.f90 fortran/j2_isotropic_3d.f90 --name j2_isotropic_3d
uv run manforge list / clean
make docker-build && make docker-test
```

There is no linter configured. No CI/CD workflows exist.

## Architecture

manforge is a framework for validating Fortran UMAT (Abaqus user material) constitutive models. Users define a material model in Python; the framework handles return mapping, consistent tangent computation (via autograd), parameter fitting, and cross-validation against compiled Fortran UMAT subroutines. Float64 is enabled globally in `src/manforge/__init__.py`.

### Three-layer design

1. **Material model** — `core/material/base.py`: `MaterialModel` ABC。状態変数は `Implicit` / `Explicit` のクラス属性で宣言し、`__init_subclass__` が `state_names` 等を導出・検証する。ユーザーが書くのは `yield_function`(必須)+ `update_state`(explicit 状態)/ `state_residual`(implicit 状態)。特殊フック(`elastic_check` / `elastic_update_state` / `user_defined_return_mapping` / `user_defined_tangent` / `flow` override / `dlambda` 行差し替え)の契約と設計理由はすべて base.py の docstring。
2. **Solver** — `simulation/integrator/`: `stress_update` = UMAT 1回分(弾性予測 → 分岐 → return mapping → consistent tangent)。NR 経路(スカラー/ベクトル)は宣言から自動選択。`nr_jacobian="broyden"` は初回のみ厳密ヤコビアン + Broyden 更新(接線は常に厳密)。NR 未知数ベクトルのレイアウトは `simulation/_layout.py`(`ResidualLayout`)、統一残差は `simulation/_residual.py` が単一の真実。
3. **Application** — `simulation/driver.py`(`StrainDriver` / `StressDriver` / `MixedDriver`、`run` / `iter_run`)、`fitting/`(`fit_params` + `ParameterSpace` / `FitTarget` / `Observation` / metrics)、`verification/`(`JacobianChecker` / `TangentChecker` / `CrosscheckStrainDriver` 等)、`service/`(WebAPI 向け契約層: `ModelRegistry` / `run_simulation` / `ParamMeta`。Web 依存ゼロ、plain I/O、非収束は値で返す)。

### Voigt convention(横断契約 — 全層が従う)

3D ソリッドは `[11, 22, 33, 12, 13, 23]`(ABAQUS UMAT 規約)。**量の種類で剪断の扱いが異なる 2 層契約**:

- **応力系**(σ, s, 背応力 α など): **physical shear** — テンソル成分そのもの
- **ひずみ系**(ε, ε_p, Δε): **engineering shear**(γ12 = 2ε12)— ABAQUS `DSTRAN`・弾性剛性 `σ = C:ε`・driver 入力すべて

ノルムや等価量の計算では Mandel スケーリング(剪断 ×√2)を内部適用。ヘルパーは `utils/voigt.py`、次元ごとの演算子(`dev` / `vonmises` / `inner_product` / `strain_norm` 等)は `core/dimension.py` の `StressDimension` に集約され `MaterialModel` がデリゲートする。

### FlowVector(横断契約)

塑性流れ方向は同一方向に 2 つの Voigt 表現(剪断 ×2 差)があり、取り違えは単軸/二軸で消える silent error になる。そのため `model.flow(state)` は生 ndarray でなく `FlowVector` を返し、使用側が `.strain_like`(engineering shear — `C @ n`、Δε_p)か `.stress_like`(physical shear — 背応力発展則)を名指しする。生配列演算は `TypeError`。`flow` を override するときは `self.strain_flow(...)` / `self.stress_flow(...)` でタグ付けして返す(モデル内で ×2/÷2 の変換算術を書かない)。詳細は `core/state.py`。

### 非収束の扱い(層ごとに方針が違う)

- **Integrator**: 既定 raise。`raise_on_nonconverged=False` で `converged=False` を返す。Δλ<0 は KKT 違反なので非収束扱い。許容値はスケール不変(`tol × max(1, ‖σ_trial‖_∞)`)。
- **Driver**: 非収束 step を yield して停止(非収束イテレートで状態を進めない)。
- **fitting**: 評価の非収束は有限ペナルティ(`FAILURE_PENALTY`)でスコアし続行。全評価失敗時のみ raise。設計理由は `fitting/objective.py`。
- **service**: `run_simulation` は部分履歴 + `converged` / `failed_step` を値で返す(呼び出し側バグのみ `ValueError`)。

### Test layout

`tests/{unit,integration}/` mirrors `src/manforge/` 1:1 at the file level. `tests/e2e/` covers CLI subprocess tests and the fitting pipeline. `tests/benchmarks/<model>/` is the home of numerical-equivalence harnesses (analytical-vs-numerical, Python-vs-Fortran). `@pytest.mark.fortran` auto-skips when the compiled `.so` is missing; `@pytest.mark.slow` is opt-in for >1s cases.

| directory | purpose |
|---|---|
| `tests/unit/` | Pure-logic tests — no integrator, no driver |
| `tests/integration/` | Orchestration tests — integrator + driver in the loop |
| `tests/e2e/` | CLI subprocess + fitting pipeline smoke |
| `tests/benchmarks/` | Numerical equivalence harnesses only |

### Models / Fortran

参照実装は `src/manforge/models/`(J2 / AF / OW / YU — 各ファミリ親クラス + 3D/PS/1D 具象)。Fortran 側は `fortran/`(ビルド手順は `fortran/README.md`)。`@verified_against_fortran` / `FortranBinding` は `core/material/fortran_binding.py`、ランタイム比較 `check_bindings` は `verification/fortran_check.py`。非公開モデルは別リポジトリ(`manforge-internal`)に置き、`MODELS` dict で service 層に登録する。
