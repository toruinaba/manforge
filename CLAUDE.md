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

# Documentation (Sphinx + MyST; needs `uv sync --extra docs`)
make docs                    # HTML → docs/_build/html。警告はエラー扱い
```

There is no linter configured. No CI/CD workflows exist.

## Documentation

- **ガイドは `docs/` の Markdown(日本語)、API リファレンスは docstring(英語)から生成する。** README は概要・インストール・クイックスタートと `docs/` への案内だけを持つ(`docs/index.md` が README のその部分を include するので、二重には書かない)。使い方の説明を足すときは README ではなく `docs/` の該当ページに書く。
- **`make docs` が通ることを確認する。** 警告はエラー扱いで、次を検出する: 解決できない相互参照、`docs/` 内のリンク切れ、公開パッケージの `__all__` に載っているのにリファレンスに項目が無い名前(docstring の無いクラス・関数、`docs/reference/*.md` に説明の無い定数)。
- **docstring の相互参照は公開パッケージのパスで書く**(`` :class:`~manforge.core.MaterialModel` ``。`manforge.core.material.MaterialModel` ではない)。非公開の名前は参照にせず ``` ``literal`` ``` で書く。

## Architecture

manforge is a framework for validating Fortran UMAT (Abaqus user material) constitutive models. Users define a material model in Python; the framework handles return mapping, consistent tangent computation (via autograd), parameter fitting, and cross-validation against compiled Fortran UMAT subroutines. Arithmetic is float64 throughout — autograd wraps NumPy, so there is nothing to enable.

### Public import paths

公開の import 元はパッケージ単位の 7 つ: `manforge.core`(モデル定義に要るもの全部 — `MaterialModel` / 状態宣言 / 次元 / `hand_block` / `verified_against_fortran` / 結果型)、`manforge.models`、`manforge.simulation`(integrator / driver / `FieldHistory` / `FortranModule` / `JacobianBlocks`)、`manforge.fitting`、`manforge.verification`(検査のみ)、`manforge.service`、`manforge.utils`。どの名前も 1 つのパッケージの `__all__` にだけ載せる(例外なし。`tests/unit/test_public_api.py` が確認する)。README・例・テスト・docstring の例はこの形で書く。サブモジュールのパス(`manforge.core.state` 等)は実装の置き場所で、`src/` 内部の import と、非公開ヘルパを直接試すテストだけが使う。

### Three-layer design

1. **Material model** — `core/material/base.py`: `MaterialModel` ABC。状態変数は `Implicit` / `Explicit` のクラス属性で宣言し、`__init_subclass__` が `state_names` 等を導出・検証する。ユーザーが書くのは `yield_function`(必須)+ `update_state`(explicit 状態)/ `state_residual`(implicit 状態)。パラメータは `param_names` が唯一の宣言で、基底の `__init__` が同名のキーワード引数を受け取って属性に入れる(モデルは `__init__` を書かない。全パラメータ必須で既定値なし — UMAT の `PROPS` と同じ規則)。次元は、クラス本体に `dimension = X` と書いたクラスとその子は固定(別の `dimension=` は `TypeError`)、書かないクラスは構築時の `dimension=` で選ぶ。特殊フック(`elastic_predictor` / `elastic_tangent` / `elastic_check` / `elastic_update_state` / `user_defined_return_mapping` / `user_defined_tangent` / `flow` override / `dlambda` 行差し替え)の契約と設計理由はすべて base.py の docstring。予測子以外のフックは 1 ステップの入力をまとめた `Increment`(`inc.state_n` / `inc.strain_inc` / `inc.trial`、`core/state.py`)を受け取る。フックは位置引数で呼ばれるので、引数リストが違うとクラス定義時に `TypeError`。
2. **Solver** — `simulation/integrator/`: `stress_update` = UMAT 1回分(弾性予測 → 分岐 → return mapping → consistent tangent)。結果(`StressUpdateResult`)は自分の `Increment` を持つので、ヤコビアンの内省は `JacobianChecker(model).compute(result)` だけで済む。NR 経路(スカラー/ベクトル)は宣言から自動選択。`nr_jacobian="broyden"` は初回のみ厳密ヤコビアン + Broyden 更新(接線は常に厳密)。既定は全層(integrator / fitting / service)で `"exact"`。NR 未知数ベクトルのレイアウトは `simulation/_layout.py`(`ResidualLayout`)、統一残差は `simulation/_residual.py` が単一の真実。
3. **Application** — `simulation/driver.py`(`StrainDriver` / `StressDriver` / `MixedDriver`、`run` / `iter_run`。状態量の履歴は要求せず `DriverResult.history(name)` で取る)、`fitting/`(`fit_params(model, targets, space, *, driver_factory=...)` の 1 形式 + `ParameterSpace` / `FitTarget` / `Observation` / metrics。各評価のモデルは `model.with_params(...)`)、`verification/`(`JacobianChecker`(`check_hand_blocks` / `check_solution`)/ `TangentChecker` / `crosscheck` / `check_bindings`)、`service/`(WebAPI 向け契約層: `ModelRegistry` / `run_simulation`。Web 依存ゼロ、plain I/O、非収束は値で返す)。

### Voigt convention(横断契約 — 全層が従う)

3D ソリッドは `[11, 22, 33, 12, 13, 23]`(ABAQUS UMAT 規約)。**量の種類で剪断の扱いが異なる 2 層契約**:

- **応力系**(σ, s, 背応力 α など): **physical shear** — テンソル成分そのもの
- **ひずみ系**(ε, ε_p, Δε): **engineering shear**(γ12 = 2ε12)— ABAQUS `DSTRAN`・弾性剛性 `σ = C:ε`・driver 入力すべて

ノルムや等価量の計算では Mandel スケーリング(剪断 ×√2)を内部適用。Mandel 係数と次元ごとの演算子(`dev` / `vonmises` / `inner_product` / `strain_norm` 等)は `core/dimension.py` の `StressDimension` に集約され `MaterialModel` がデリゲートする。

### FlowVector(横断契約)

塑性流れ方向は同一方向に 2 つの Voigt 表現(剪断 ×2 差)があり、取り違えは単軸/二軸で消える silent error になる。そのため `model.flow(state)` は生 ndarray でなく `FlowVector` を返し、使用側が `.strain_like`(engineering shear — `C @ n`、Δε_p)か `.stress_like`(physical shear — 背応力発展則)を名指しする。生配列演算は `TypeError`。`flow` を override するときは `self.strain_flow(...)` / `self.stress_flow(...)` でタグ付けして返す(モデル内で ×2/÷2 の変換算術を書かない)。詳細は `core/state.py`。

### 非収束の扱い(スイッチは 1 つ、層ごとに受け止め方が違う)

スイッチは integrator の `raise_on_nonconverged` だけ。driver は自分の外側 NR の失敗にも同じ設定を使い、独自のスイッチを持たない。integrator の契約(`StressIntegrator` Protocol: `dimension` / `ntens` / `raise_on_nonconverged` / `initial_state` / `elastic_tangent` / `stress_update`)は `simulation/integrator/base.py`。

- **Integrator**: 既定 raise。`raise_on_nonconverged=False` で `converged=False` を返す。Δλ<0 は KKT 違反なので非収束扱い。許容値はスケール不変(`tol × max(1, ‖σ_trial‖_∞)`)。
- **Driver**: `iter_run` は非収束 step を yield して停止(非収束イテレートで状態を進めない)。`run` は完了した step だけを返し、`DriverResult.converged` / `failed_step` で知らせる(失敗した step は配列に入れない)。
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

参照実装は `src/manforge/models/`(J2 / J2 損傷 / AF / OW / YU — 各ファミリ親クラス + 3D/PS/1D 具象)。Fortran 側は `fortran/`(ビルド手順は `fortran/README.md`)。`@verified_against_fortran` / `FortranBinding` は `core/material/fortran_binding.py`、ランタイム比較 `check_bindings` は `verification/fortran_check.py`。手導出したヤコビアンのブロックは `@hand_block(row, col)`(`core/material/hand_block.py`)でモデル上に宣言し、`JacobianChecker.check_hand_blocks` が AD の残差ヤコビアン(`simulation/jacobian.py`)とブロックごとに照合する。手書きのリターンマッピング(`user_defined_return_mapping`)の解は `JacobianChecker.check_solution` が AD 側の残差に代入して行ごとに検査する(残差そのものは宣言しない — 比べるのは根だけ)。2 つの実装を同じ載荷履歴で進めて比べるのは `crosscheck(integrator_a, integrator_b, load, *, driver_factory=...)`(`verification/crosscheck.py`)。UMAT 自身の収束フラグは `FortranIntegrator(parse_umat_failure=...)` で読ませる。非公開モデルは別リポジトリ(`manforge-internal`)に置き、`MODELS` dict で service 層に登録する。
