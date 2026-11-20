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
uv run pytest src/returnmap/models/j2_isotropic -m "not fortran" -v  # One model family

# Fortran compilation (requires gfortran; this host has none — use Docker)
uv run returnmap build fortran/abaqus_stubs.f90 src/returnmap/models/j2_isotropic/fortran/j2_isotropic_3d.f90 --name j2_isotropic_3d
uv run returnmap list / clean
make docker-build && make docker-test

# Documentation (Sphinx + MyST; needs `uv sync --extra docs`)
make docs                    # HTML → docs/_build/html。警告はエラー扱い
```

There is no linter configured. No CI/CD workflows exist.

## Documentation

- **ガイドは `docs/` の Markdown(日本語)、API リファレンスは docstring(英語)から生成する。** README は概要・インストール・クイックスタートと `docs/` への案内だけを持つ(`docs/index.md` が README のその部分を include するので、二重には書かない)。使い方の説明を足すときは README ではなく `docs/` の該当ページに書く。
- **`make docs` が通ることを確認する。** 警告はエラー扱いで、次を検出する: 解決できない相互参照、`docs/` 内のリンク切れ、公開パッケージの `__all__` に載っているのにリファレンスに項目が無い名前(docstring の無いクラス・関数、`docs/reference/*.md` に説明の無い定数)。
- **docstring の相互参照は公開パッケージのパスで書く**(`` :class:`~returnmap.core.MaterialModel` ``。`returnmap.core.material.MaterialModel` ではない)。非公開の名前は参照にせず ``` ``literal`` ``` で書く。

## Architecture

returnmap is a framework for validating Fortran UMAT (Abaqus user material) constitutive models. Users define a material model in Python; the framework handles return mapping, consistent tangent computation (via autograd), parameter fitting, and cross-validation against compiled Fortran UMAT subroutines. Arithmetic is float64 throughout — autograd wraps NumPy, so there is nothing to enable.

### Public import paths

公開の import 元はパッケージ単位の 7 つ: `returnmap.core`(モデル定義に要るもの全部 — `MaterialModel` / 状態宣言 / 次元 / `hand_block` / `verified_against_fortran` / 結果型)、`returnmap.models`、`returnmap.simulation`(integrator / driver / `FieldHistory` / `FortranModule` / `JacobianBlocks`)、`returnmap.fitting`、`returnmap.verification`(検査のみ)、`returnmap.service`、`returnmap.utils`。どの名前も 1 つのパッケージの `__all__` にだけ載せる(例外なし。`tests/unit/test_public_api.py` が確認する)。README・例・テスト・docstring の例はこの形で書く。サブモジュールのパス(`returnmap.core.state` 等)は実装の置き場所で、`src/` 内部の import と、非公開ヘルパを直接試すテストだけが使う。

### Three-layer design

1. **Material model** — `core/material/base.py`: `MaterialModel` ABC。状態変数は `Implicit` / `Explicit` のクラス属性で宣言し、`__init_subclass__` が `state_names` 等を導出・検証する。ユーザーが書くのは `yield_function`(必須)+ `update_state`(explicit 状態)/ `state_residual`(implicit 状態)。パラメータは `param_names` が唯一の宣言で、基底の `__init__` が同名のキーワード引数を受け取って属性に入れる(モデルは `__init__` を書かない。全パラメータ必須で既定値なし — UMAT の `PROPS` と同じ規則)。次元は、クラス本体に `dimension = X` と書いたクラスとその子は固定(別の `dimension=` は `TypeError`)、書かないクラスは構築時の `dimension=` で選ぶ。特殊フック(`elastic_predictor` / `elastic_tangent` / `elastic_check` / `elastic_update_state` / `user_defined_return_mapping` / `user_defined_tangent` / `flow` override / `dlambda` 行差し替え)の契約と設計理由はすべて base.py の docstring。予測子以外のフックは 1 ステップの入力をまとめた `Increment`(`inc.state_n` / `inc.strain_inc` / `inc.trial`、`core/state.py`)を受け取る。フックは位置引数で呼ばれるので、引数リストが違うとクラス定義時に `TypeError`。
2. **Solver** — `simulation/integrator/`: `stress_update` = UMAT 1回分(弾性予測 → 分岐 → return mapping → consistent tangent)。結果(`StressUpdateResult`)は自分の `Increment` を持つので、ヤコビアンの内省は `integrator.jacobian(result)` だけで済む。NR の未知数は常に `[σ, Δλ, implicit 状態]`(explicit 状態は `update_state` で求め、未知数に入らない。σ を `Implicit` と宣言するのは σ 行の残差を自分で書くときで、宣言しなければ既定の関連流れの残差が使われる)。`nr_jacobian="broyden"` は初回のみ厳密ヤコビアン + Broyden 更新(接線は常に厳密)。既定は全層(integrator / fitting / service)で `"exact"`。NR 未知数ベクトルのレイアウトは `simulation/_layout.py`(`ResidualLayout`)、統一残差は `simulation/_residual.py` が単一の真実。
3. **Application** — `simulation/driver.py`(`StrainDriver` / `StressDriver` / `MixedDriver`、`run` / `iter_run`。状態量の履歴は要求せず `DriverResult.history(name)` で取る)、`simulation/compare.py`(`compare_models(models, load, *, targets=...)`: 複数モデルの応答・計算コスト・フィットの良さを並べる。計算の結果なので `*Result`、非収束は行として返す)、`fitting/`(`fit_params(model, targets, space, *, driver_factory=...)` の 1 形式 + `ParameterSpace` / `FitTarget` / `Observation` / metrics。各評価のモデルは `model.with_params(...)`、結果のモデルは `FitResult.model`)、`verification/`(検査はすべて「関数が報告を返す」形: `check_hand_blocks` / `check_solution` / `check_tangent` / `crosscheck` / `compare_jacobians` / `check_bindings`)、`service/`(WebAPI 向け契約層: `ModelRegistry` / `run_simulation`。レジストリに登録するのも `run_simulation(model, params, load)` が受け取るのもテンプレートのモデルで、`model.with_params(**params)` で走る。Web 依存ゼロ、plain I/O、非収束は値で返す)。

### Voigt convention(横断契約 — 全層が従う)

3D ソリッドは `[11, 22, 33, 12, 13, 23]`(ABAQUS UMAT 規約)。**量の種類で剪断の扱いが異なる 2 層契約**:

- **応力系**(σ, s, 背応力 α など): **physical shear** — テンソル成分そのもの
- **ひずみ系**(ε, ε_p, Δε): **engineering shear**(γ12 = 2ε12)— ABAQUS `DSTRAN`・弾性剛性 `σ = C:ε`・driver 入力すべて

ノルムや等価量の計算では Mandel スケーリング(剪断 ×√2)を内部適用。Mandel 係数と次元ごとの演算子(`dev` / `vonmises` / `inner_product` / `strain_norm` 等)は `core/dimension.py` の `StressDimension` に集約され `MaterialModel` がデリゲートする。

### FlowVector(横断契約)

塑性流れ方向は同一方向に 2 つの Voigt 表現(剪断 ×2 差)があり、取り違えは単軸/二軸で消える silent error になる。そのため `model.flow(state)` は生 ndarray でなく `FlowVector` を返し、使用側が `.strain_like`(engineering shear — `C @ n`、Δε_p)か `.stress_like`(physical shear — 背応力発展則)を名指しする。生配列演算は `TypeError`。`flow` を override するときは `self.strain_flow(...)` / `self.stress_flow(...)` でタグ付けして返す(モデル内で ×2/÷2 の変換算術を書かない)。詳細は `core/state.py`。

### 非収束の扱い(スイッチは 1 つ、層ごとに受け止め方が違う)

スイッチは integrator の `raise_on_nonconverged` だけ。driver は自分の外側 NR の失敗にも同じ設定を使い、独自のスイッチを持たない。integrator の契約(`StressIntegrator` Protocol: `dimension` / `ntens` / `raise_on_nonconverged` / `initial_state` / `increment` / `elastic_tangent` / `stress_update`)は `simulation/integrator/base.py`。

- **Integrator**: 既定 raise。`raise_on_nonconverged=False` で `converged=False` を返す。Δλ<0 は KKT 違反なので非収束扱い(手書き経路の結果も同じ条件で確かめる: Δλ<0 と非有限の値は、`converged=True` と報告されていても非収束)。許容値はスケール不変(`tol × max(1, ‖σ_trial‖_∞)`)。
- **Driver**: `iter_run` は非収束 step を yield して停止(非収束イテレートで状態を進めない)。`run` は完了した step だけを返し、`DriverResult.converged` / `failed_step` で知らせる(失敗した step は配列に入れない)。
- **fitting**: 評価の非収束は有限ペナルティ(`FAILURE_PENALTY`)でスコアし続行。全評価失敗時のみ raise。設計理由は `fitting/objective.py`。
- **service**: `run_simulation` は部分履歴 + `converged` / `failed_step` を値で返す(呼び出し側バグのみ `ValueError`)。
- **compare_models**: 解けなかったモデルは `ModelRun.converged=False` / `failed_step` の行として返し、ほかのモデルの比較を続ける(integrator が raise した場合も同じ。呼び出し側バグのみ例外)。

### 検査の形と誤差の測り方(横断契約 — `verification/` の全検査が従う)

- **検査は関数で、報告(`*Report`)を返す。** 報告は `.passed` と `.summary()` を持ち、テストはどの検査でも `assert report.passed, report.summary()`。検査用のクラス(`XxxChecker`)は作らない。計算の結果は `*Result`、検査の結果は `*Report`(`tests/unit/test_public_api.py` が確認する)。
- **誤差は `max|候補 − 参照| / max|参照|`、合格は `<= tol`。** 配列(ブロック・接線・応力ベクトル)全体で測り、成分ごとに割らない。分母に単位を持つ定数(`+ 1e-2`、`max(·, 1)` など)を置かない — モデルを書いた単位で合否が変わるため。実装は `verification/_measure.py` の 1 箇所で、検査はそれを使う。NaN は最悪の誤差として扱う。割る相手が無い量(厳密に 0 のヤコビアンのブロック、0 から動いていない状態量)は、問題の大きさから決まる尺度より 12 桁以上小さいことで見分け(`NOISE_FLOOR`)、その尺度に対して測る — 素の数値とは比べない。
- **driver を省略できる入口は同じ規則で決める**(`fit_params`、`crosscheck`、`compare_models`): ひずみ履歴 → `StrainDriver`、応力履歴 → `StressDriver`(`simulation/driver.py` の `default_driver_for`)。

### Test layout

Tests live in two places. **Framework tests** are in `tests/`: `tests/{unit,integration}/` follow the module layout of `src/returnmap/` — a module's tests live in `test_<module>.py` at the mirrored path, under `unit/` or `integration/` (most modules have one of the two, not both). A few test files have no single source module (`test_public_api.py`, `test_flow.py`), and tests that span several model families stay here too (`tests/integration/models/test_kinematic_common.py`, `test_reference_models.py`). `tests/e2e/` covers CLI subprocess tests and the fitting pipeline. **Model tests** live with the model, in `src/returnmap/models/<family>/tests/`, and are shipped in the wheel (`pytest --pyargs returnmap.models.<family>`). `@pytest.mark.fortran` auto-skips when the compiled `.so` is missing; `@pytest.mark.slow` is opt-in for >1s cases. The repository-root `conftest.py` puts `fortran/` (where the f2py modules are built) on `sys.path` for both.

| directory | purpose |
|---|---|
| `tests/unit/` | Tests of one module's own logic. Most need no integrator or driver; `unit/fitting/` does run them (an objective cannot be evaluated otherwise) |
| `tests/integration/` | Orchestration tests — integrator + driver in the loop |
| `tests/e2e/` | CLI subprocess + fitting pipeline smoke |
| `src/returnmap/models/<family>/tests/` | Everything about that model: its own logic, its response, hand-written path vs autodiff (`test_analytical_vs_numerical.py`), Python vs UMAT (`test_numerical_vs_fortran.py`) |

### Models / Fortran

参照実装は `src/returnmap/models/`(J2 / J2 損傷 / AF / OW / YU — 各ファミリ親クラス + 3D/PS/1D 具象)。**1 ファミリ 1 ディレクトリ**(`models/<family>/`)で、モデル本体(`model.py` など)、UMAT のソース(`fortran/`、任意)、テスト(`tests/`、任意)をまとめて持つ。ファミリのテストは公開の import パスとそのファミリの中のものだけを import する(ディレクトリごと動かせるようにするため。非公開リポジトリから本体への格上げも移動で済む)。公開名は `returnmap.models` から import する形のまま。参照モデルは全パラメータの `param_meta`(説明・単位・目安の値・範囲)を持ち、目安の値で計算が通る(`tests/integration/models/test_reference_models.py` が確認する。service の入力フォーム、チュートリアル、marimo ノートブックのスライダーがこれを読む)。`fortran/` は全 UMAT が共有するもの(`abaqus_stubs.f90`)と f2py モジュールのビルド先(ビルド手順は `fortran/README.md`)。`@verified_against_fortran` / `FortranBinding` は `core/material/fortran_binding.py`、ランタイム比較 `check_bindings` は `verification/fortran_check.py`。手導出したヤコビアンのブロックは `@hand_block(row, col)`(`core/material/hand_block.py`)でモデル上に宣言し、`check_hand_blocks(model, results)` が AD の残差ヤコビアン(`simulation/jacobian.py`)とブロックごとに照合する。手書きのリターンマッピング(`user_defined_return_mapping`)の解は `check_solution(model, results)` が AD 側の残差に代入して行ごとに検査する(残差そのものは宣言しない — 比べるのは根だけ)。2 つの実装を同じ載荷履歴で進めて比べるのは `crosscheck(integrator_a, integrator_b, load, *, driver_factory=...)`(`verification/crosscheck.py`)。UMAT 自身の収束フラグは `FortranIntegrator(parse_umat_failure=...)` で読ませる。非公開モデルは別リポジトリ(`returnmap-internal`)に置き、`MODELS` dict で service 層に登録する。
