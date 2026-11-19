# 変更履歴

このファイルは、利用者から見える変更を版ごとにまとめる。設計の経緯と理由は
[`docs/design/api-redesign.md`](docs/design/api-redesign.md) にある。

## 0.2.0(未リリース)

API を全面的に見直した。**0.1.0 との互換性は無い**(0.x の間は破壊的変更を行う。互換シムは置かない)。
既存モデルの計算結果は変わらない — 全参照モデルの応力・状態・接線の履歴が、0.1.0 と丸め誤差の範囲で一致する。

以下は 0.1.0 から移行するときの対応表である。

### モデルの定義

**フックの署名。** 予測子以外のフックは、1 ステップの入力をまとめた `Increment`(`inc.state_n` /
`inc.strain_inc` / `inc.trial`)を受け取る。引数リストが違うモデルは、クラス定義時に `TypeError` になる。

| 0.1.0 | 0.2.0 |
|---|---|
| `elastic_stiffness(state=None)` | `elastic_stiffness(state)`(状態は必須) |
| `elastic_check(state_n, *, stress_trial, strain_inc=None)` | `elastic_check(inc)` |
| `elastic_update_state(state_n, *, stress_trial, strain_inc=None)` | `elastic_update_state(inc)` |
| `update_state(dlambda, state_new, state_n, *, stress_trial=None, strain_inc=None)` | `update_state(new, dlambda, inc)` |
| `state_residual(state_new, dlambda, state_n, *, stress_trial, strain_inc=None)` | `state_residual(new, dlambda, inc)` |
| `user_defined_return_mapping(stress_trial, C, state_n)` | `user_defined_return_mapping(inc)` |
| `user_defined_tangent(stress, state, dlambda, C, state_n, *, stress_trial=None, strain_inc=None)` | `user_defined_tangent(new, dlambda, inc)` |
| — | `elastic_tangent(inc)`(予測子を override したときの手書きの弾性接線) |
| λ、μ から剛性を手で組む | `self.isotropic_stiffness()`(`E` と `nu` から) |

- `stress_trial` は `inc.trial["stress"]`、ステップ開始時の状態は `inc.state_n`、Δε は `inc.strain_inc`。
- `default_stress_residual(new, dlambda, stress_trial)` は σ_trial を必ず受け取る(`None` は不可)。

**コンストラクタ。** モデルは `__init__` を書かない。`param_names` が唯一の宣言で、基底クラスが同名の
キーワード引数を受け取る。

| 0.1.0 | 0.2.0 |
|---|---|
| `def __init__(self, dimension=SOLID_3D, *, E, nu, ...)` と `self.E = E` の並び | 削除する(`param_names` だけ書く) |
| パラメータの既定値(`p_D=0.0`、`h=0.2` など) | 無し。全パラメータ必須 |
| `Model(PLANE_STRAIN, E=...)`(位置引数の次元) | `Model(dimension=PLANE_STRAIN, E=...)` |
| 次元別クラスの `__init__` | クラス本体に `dimension = SOLID_3D` と書く。固定したクラスに別の次元は渡せない |
| 定数を `__init__` で `self.ae = 0.5` | クラス属性 `ae = 0.5` |
| パラメータ以外の設定 | `__init__(self, *, c=..., **params)` で受け取り、`super().__init__(**params)` |

**手書きのリターンマッピング。** `user_defined_return_mapping` の結果は、自動微分経路と同じ条件で確かめられる。
Δλ < 0 や非有限の値は、`converged=True` と報告されていても非収束として扱われる。

### シミュレーション

| 0.1.0 | 0.2.0 |
|---|---|
| `driver.run(load, collect_state={"ep": FieldType.STRAIN})`、`result.fields["ep"].data` | `driver.run(load)`、`result.history("ep")` |
| `driver.iter_run(..., raise_on_nonconverged=...)` | 廃止。スイッチは integrator の `raise_on_nonconverged` だけ |
| 非収束のステップが結果の配列の最後に入る | 入らない。`result.converged` / `result.failed_step` で知らせる |
| `integrator.elastic_stiffness(...)` | `integrator.elastic_tangent(stress_n, state_n)` |
| `result._state_n` | `result.increment.state_n`(`result.stress_trial` はそのまま使える) |
| `StressDriver` に `(N,)` や `(N, 1)` の履歴(全成分に同じ値が入って走っていた) | `ValueError`。`(N, ntens)` で渡す |

- `StressIntegrator` は `typing.Protocol` になった。自作の integrator は `dimension` / `ntens` /
  `raise_on_nonconverged` / `initial_state` / `increment` / `elastic_tangent` / `stress_update` を持つ。
- `nr_jacobian` の既定値は全層で `"exact"`。**`service.run_simulation` の既定は `"broyden"` から変わった**
  (速度を優先するなら明示する)。

### フィッティング

| 0.1.0 | 0.2.0 |
|---|---|
| `fit_params(model, driver_factory, exp_data, fit_config, fixed_params=...)` | `fit_params(model, targets, space, *, driver_factory=...)` |
| `exp_data={"strain": ..., "stress": ...}` | `[FitTarget.stress_strain(strain, stress)]` |
| `fit_config={...}` と `fixed_params={...}` | `ParameterSpace(free={...}, fixed={...})` |
| `build_objective(model, driver_factory, targets, space, failures)` | `build_objective(model, targets, space, *, driver_factory=..., failures=...)` |

- 各評価のモデルは `model.with_params(**params)` で作られる。次元と、パラメータ以外の設定が引き継がれる
  (0.1.0 では設定が既定値に戻っていた)。

### 検証

| 0.1.0 | 0.2.0 |
|---|---|
| `JacobianChecker(model).compute(result, state_n, strain_inc=...)` | `JacobianChecker(model).compute(result)`(結果が自分の増分を持つ) |
| `JacobianChecker(model).compare(result_a, result_b, state_n, ...)` | `JacobianChecker(model).compare(result_a, result_b)` |
| テスト内の「メソッド ↔ ブロック」の対応表 | モデル上で `@hand_block(row, col)`、`JacobianChecker(model).check_hand_blocks(results)` |
| `CrosscheckStrainDriver(a, b).run(load)` / `CrosscheckStressDriver` | `crosscheck(a, b, load)`(driver は載荷履歴から決まる。それ以外は `driver_factory=`) |
| `cc.iter_run(load)` で最初の失敗を探す | `crosscheck(..., stop_on_failure=True)`、`result.first_failure` |
| `result.n_cases` / `result.cases` | `result.n_steps` / `result.steps` |
| `JacobianChecker(model, rtol=...)`、`check_bindings(..., rtol=...)` | `tol=` |
| `check_tangent(...)` | `TangentChecker(integrator).check(...)` |

追加:

- `JacobianChecker(model).check_solution(results)` — 手書きのリターンマッピングの解を、自動微分側の残差に代入して行ごとに検査する。
- `crosscheck(..., lockstep=True)` — 各ステップで候補側に参照側の入力を渡して 1 ステップずつ比べる。
- `FortranIntegrator(parse_umat_failure=...)` — UMAT 自身の収束フラグを読む。
- `integrator.jacobian(result)` — 自動微分の残差ヤコビアン。

`crosscheck` と `JacobianChecker.compare` の相対誤差は、成分ごとではなく配列全体で測るようになった。

### import パス

公開の import 元は 7 つのパッケージで、どの名前も 1 つのパッケージにだけ属する。
サブモジュールのパス(`manforge.core.state` など)は動くが、文書化されたインターフェースではない。

| 0.1.0 | 0.2.0 |
|---|---|
| `from manforge.core.material import MaterialModel, ParamMeta` | `from manforge.core import MaterialModel, ParamMeta` |
| `from manforge.core.state import Implicit, Explicit, NTENS, SCALAR` | `from manforge.core import ...` |
| `from manforge.core.dimension import SOLID_3D, ...` | `from manforge.core import ...` |
| `from manforge.simulation.driver import StrainDriver` ほか | `from manforge.simulation import ...` |
| `from manforge.verification import FortranModule, JacobianBlocks` | `from manforge.simulation import FortranModule, JacobianBlocks` |
| `from manforge.verification import verified_against_fortran` | `from manforge.core import verified_against_fortran` |
| `from manforge.service import ParamMeta` | `from manforge.core import ParamMeta` |
| `from manforge.utils.smooth import smooth_sqrt` ほか | `from manforge.utils import ...` |

削除: `manforge.autodiff`、`manforge.utils.tensor`、`manforge.utils.voigt`、`StressDimension.mandel_factors_jnp` /
`identity_jnp`、`Comparator` / `CaseResult` / `ComparisonResult` / `CrosscheckCaseResult`。

### ドキュメント

- ガイド(日本語)を `docs/` に、API リファレンス(英語)を docstring から生成する形にした。
  `make docs` で HTML を作る(`uv sync --extra docs` が必要)。
- README は概要・インストール・クイックスタートと、`docs/` への案内だけを持つ。

## 0.1.0

最初のタグ。
