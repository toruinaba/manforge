# 変更履歴

このファイルは、利用者から見える変更を版ごとにまとめる。設計の経緯と理由は
[`docs/design/api-redesign.md`](docs/design/api-redesign.md) にある。

## 0.2.0(未リリース)

API を全面的に見直した。**0.1.0 との互換性は無い**(0.x の間は破壊的変更を行う。互換シムは置かない)。
既存モデルの計算結果は変わらない — 全参照モデルの応力・状態・接線の履歴が、0.1.0 と丸め誤差の範囲で一致する。

以下は 0.1.0 から移行するときの対応表である。

### パッケージ名

**`manforge` から `returnmap` に改名した。** 0.1.0 の名前は、何をする道具かが読み取れなかった。モデルを書いて比べる
ときも、手書きのアルゴリズムを検査するときも、UMAT と突き合わせるときも、中身はリターンマッピングを解いて結果を見る
ことなので、それを名前にした。機能は変わらない。

| 0.1.0 | 0.2.0 |
|---|---|
| `import manforge`、`from manforge.core import ...` | `import returnmap`、`from returnmap.core import ...` |
| `uv run manforge build ...`(CLI) | `uv run returnmap build ...` |
| 配布名 `manforge` | `returnmap` |
| Docker イメージ `manforge-fortran` | `returnmap-fortran`(`make docker-build` で作り直す) |

下の表の左の列は 0.1.0 の書き方なので、`manforge` のままにしてある。

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

`ReturnMappingResult` は全フィールドが必須になった。`converged`(0.1.0 の既定は `True`)、`n_iterations`、
`residual_history` を省略すると `TypeError`。閉形式解は `n_iterations=1, residual_history=[f_trial, 0.0],
converged=True` と書く。

### シミュレーション

| 0.1.0 | 0.2.0 |
|---|---|
| `driver.run(load, collect_state={"ep": FieldType.STRAIN})`、`result.fields["ep"].data` | `driver.run(load)`、`result.history("ep")` |
| `driver.iter_run(..., raise_on_nonconverged=...)` | 廃止。スイッチは integrator の `raise_on_nonconverged` だけ |
| 非収束のステップが結果の配列の最後に入る | 入らない。`result.converged` / `result.failed_step` で知らせる |
| `integrator.elastic_stiffness(...)` | `integrator.elastic_tangent(stress_n, state_n)` |
| `result._state_n` | `result.increment.state_n`(`result.stress_trial` はそのまま使える) |
| `StressDriver` に `(N,)` や `(N, 1)` の履歴(全成分に同じ値が入って走っていた) | `ValueError`。`(N, ntens)` で渡す |
| `FieldHistory(FieldType.STRAIN, "Strain", data)` | `FieldHistory(FieldType.STRAIN, data)`(ラベルは省略できる。付けるなら `name="..."`) |
| `FieldHistory.cyclic_strain(...).name == "Strain"` | `""`(付けるなら `name="..."`) |
| `step.i`(`iter_run` が返す `DriverStep`) | `step.index`(`CrosscheckStep.index` と同じ名前) |
| `SOLID_3D.name == "3D_SOLID"` | `"SOLID_3D"`(import する名前と同じ。エラーメッセージに出る) |

- `StressIntegrator` は `typing.Protocol` になった。自作の integrator は `dimension` / `ntens` /
  `raise_on_nonconverged` / `initial_state` / `increment` / `elastic_tangent` / `stress_update` を持つ。
- `nr_jacobian` の既定値は全層で `"exact"`。**`service.run_simulation` の既定は `"broyden"` から変わった**
  (速度を優先するなら明示する)。
- `result.history("strain")` が使える(`result.strain` と同じ。`"stress"` と同様)。そのため、状態量に `strain` という
  名前は付けられない(クラス定義時に `TypeError`)。

追加:

- `compare_models(models, load)`(`returnmap.simulation`)— 複数のモデルを同じ載荷履歴で走らせ、応答(各モデルの
  `DriverResult`)、計算コスト(塑性ステップあたりの NR 反復数、ステップあたりの時間)、`targets=` を渡せば
  フィットの良さを並べる。途中で解けなくなったモデルは例外にならず、表の 1 行になる。モデルの代わりに integrator を
  渡せば、解き方(自動微分と手書き、`"exact"` と `"broyden"`、Python と UMAT)を比べられる。
  ガイドは「モデルを比べる」。

### フィッティング

| 0.1.0 | 0.2.0 |
|---|---|
| `fit_params(model, driver_factory, exp_data, fit_config, fixed_params=...)` | `fit_params(model, targets, space, *, driver_factory=...)` |
| `exp_data={"strain": ..., "stress": ...}` | `[FitTarget.stress_strain(strain, stress)]` |
| `fit_config={...}` と `fixed_params={...}` | `ParameterSpace(free={...}, fixed={...})` |
| `build_objective(model, driver_factory, targets, space, failures)` | `build_objective(model, targets, space, *, driver_factory=..., failures=...)` |
| `Model(**{k: result.params[k] for k in Model.param_names})` | `result.model`(パラメータだけなら `result.model_params`) |
| `result.n_iter` | `result.n_evaluations`(中身は以前から目的関数の評価回数) |
| `Observation(field="Stress")`、`x_field="Strain"` | `field="stress"`、`x_field="strain"`(`result.history(...)` が受け取る名前と同じ) |

- `driver_factory` は省略できる。省略すると載荷履歴から決まる(ひずみ履歴 → `StrainDriver`、応力履歴 → `StressDriver`。
  `crosscheck` と同じ規則)。
- 各評価のモデルは `model.with_params(**params)` で作られる。次元と、パラメータ以外の設定が引き継がれる
  (0.1.0 では設定が既定値に戻っていた)。
- `FitResult.params` は探索空間のすべての名前を持つ(比で探索するときの補助変数も含む)。モデルのパラメータは
  `FitResult.model_params`、それを入れたモデルは `FitResult.model`。

修正:

- **`metric="pcm"` が、ノイズのある実測データで誤ったパラメータを良いと評価していた。** 2 本のカーブを弧長で対応づける
  ときに、ノイズで伸びた折れ線の長さをそのまま使っていたためで、点数の多い実測データほど大きくずれた(応力範囲の
  0.4 % のノイズがある 2000 点の繰返し試験で、降伏応力を真値の 1.8 倍にしたモデルのほうが真値より良い値になり、
  同定結果が上限に張り付いた)。実測カーブの弧長を、ノイズの大きさに合わせた物差しで測るようにした。ノイズの無い
  カーブの値は変わらない。

### サービス

レジストリに登録するものと `run_simulation` が受け取るものは、モデルのクラスではなく**テンプレートのモデル**
(インスタンス)になった。クラスだけでは、次元を固定していないクラスの次元と、パラメータ以外の設定を選べなかった。

| 0.1.0 | 0.2.0 |
|---|---|
| `reg.register("j2", J2Isotropic1D)` | `reg.register("j2", J2Isotropic1D(E=..., nu=..., ...))` |
| `reg.get(name)` がクラスを返す | テンプレートのモデルを返す |
| `run_simulation(model_cls, params, load)` | `run_simulation(model, params, load)`(`model.with_params(**params)` で走る) |
| `ModelDescriptor.ntens` は登録したクラスの既定の次元のもの | 登録したモデルの次元のもの |
| 非公開パッケージの `MODELS: dict[str, type[MaterialModel]]` | `MODELS: dict[str, MaterialModel]` |

- `params` は従来どおり `param_names` の全部を過不足なく持つ(過不足は `ValueError`)。テンプレートが持つ
  パラメータの値は使われない。
- `default_registry()` のテンプレートは、各クラスの `param_meta` の `default` を持つ。

### 検証

検査はすべて関数で、報告オブジェクトを返す。報告は `.passed` と `.summary()` を持つので、どの検査も
`assert report.passed, report.summary()` と書ける。`JacobianChecker` と `TangentChecker` は無くなった。

| 0.1.0 | 0.2.0 |
|---|---|
| `JacobianChecker(model).compute(result, state_n, strain_inc=...)` | `integrator.jacobian(result)`(結果が自分の増分を持つ) |
| `JacobianChecker(model, rtol=...).compare(result_a, result_b, state_n, ...)` | `compare_jacobians(model, result_a, result_b, tol=...)` |
| テスト内の「メソッド ↔ ブロック」の対応表 | モデル上で `@hand_block(row, col)`、`check_hand_blocks(model, results)` |
| `TangentChecker(integrator).check(stress, state, strain_inc)`、`check_tangent(integrator, stress, state, strain_inc)` | `check_tangent(integrator, strain_inc, stress_n, state_n)`(`stress_update` と同じ引数順) |
| `TangentChecker(..., denom_offset=...)` | 廃止(下の「誤差の測り方」) |
| `result.ddsdde_ad` | `report.ddsdde`(手書きの接線や UMAT の接線も検査するので) |
| `CrosscheckStrainDriver(a, b).run(load)` / `CrosscheckStressDriver` | `crosscheck(a, b, load)`(driver は載荷履歴から決まる。それ以外は `driver_factory=`) |
| `cc.iter_run(load)` で最初の失敗を探す | `crosscheck(..., stop_on_failure=True)`、`report.first_failure` |
| `result.n_cases` / `result.cases` | `report.n_steps` / `report.steps` |
| `check_bindings(..., rtol=...)` が `{名前: (合否, 誤差)}` を返す | `check_bindings(..., tol=...)` が `BindingReport` を返す(`report.checks[名前].rel_err`) |
| `JacobianComparisonResult` / `TangentCheckResult` / `CrosscheckResult` | `JacobianComparisonReport` / `TangentReport` / `CrosscheckReport` |

**誤差の測り方。** 全検査で 1 つに揃えた: 配列(ブロック、接線、応力ベクトル)全体の最大差を、参照側の最大値で割る
— `max|候補 − 参照| / max|参照|`。合格は `誤差 <= tol`。0.1.0 では検査ごとに違い、分母に単位を持つ定数が入っていた
(`TangentChecker` は成分ごとに `|Δ| / (|fd| + 0.01)`、`check_bindings` は成分ごとに `|Δ| / (|f| + 1e-14)`)。
そのため同じ許容値でも、モデルを書いた単位によって合否が変わった。移行で変わりうる点:

- `check_tangent`: 0 に近い成分の誤差が、接線の最大成分に対する比になる。`denom_offset` で調整していた許容値は見直す。
- `check_bindings`: 厳密には 0 の成分に乗った丸め誤差では不合格にならない。
- `crosscheck` と `compare_jacobians` も、成分ごとではなく配列全体で測る。
- 参照側は、`check_tangent` では中心差分、`crosscheck` / `compare_jacobians` では 1 つ目の引数、`check_bindings` では Python 側。

追加:

- `check_solution(model, results)` — 手書きのリターンマッピングの解を、自動微分側の残差に代入して行ごとに検査する。
- `hand_jacobian(model, result)`(`returnmap.simulation`)— 宣言済みのブロックを手導出、残りを自動微分で埋めたヤコビアン。
- `crosscheck(..., lockstep=True)` — 各ステップで候補側に参照側の入力を渡して 1 ステップずつ比べる。
- `FortranIntegrator(parse_umat_failure=...)` — UMAT 自身の収束フラグを読む。
- `integrator.jacobian(result)` — 自動微分の残差ヤコビアン。

### import パス

公開の import 元は 7 つのパッケージで、どの名前も 1 つのパッケージにだけ属する。
サブモジュールのパス(`returnmap.core.state` など)は動くが、文書化されたインターフェースではない。

| 0.1.0 | 0.2.0 |
|---|---|
| `from manforge.core.material import MaterialModel, ParamMeta` | `from returnmap.core import MaterialModel, ParamMeta` |
| `from manforge.core.state import Implicit, Explicit, NTENS, SCALAR` | `from returnmap.core import ...` |
| `from manforge.core.dimension import SOLID_3D, ...` | `from returnmap.core import ...` |
| `from manforge.simulation.driver import StrainDriver` ほか | `from returnmap.simulation import ...` |
| `from manforge.verification import FortranModule, JacobianBlocks` | `from returnmap.simulation import FortranModule, JacobianBlocks` |
| `from manforge.verification import verified_against_fortran` | `from returnmap.core import verified_against_fortran` |
| `from manforge.service import ParamMeta` | `from returnmap.core import ParamMeta` |
| `from manforge.utils.smooth import smooth_sqrt` ほか | `from returnmap.utils import ...` |

削除: `manforge.autodiff`、`manforge.utils.tensor`、`manforge.utils.voigt`、`StressDimension.mandel_factors_jnp` /
`identity_jnp`、`Comparator` / `CaseResult` / `ComparisonResult` / `CrosscheckCaseResult`。

### ドキュメント

- ガイド(日本語)を `docs/` に、API リファレンス(英語)を docstring から生成する形にした。
  `make docs` で HTML を作る(`uv sync --extra docs` が必要)。
- README は概要・インストール・クイックスタートと、`docs/` への案内だけを持つ。

## 0.1.0

最初のタグ。
