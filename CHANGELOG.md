# 変更履歴

このファイルは、利用者から見える変更を版ごとにまとめる。設計の経緯と理由は
[`docs/design/api-redesign.md`](docs/design/api-redesign.md) にある。

## 未リリース

文書を足した。計算結果は変わらない。

- **「新しいモデルを足す」** — Python のモデルから検証済みの UMAT までの手順のページと、その完成形の例
  (`examples/model_family/`: J2 + Voce 硬化の 1 ファミリを、独立したリポジトリの形で持つ)。
- **組み込みモデルの定式化** — 各ファミリのディレクトリに `README.md` を置いた(式、状態変数、パラメータ、
  飽和値などの性質、検証の範囲)。ガイドの「組み込みモデルと次元選択」から読める。
- **`OWKinematic` の `gamma` に単位を付けた**(`param_meta` の `unit="1/MPa"`)。発展則が
  `γ·‖α‖·α` なので、Armstrong-Frederick の `gamma`(無次元)とは単位が違う。値と計算は変わらない。

## 0.3.2(2026-11-21)

自動微分のリターンマッピングが、失敗したステップを 1 回解き直すようになった。今まで解けていた計算の結果は
変わらない(ビット単位で同じ)。変わるのは、今まで非収束になっていたステップの一部が解けるようになることである。

- **失敗したステップの解き直し。** 自動微分の NR は試行点(σ_trial、Δλ = 0)から始める。増分が降伏ひずみの数倍に
  なると、この反復が Δλ < 0 の根に 2 次収束したり、発散したりするモデルがある。そのステップだけ、Δλ の初期値を
  塑性側の見積もり(試行点で σ と Δλ だけの Newton 1 歩を取ったときの Δλ)に変えて、もう 1 回解く。方程式は同じ
  なので、解ければ同じステップの根であり、増分の分割のような別の積分にはならない。
- **`nr_restart=False`** — 解き直しをしない integrator(0.3.1 までの解法)。解き直しの有無を同じ履歴で比べる
  ために残してある: `PythonIntegrator(model, nr_restart=False)`。`compare_models` は integrator をそのまま
  受け取るので、2 つを並べられる。
- **記録** — `StressUpdateResult.n_restarts` / `ReturnMappingResult.n_restarts`(0 か 1)、`ModelRun.n_restarts`
  (履歴での合計)。`compare_models` の表には、解き直しがあったときだけ `restarts` の列が出る。解き直した
  ステップの `n_iterations` と `residual_history` は 2 回分を持つ。
- **変わらないもの** — 手書きのリターンマッピング(`user_defined_return_mapping`)と UMAT は対象外で、今までどおり。
  2 回目も解けなければ今までどおり非収束(`raise_on_nonconverged` に従う)。
- **変わりうるもの** — 今まで途中で止まっていた履歴が最後まで進む。`fit_params` では、ペナルティになっていた評価の
  一部が成功するようになるので、探索の道筋が変わることがある。

確かめた範囲: 組み込みの 21 モデルの検査用の履歴は、解き直しの有無で結果が同じ(解き直しは 1 回も起きない)。
returnmap-internal の 23 モデルを合わせた 85 本の履歴では、完走が 78 本から 85 本になった。

## 0.3.1(2026-11-20)

returnmap-internal のモデルを共通の検査にかけて分かった、検査の側の抜けを埋めた。0.3.0 からの非互換は無い。

- **接線が 0 のステップで `check_tangent` が必ず不合格になっていた。** 降伏棚や 1 次元の完全塑性では、整合接線が
  厳密に 0 になる。中心差分は丸め誤差(弾性剛性の 1e-12 倍程度)を返すので、「0 と丸め誤差」を比べて誤差 1 と
  判定していた。差分の接線が弾性剛性より 8 桁以上小さいときは、弾性剛性に対して測るようにした
  (`TangentReport.elastic_scale`)。それ以外の接線の測り方は変わらない。
- **`Umat(props=[...])`** — UMAT が受け取るパラメータの名前を、その順で宣言する。`param_names` と順が違う UMAT や、
  一部しか受け取らない UMAT(Python 側だけで分岐を選ぶパラメータがあるもの)を宣言できる。省略すると
  `param_names` の順。
- **`standard_histories(model, eps_y=...)`** — 標準の載荷履歴を測る尺度のひずみを渡せる。弾性域の無いモデル
  (下負荷面モデルなど)は降伏ひずみが定まらないので、ファミリの `verification.py` で
  `histories = lambda model: standard_histories(model, eps_y=...)` と書く。標準と同じ名前で宣言した履歴は、
  標準のものを置き換える。

## 0.3.0(2026-11-20)

モデルの置き方と検証の仕方を整えた。公開の import パスと計算結果は 0.2.1 から変わらない。変わるのは、
リポジトリの中のファイルの場所、`returnmap build` の書き方、Makefile のターゲット名である。

**モデルを 1 ファミリ 1 ディレクトリにまとめた。** 1 つのモデルに属するものが 5 箇所に散っていた(モデル本体、
`tests/unit/models/`、`tests/integration/models/`、`tests/benchmarks/<model>/`、`fortran/`)のを、
`src/returnmap/models/<family>/` に集めた。計算結果は変わらない。

```
src/returnmap/models/yu_kinematic/
  __init__.py      # 公開するクラス
  model.py         # モデル本体
  fortran/         # UMAT のソース(持つモデルだけ)
  tests/           # そのモデルのテスト
```

- **import は変わらない。** `from returnmap.models import YUKinematic3D` のまま。
- **モデルのテストも配布物に入る。** pip で入れた先で `pytest --pyargs returnmap.models.yu_kinematic` を実行できる
  (UMAT との比較は、コンパイルしたモジュールが無ければ skip される)。UMAT のソースも入る。
- **UMAT のソースの場所**: `fortran/j2_isotropic_3d.f90` → `src/returnmap/models/j2_isotropic/fortran/`、
  `fortran/yu_kinematic_{3d,ps}.f90` → `src/returnmap/models/yu_kinematic/fortran/`。`fortran/` には全 UMAT が共有する
  `abaqus_stubs.f90` が残り、f2py モジュールのビルド先でもある(`make fortran-build-*` は従来どおり)。
- **Makefile**: `make test` は全モデルのテスト(slow と fortran を除く)を含む。`test-benchmarks` /
  `test-benchmarks-fortran` は `test-models` / `test-models-fortran` になった。`tests/benchmarks/` は無くなった。
- **`@verified_against_fortran(test=...)`** のパスは、モデルのモジュールがあるディレクトリから見た相対パスで書ける
  (`"tests/test_numerical_vs_fortran.py::..."`)。ファミリのディレクトリを動かしても切れない。リポジトリ直下からの
  パスも従来どおり使える。

**共通の検査 `check_model` を足した。** モデルが宣言しているものに応じて、当てはまる検査を全部走らせ、何を検査したかを
報告する。Python のモデルだけ(自動微分で解くだけ)なら「収束」と「接線」で合格し、`@hand_block` や手書きの
リターンマッピングを足すと、その検査が加わる。

```python
report = check_model(YUKinematic1D.at_defaults())
assert report.passed, report.summary()
```

- **`MaterialModel.at_defaults()`** — `param_meta` の目安の値でモデルを作る(`Model.at_defaults(gamma=50.0)` のように
  一部を置き換えられる)。コンストラクタが全パラメータを要求するのは変わらない。
- **検証の宣言 `verification.py`** — 各ファミリのディレクトリに置く。`MODELS`(検査にかけるモデル)と、必要なら
  `histories(model)`(標準の載荷履歴に足すもの)。`check_model` が名前で見つける。
- **`tests/test_verification.py`** — 各ファミリで `check_model` を回す。全参照モデル 21 個が対象。
- **`standard_histories(model)`** — 検査に使う標準の載荷履歴(単軸と、直応力・せん断を同時に動かす多軸)。
- `estimate_yield_strain` は、降伏関数が応力以外の状態を見るモデル(準片側損傷)でも降伏ひずみを返す。これまでは
  単位を持つ値(剛性の逆数)に落ちていて、`generate_strain_history` の履歴が降伏に届かなかった。

**意図した近似を、近似になるステップだけ宣言できるようにした。**

- `@hand_block(row, col, approx_when="メソッド名", note=...)` — そのメソッドが真を返すステップだけ近似と宣言する。
  該当するステップは比較から外して数え、ほかのステップは厳密な許容値のまま検査する(`approx=` も付けると、
  外さずにその値で判定する)。これまでの `approx=` だけの宣言(全ステップに効く)はそのまま使える。
- `@hand_tangent(approx=..., approx_when=..., note=...)` — 手書きの接線(`user_defined_tangent`)の近似を
  同じ形で宣言する。`check_model` が読む。これまで接線には近似を宣言する手段が無かった。
- `HandBlockCheck` に `n_approx_points` / `approx_err` / `approx_tol`、`ModelReport` に `approximations`
  (モデルが宣言している近似の理由)を足した。

**YU の 3D と平面応力の手導出に、近似の宣言を付けた。** `check_model` を全参照モデルに回して分かったこと:
手導出のヤコビアンと手書きの接線が、停留曲面のゲート(`smooth_heaviside(g_stag)`)を定数として扱っている。
ゲートが 0 か 1 のステップでは厳密だが、収束点の `g_stag` が平滑化の幅に入ったステップ(ゲートが中間の値)では、
接線が最大 4 % ずれ、平面応力の `[theta][beta]` ブロックが自動微分と合わない。応力と状態はそのステップでも
自動微分と一致する。導出は変えず、そのステップだけ近似であること(`stagnation_gate_in_transition`)を宣言した。
標準の履歴では 80 ステップ中 1〜3 ステップが該当する。

**UMAT のビルドとテストに要るものを、パッケージが配るようにした。** returnmap を入れてあれば、モデルを置く
リポジトリに要るのは UMAT のソースとテストだけになる。

- **`abaqus_stubs.f90` がパッケージに入った**(`fortran/abaqus_stubs.f90` → `src/returnmap/fortran/`)。
  `returnmap build --abaqus-stubs <source.f90> --name <module>` が一緒にコンパイルする。`returnmap stubs` は
  そのパスを出す(f2py やコンパイラを直接呼ぶとき用)。
- **`returnmap build --link <lib>`** — リンクするライブラリを渡す(`--link lapack --link blas`)。これまでは
  渡せなかったので、LAPACK を呼ぶ UMAT(YU)は Makefile から f2py を直接呼んでいた。`make fortran-build-*` は
  すべて `returnmap build` を使うようになった。
- **pytest のプラグイン**(`returnmap.pytest_plugin`)— returnmap を入れてあれば自動で効く。`fortran` / `slow`
  マーカーを登録し、プロジェクト直下の `fortran/`(ビルド先)を `sys.path` に足す。`conftest.py` と
  `pyproject.toml` に同じことを書かなくてよい。pip で入れた先で `pytest --pyargs returnmap.models` を実行しても、
  未登録マーカーの警告が出なくなった。

| これまで | これから |
|---|---|
| `returnmap build fortran/abaqus_stubs.f90 <source.f90> --name <module>` | `returnmap build --abaqus-stubs <source.f90> --name <module>` |
| `python -m numpy.f2py -c abaqus_stubs.f90 <source.f90> -m <module> -llapack -lblas` | `returnmap build --abaqus-stubs <source.f90> --name <module> --link lapack --link blas` |

**UMAT を宣言する。** UMAT まで行ったモデルは、ファミリの `verification.py` に `UMATS` を書く。

- **`Umat(model, module=, sources=, libraries=, ...)`**(`returnmap.verification`)— モデルのクラスと、その UMAT の
  ソース・リンクするライブラリの対応。
- **`returnmap build --family <package>`** — 宣言された UMAT を全部ビルドする(`--family returnmap.models` で
  全ファミリ)。`make fortran-build-umat` / `fortran-build-yu` はこれを呼ぶ。`fortran-build-yu-ps` は
  `fortran-build-yu` に含めた。`fortran-build-all` を足した。
- **`check_model` に `UMAT` の行** — ビルド済みの UMAT を、Python のモデルと同じ載荷履歴で走らせて全ステップで
  比べる。未ビルドなら `not built` で、不合格にはしない。


## 0.2.1(2026-11-20)

既存のモデルを使う人のための入口を整えた。0.2.0 からの非互換は無い。

- **チュートリアル「試験データにモデルを合わせる」**(`docs/guide/tutorial.md`)。CSV の読み込みから、手での調整、
  フィット、全点での仕上げ、パラメータを振るところまで。通しのスクリプトは `examples/example_identification.py`、
  スライダーで調整する marimo のノートブックは `examples/notebook_tuning.py`(`uv sync --extra notebook`)。
- **README のクイックスタートを引張・圧縮試験にした。** これまでは 3D モデルに ε11 だけを与える単軸ひずみだった。
- **`fit_params(..., progress=...)`** — 評価のたびに `progress(回数, その残差, 最良の残差)` を呼ぶ。`progress=print`
  で長いフィットの進み具合が見える。
- **`SlowFitWarning`** — 目的関数の 1 回の評価に 0.5 秒(`SLOW_EVALUATION_SECONDS`)以上かかるとき、`fit_params` が
  探索の前に 1 度だけ警告する。実測の時間、手書きのリターンマッピングを持たないモデルならその旨、短くする手段を
  知らせる。フィットはそのまま進む。
- **`YUKinematic` の `param_meta`** — 12 個のパラメータに、説明・単位・目安の値・範囲を付けた。全参照モデルの
  `param_meta` が揃っていること、目安の値で計算が通ることをテストで確認している。

## 0.2.0(2026-11-20)

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
- **`fit_params` が、桁の大きいパラメータを動かさないまま収束と報告することがあった。** `"L-BFGS-B"` と
  `"Nelder-Mead"` は探索変数をそのままの値で扱っていたので、`C_k`(1e4 台)と `gamma`(1e2 台)のように桁の違う
  パラメータでは、大きいほうがほとんど動かなかった。探索変数を、それぞれの範囲(`ParameterSpace` の上下限)の幅で
  割って探索するようにした。結果・履歴・範囲の書き方は変わらない。同じ最適値に、これまでの 1/2〜1/3 の評価回数で
  着く(AF モデルの 3 パラメータの同定で 104〜184 回 → 48〜56 回)。自由変数には、ありうる範囲を表す有限の上下限を
  与えること。

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
