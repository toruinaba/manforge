# API 再設計 設計メモ

- 状態: 2026-11-18 に全体を合意。決まっていない点は 6 章の未決事項だけである。
- 基準点: manforge `v0.1.0`(`10c586c`)、manforge-internal `99bc291`。本文中の `file:line` はこの基準点のもの。
- 対象読者: manforge と manforge-internal の両方を触る人。

## 1. 背景

機能を後方互換のまま足し続けた結果、API の一貫性が落ちている。調査で確認した症状は次のとおり。

- **`=None` が「省略可」ではなく「古い呼び出し経路では渡ってこない」を意味している。** `strain_inc=None` が 26 箇所、`stress_trial=None` が 15 箇所ある。`stress_update` 経由では両方とも常に渡される。
- **試行状態が状態(複数フィールド)になったのに、下流のフックはベクトル 1 本しか受け取らない。** `elastic_predictor`(#101)は σ 以外の試行値も返せるが、`elastic_check` / `state_residual` / `user_defined_return_mapping` が受け取るのは `stress_trial` だけである。
- **同じ概念に層ごとに別のスイッチ・別の名前がある。** 非収束の扱い、`tol`、状態履歴の要求方法、`dimension` の渡し方など。
- **検証層に性格の違うものが混ざっている。** 内省(`JacobianBlocks.part`)、テスト補助(Crosscheck)、導出の記録(`verified_against_fortran`)。

根拠の一覧は付録 A にある。

## 2. 方針
1. **0.x は破壊的変更可。** manforge と manforge-internal を同時に直す。互換シムは置かない。
2. **UMAT に入る量には必ず手書きスロットがあり、自動微分はそれを照合する参照側に立つ。** 自動微分だけで得られる量を成果物(Fortran UMAT)の前提にしない。
3. **フックには引数を足し続けない。** ステップの文脈を 1 つのオブジェクトにして渡す。
4. **設計の重点は「コアが現在表現できること全体」で決める。** manforge-internal での使用件数は移行コストの見積もりにだけ使う。
5. **公開 import パスは当面いじらない。** 深いモジュールパス(`manforge.core.state` など)も公開扱いとする。絞り込みとモジュールの移動は別の機会に行う。

## 3. コアが表現する範囲

弾性ステップでの内部変数の動きで 3 つに分ける。

| 型 | 弾性ステップでの内部変数 | 応力への戻り | 例 | 扱い |
|---|---|---|---|---|
| A | 動かない | — | 古典塑性、塑性駆動の損傷、E(ε_eq) の低下 | 対象 |
| B | 応力に追従して代数的に動く | なし | subloading の正規降伏比 R | 対象 |
| C | 動き、応力を変える | あり | 弾性エネルギー駆動の損傷、クリープ、二曲面 | **対象外** |

C のうち、全ひずみから閉形式で決まるもの(κ = max(κ_n, ε_eq)、D = g(κ))は予測子で書ける。連立を解く必要があるものは今回の設計に含めない。

弾性則は 2 種類を扱う。

- **線形**: σ_trial = σ_n + C(state_n) Δε。損傷で剛性が下がっても、D が弾性ステップで固定なら線形である。
- **非線形(代数的)**: 片側効果など。予測子で σ_trial を定義する。

## 4. 設計

### 4.1 ステップ文脈(名前は未決事項 1)

1 ステップの入力をまとめたオブジェクトを、全フックに渡す。

```python
@dataclass(frozen=True)
class StepContext:
    state_n: State          # ステップ開始時の状態。state_n["stress"] は σ_n
    strain_inc: FloatArray  # Δε(engineering shear)
    trial: State            # 試行状態。elastic_predictor の結果を state_n に重ねたもの
```

- フックの引数名は `step` とする。
- `trial` は全フィールドを持つ。予測子が返さなかったフィールドは `state_n` の値である。
- 将来 `DTIME` や温度が必要になったら、ここにフィールドを足す。既存モデルのフック署名は変わらない。
- 整合接線の計算では、フレームワークが Δε を摂動した `StepContext` を作り直して残差に渡す。試行状態の全フィールドが予測子を通して摂動される。

### 4.2 フック一覧

| 現在 | 再設計後 |
|---|---|
| `yield_function(state)` | 変更なし |
| `flow(state)` | 変更なし |
| `elastic_predictor(state_n, *, strain_inc)` | 変更なし(文脈を作る側なので `step` は受け取らない) |
| — | `elastic_tangent(step)`(新設、4.4) |
| `elastic_check(state_n, *, stress_trial, strain_inc=None)` | `elastic_check(step)` |
| `elastic_update_state(state_n, *, stress_trial, strain_inc=None)` | 廃止(4.3) |
| `update_state(dlambda, state_new, state_n, *, stress_trial=None, strain_inc=None)` | `update_state(new, dlambda, step)` |
| `state_residual(state_new, dlambda, state_n, *, stress_trial, strain_inc=None)` | `state_residual(new, dlambda, step)` |
| `default_stress_residual(state_new, dlambda, stress_trial)` | `default_stress_residual(new, dlambda, step)` |
| `user_defined_return_mapping(stress_trial, C, state_n)` | `user_defined_return_mapping(step)` |
| `user_defined_tangent(stress, state, dlambda, C, state_n, *, stress_trial=None, strain_inc=None)` | `user_defined_tangent(new, dlambda, step)` |

- 引数順は `(new, dlambda, step)` に統一する。全モデルの署名が変わるので、順序の統一に追加の費用はない。
- `C` 引数は廃止する。状態依存剛性では「どの時点の C か」が曖昧で、モデルは `self.elastic_stiffness(...)` を自分で呼べる。
- `update_state` に渡る `new` の explicit フィールドは、従来どおり `state_n` の値である。試行値ではない。試行値は `step.trial` から読む。

片側損傷の残差は、σ̃_trial を Δε から組み直す必要がなくなる。

```python
def state_residual(self, new, dlambda, step):
    C0 = self.undamaged_stiffness()
    R_eff = (new["stress_eff"] - step.trial["stress_eff"]
             + dlambda * (C0 @ self.flow(new).strain_like))
    ...
```

### 4.3 弾性ステップ
1 ステップの流れは次のとおり。

1. `elastic_predictor` が試行状態を作る。
2. `elastic_check(step)` が分岐を決める。既定は `yield_function(step.trial) <= 0`。
3. 弾性なら `step.trial` がそのまま収束状態になる。
4. 塑性なら、陰的未知数は試行値から NR を始める。

`elastic_update_state` は廃止し、予測子に統合する。B 型の量は試行状態のフィールドとして返す。

```python
def elastic_predictor(self, state_n, *, strain_inc):
    sig = state_n["stress"] + self.elastic_stiffness(state_n) @ strain_inc
    return [self.stress.update(sig), self.R.update(self.normal_yield_ratio(state_n, sig))]

def elastic_check(self, step):
    R_hat = step.trial["R"]
    return bool(R_hat <= self.R_e) or bool(R_hat <= step.state_n["R"])
```

- 既定の分岐が試行状態全体を見るので、integrator の特例(`integrator/base.py:437`)は不要になる。
- 統合の等価性は、テスト内の subloading fixture で確認した。載荷・除荷・再載荷・反転にせん断を加えた 38 ステップで、応力・R・接線・分岐・反復数が完全一致した。manforge-internal の `ExtendedSubloading` はパラメータとテストが無く未確認である。
- この統合は B 型であることが前提である。弾性則が R に依存する拡張をすると C 型になり、成り立たない。

### 4.4 弾性の定義と弾性接線

弾性の定義は予測子に一本化する。`elastic_stiffness` は線形弾性のときの簡易記法と位置づける。

弾性接線には手書きスロットを設ける。現状は予測子を override すると自動微分の値しか得られず、手書きの弾性接線を置く場所がない(`user_defined_tangent` は塑性ステップでしか呼ばれない)。

| integrator | 予測子が既定 | 予測子を override |
|---|---|---|
| `PythonNumericalIntegrator` | `elastic_stiffness(state_n)` | 予測子の自動微分 |
| `PythonAnalyticalIntegrator` | `elastic_stiffness(state_n)` | `elastic_tangent(step)`。未実装ならエラー |
| `PythonIntegrator` | `elastic_stiffness(state_n)` | `elastic_tangent(step)` があれば使い、なければ自動微分 |

- σ の予測が線形なモデル(subloading 型)の `elastic_tangent` は、`elastic_stiffness(step.state_n)` を返す 1 行になる。
- driver の初期推定は、integrator が返す弾性接線から取る。モデルの `elastic_stiffness` を直接読まない。
- `elastic_stiffness(state=None)` の `None` は廃止する。無損傷剛性が要るモデルは、別名のヘルパで取る。

### 4.5 手書き経路
`user_defined_return_mapping(step)` は試行状態の全フィールドと Δε を受け取る。これで、予測子を override したモデルでも閉形式や手書き NR が書ける。

この再設計で、UMAT の構成要素すべてに手書きスロットが揃う。

| UMAT の要素 | 手書きスロット | 照合相手 |
|---|---|---|
| 試行状態 | `elastic_predictor` | 次元間突合、Fortran |
| 分岐 | `elastic_check` | 同上 |
| 弾性接線 | `elastic_tangent` | 予測子の自動微分 |
| 残差・ヤコビアンのブロック | 手導出ブロックの宣言(4.8) | `JacobianBlocks.part` |
| NR ループ | `user_defined_return_mapping` | 数値 NR |
| 塑性接線 | `user_defined_tangent` | 陰関数微分 |

### 4.6 結果オブジェクト
- **`StressUpdateResult` はステップ文脈を持つ。** 現在の `_state_n` と `stress_trial` を `step` に置き換える。ヤコビアンの内省が `state_n` や `strain_inc` の持ち回りなしで済み、別の残差で計算する取り違えが起きなくなる。
- **`FortranIntegrator` の結果**は `state_n` と `strain_inc` だけを持ち、`trial` は持たない。
- **`DriverResult` に `converged` と `failed_step` を持たせる。** 非収束ステップの結果は配列に入れない。service の `SimulationOutput` と同じ意味になる。

### 4.7 非収束と integrator の契約
- **スイッチは integrator の `raise_on_nonconverged` だけにする。** driver は自分の外側 NR の失敗にも同じ設定を使う。`iter_run(raise_on_nonconverged=...)` は廃止する。
- **`StressIntegrator` を `typing.Protocol` にする。** `stress_update` と弾性接線の取得を必須にし、`hasattr` による分岐と `types.MethodType` による動的 bind をやめる。弾性接線を持たない Fortran UMAT は、ゼロひずみ増分の呼び出しで代用する実装を integrator 側に持つ。

段 1 で実装した契約は次のとおり。

| メンバ | 内容 |
|---|---|
| `dimension` / `ntens` | 次元 |
| `raise_on_nonconverged` | 非収束の唯一のスイッチ。driver も読む |
| `initial_state()` | 初期状態 |
| `elastic_tangent(stress_n, state_n)` | Δε = 0 での弾性接線。driver の初期推定に使う |
| `stress_update(strain_inc, stress_n, state_n)` | UMAT 1 回分 |

- integrator から `elastic_stiffness` は無くした。Python の integrator の `elastic_tangent` は、段 1 の時点ではモデルの `elastic_stiffness(state_n)` を返す。予測子の微分に切り替えるのは段 2(4.4)である。
- `trial_stress` は契約に含めない。Python の integrator だけが持つ。`SubSteppingIntegrator` は、包んだ integrator がこれを持たないとき `stress_trial=None` を返す。Fortran UMAT の結果が `stress_trial=None` であることに合わせた。段 2 で結果が文脈を持つようになれば、この分岐は不要になる。
- fitting は、`driver.run` が `converged=False` を返したとき `RuntimeError` にする。非収束を値で返す integrator を `driver_factory` が使っても、ペナルティとして扱われる。

### 4.8 手導出ブロックの宣言(API の細部は段 3 で確定)

手導出したヤコビアンのブロックを、モデル上で名前付きで宣言する。

```python
@hand_block("theta", "stress")        # ∂R_θ/∂σ
def dRt_ds(self, new, dlambda, step): ...
```

- **名前**: 行は残差の行名、列は未知数名で、`JacobianBlocks.part[row][col]` と同じである。クラス定義時にレイアウトと照合し、存在しない名前はその場でエラーにする。
- **組み立て**: フレームワークがレイアウトに従って `.part` と同型に組む。固定スライスと、次元ごとの `calc_jacobian` が不要になる。
- **未導出のブロック**: 自動微分で埋められる。導出を 1 ブロックずつ進めながら、途中でも解ける。各ブロックの出所(手導出か自動微分か)は結果に残す。
- **近似を含むブロック**: 手書き NR が explicit 状態を凍結して扱うモデルでは、手導出と自動微分が一致しないブロックがある(Continuous YU の θ / β / R 行。現在のテストは 2e-2 を許容している)。許容値と理由をブロックごとに宣言できるようにする。
- **Fortran 側の対応**: 既存の `verified_against_fortran` を同じメソッドに重ねる。
- **対象外**: 手書き NR のループ本体は汎用化しない。`project_theta` のようなモデル固有の処理があるためである。

照合は 1 回の呼び出しにする。宣言済みブロックを自動微分と履歴全体で比べ、ブロック別の最悪誤差と導出済み数を返す。

### 4.9 検証層の役割

| 役割 | 中身 | 寿命 |
|---|---|---|
| 内省 | `JacobianBlocks.part`、レイアウト、自動微分の接線 | 恒久 |
| テスト補助 | Crosscheck、`TangentChecker`、履歴生成、手導出ブロックの照合 | テストと同じ |
| 導出の記録 | `verified_against_fortran`、手導出ブロックの宣言 | モデルと同じ |

- **内省は integrator から取れるようにする。** `integrator.jacobian(result)` が `JacobianBlocks` を返す。`.part[row][col]` の形は変えない。
- **`JacobianChecker.compare` は見直す。** 2 つの解を同じ自動微分残差で比べるもので、手導出と自動微分の比較ではない。`iter_blocks` は両リポジトリで使用 0 件である。
- **モジュールの移動は今回しない**(方針 5)。`manforge.verification` のパスは残し、中身の役割だけを整理する。

### 4.10 パラメータ宣言と `dimension`
`param_names` から属性を設定する処理を基底に持たせ、次元はクラス属性で固定する。

```python
class ChabocheKinematic(MaterialModel):
    param_names = ["E", "nu", "sig_y0", "C", "gamma", "Qinf", "b"]
    ...

class ChabocheKinematic3D(ChabocheKinematic):
    dimension = SOLID_3D
```

- manforge-internal では、`self.X = X` の 144 行と、次元別クラスのコンストラクタ 23 個が不要になる。`SatoKinematic` は 53 個のパラメータを 3 回書き写している。
- `fitting/objective.py:124` の `inspect.signature` による判定が不要になる。
- 4.1〜4.9 と独立に進められる。

### 4.11 入口と命名の整理
- **`fit_params`**: `fit_params(model, targets, space, *, driver_factory=...)` の 1 形式にする。dict の省略形は明示的なコンストラクタに移す。
- **`nr_jacobian`**: 既定値を層の間で揃え、型を `Literal` にする。
- **`tol`**: 意味ごとに名前を分ける(NR の収束、driver の外側 NR、検査の合否)。
- **状態履歴の要求**: `collect_state={"ep": FieldType.STRAIN}` と `include_fields=["ep"]` を、名前の列に統一する。
- **残骸の削除**(段 1 で実施): `mandel_factors_jnp` / `identity_jnp`、`manforge.autodiff` パッケージ、`utils/tensor.py`、`utils/voigt.py`、`check_tangent`、`CrosscheckCaseResult` の `py_*` / `f_*`、`__all__` の `_PythonIntegratorBase`。当初は `autodiff` のうち `operators.py` と `backend.check_float64` だけを挙げていたが、それを除くと使う側が残らないためパッケージごと削除した。`utils/voigt.py` も同じ理由で参照が無くなったので削除した。

## 5. 移行計画
各段で manforge と manforge-internal の PR を対にし、両方のテストが通る状態で終える。manforge-internal は editable 依存で CI が無いので、manforge 側の PR ごとに internal のテストを手元で走らせる。

| 段 | 内容 | internal への影響 |
|---|---|---|
| 0 | タグ `v0.1.0`、internal のテスト修正 | 完了 |
| 1 | 残骸の削除(#103)、非収束スイッチの一本化、`DriverResult.converged`、`StressIntegrator` の Protocol 化 | なし(未使用)。internal のテストは変更前後で同じ結果 |
| 2 | ステップ文脈、フック署名、`elastic_tangent`、`elastic_update_state` の廃止、結果が文脈を持つ | フック 29 定義の書き換え |
| 3 | 手導出ブロックの宣言、`integrator.jacobian`、照合の補助 | Continuous YU を 1 次元ぶん試験移行 |
| 4 | パラメータ宣言と `dimension` | 次元別コンストラクタ 23 個の削除 |
| 5 | 入口と命名の整理 | なし(fitting 未使用) |
| 6 | ドキュメント基盤、公開範囲の絞り込み | import の書き換え |

- **段 2 が最大である。** フック定義は manforge のモデルに 20、manforge のテスト内モデルに約 65、manforge-internal に 29 ある。
- **段 2 の合格条件は数値の完全一致とする。** 署名の変更だけなので、既存モデルの結果は変わらないはずである。`v0.1.0` で代表的な履歴の結果を保存し、変更後とビット単位で比べる。
- **段 3 は段 2 の後に行う。** 手導出ブロックの署名が `step` を使うためである。
- **段 4 は順序を入れ替えられる。** 段 2 より先に行えば、書き換えるコンストラクタが減る。

## 6. 未決事項

1. **文脈オブジェクトの名前。** `StepContext` は driver の `DriverStep` と紛らわしい。`Increment` などの候補がある。
2. **無損傷剛性の取り方。** `elastic_stiffness(None)` をやめた後のヘルパの名前と置き場所。
3. **パラメータの既定値。** `ContinuousYUKinematic` は `H=0.0` などの既定値を持つ。`ParamMeta.default` は入力フォーム用の値で意味が違う。宣言方法を決める必要がある。
4. **派生属性の初期化。** YU の PS 版が持つ `P` / `T` のように、パラメータから導く属性をどこで作るか。
5. **次元を固定するクラスの判定。** `J2Isotropic3D` の閉形式解は 3D と平面ひずみでしか成り立たない。別の次元を渡したときに拒否する規則が要る。
6. **手導出の残差も宣言するか。** 現在の `calc_residual` は explicit 状態を凍結した形で、フレームワークの残差と構造が違う。
7. **Crosscheck ドライバの扱い。** manforge-internal では使われていない。段 3 の試験移行の結果を見て、残すか形を変えるかを決める。
8. **ドキュメントのツールと言語。** Sphinx + MyST を候補としているが未決定である。

## 付録 A. 調査で確認した事実

| 事実 | 場所 |
|---|---|
| 予測子 override 時、既定の `elastic_check` を呼ばず integrator が降伏関数を評価する | `simulation/integrator/base.py:437` |
| 弾性分岐は `user_defined_tangent` を呼ばずに return する | `simulation/integrator/base.py:454-477`、呼び出しは `:488` |
| `stress_trial=None` のとき収束解から試行応力を逆算する | `simulation/integrator/base.py:257` |
| 片側損傷の残差が σ̃_trial を Δε から組み直している | `models/j2_damage_unilateral.py:238` |
| `default_stress_residual` は型で `None` を許し実行時に raise する | `core/material/base.py:693` |
| `StressDriver.run` は `raise_on_nonconverged=True` を固定で渡す | `simulation/driver.py:425` |
| 非 raise の integrator を `StrainDriver.run` に渡すと、10 ステップ中 3 行で返り、最終行は非収束、`DriverResult` に目印がない | 実行して確認 |
| `build_objective` が `inspect.signature` で `dimension` 引数の有無を調べる | `fitting/objective.py:124` |
| manforge-internal が import する manforge のシンボルは 30 個 | 両リポジトリの import を集計 |
| manforge-internal のフックで `strain_inc` を本体で使うものは 24 定義中 0 | AST で集計 |
| manforge-internal の Fortran 突合テスト 5 本は `FortranIntegrator` / `Crosscheck*` を使わない | `tests/yu/benchmarks/test_*_vs_fortran.py` |
