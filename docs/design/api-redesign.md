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

### 4.1 ステップ文脈 `Increment`

1 ステップの入力をまとめたオブジェクトを、予測子以外の全フックに渡す。フックの引数名は `inc` とする。

```python
@dataclass(frozen=True)
class Increment:
    state_n: State          # ステップ開始時の状態。state_n["stress"] は σ_n
    strain_inc: FloatArray  # Δε(engineering shear)
    trial: State            # 試行状態。elastic_predictor の結果を state_n に重ねたもの
```

- `trial` は全フィールドを持つ。予測子が返さなかったフィールドは `state_n` の値である。
- `state_n["stress"]` は、`stress_update` に渡された σ_n である。呼び出し側の dict に入っていた値ではない。
- 将来 `DTIME` や温度が必要になったら、ここにフィールドを足す。既存モデルのフック署名は変わらない。
- 整合接線の計算では、フレームワークが Δε を摂動した `Increment` を作り直して残差に渡す。試行状態の全フィールドが予測子を通して摂動される。

### 4.2 フック一覧

| `v0.1.0` | 再設計後 |
|---|---|
| `yield_function(state)` | 変更なし |
| `flow(state)` | 変更なし |
| `elastic_stiffness(state=None)` | `elastic_stiffness(state)`(4.4) |
| `elastic_predictor(state_n, *, strain_inc)` | 変更なし(`Increment` を作る側なので受け取らない) |
| — | `elastic_tangent(inc)`(新設、4.4) |
| `elastic_check(state_n, *, stress_trial, strain_inc=None)` | `elastic_check(inc)` |
| `elastic_update_state(state_n, *, stress_trial, strain_inc=None)` | `elastic_update_state(inc)`(4.3) |
| `update_state(dlambda, state_new, state_n, *, stress_trial=None, strain_inc=None)` | `update_state(new, dlambda, inc)` |
| `state_residual(state_new, dlambda, state_n, *, stress_trial, strain_inc=None)` | `state_residual(new, dlambda, inc)` |
| `user_defined_return_mapping(stress_trial, C, state_n)` | `user_defined_return_mapping(inc)` |
| `user_defined_tangent(stress, state, dlambda, C, state_n, *, stress_trial=None, strain_inc=None)` | `user_defined_tangent(new, dlambda, inc)` |

- 引数順は `(new, dlambda, inc)` に統一した。
- `C` 引数は廃止した。状態依存剛性では「どの時点の C か」が曖昧で、モデルは `self.elastic_stiffness(...)` を自分で呼べる。
- `update_state` に渡る `new` の explicit フィールドは、`state_n` の値である。試行値ではない。
- **引数リストはクラス定義時に検査する。** フックは位置引数で呼ばれるので、古い署名のモデルは例外にならず、引数がずれたまま動いてしまう。上の表のうち署名が変わったフックについて、引数名が違えば `TypeError` にする。
- **`default_stress_residual(new, dlambda, stress_trial)` は σ_trial を受け取る形のまま**にした。当初は `inc` を渡す形を考えていたが、これはフックではなく補助関数で、Fortran に対応する手書きの残差(`calc_residual`)は σ_trial しか持たない。必要なものだけを受け取る形のほうが両方から呼べる。`None` は受け付けない。

### 4.3 弾性ステップ

1 ステップの流れは次のとおり。

1. `elastic_predictor` が試行状態を作る。
2. `elastic_check(inc)` が分岐を決める。既定は `yield_function(inc.trial) <= 0`。
3. 弾性なら `inc.trial` が収束状態になる。その上に `elastic_update_state(inc)` の更新を重ねる。
4. 塑性なら、陰的未知数は試行値から NR を始める。

既定の分岐が試行状態全体を見るので、integrator の特例(`v0.1.0` の `integrator/base.py:437`)は無くなった。

**`elastic_update_state` は残す。** 当初は予測子に統合する方針だったが、実装中に次の 2 点が分かり、撤回した。

- **既定の分岐判定が意味を失う。** 追従変数(subloading の R)を試行状態に載せると、`yield_function(試行状態)` が恒等的に 0 になる。`elastic_check` を override し忘れたモデルは、黙って常に弾性になる。`v0.1.0` では同じモデルが初回から塑性と判定されていた(テスト `_NoHook` で確認)。
- **追従変数しか持たないモデルまで弾性接線が自動微分になる。** 予測子を override することになるので、`elastic_tangent` を 1 行書かない限り、手書きの `elastic_stiffness` が使われなくなる。

役割は次のように分かれる。

| フック | いつ効くか | 何を返すか |
|---|---|---|
| `elastic_predictor` | 毎ステップ。分岐判定と塑性 NR の出発点になる | 弾性則(σ と、弾性的に動く陰的未知数の試行値) |
| `elastic_update_state` | 弾性と判定された後だけ | 応力に追従する explicit 状態 |

`elastic_update_state` の更新は応力に戻らない前提である(B 型)。弾性ステップの更新が応力を変える発展則(C 型)は扱わない。

### 4.4 弾性の定義と弾性接線

- **`elastic_stiffness(state)` は状態を必須にした。** `None` で「無損傷の剛性」を表す用法をやめた。
- **`isotropic_stiffness()` を追加した。** `self.E` と `self.nu` からモデルの次元の等方剛性を作る。`elastic_stiffness` の既定はこれを返す。剛性が状態で変わるモデルは `(1 − D) * self.isotropic_stiffness()` のように掛けて書く。λ、μ の手計算は不要になる。
- **弾性の定義は予測子である。** `elastic_stiffness` は線形弾性のときの簡易記法で、予測子を override したモデルでは参照剛性にとどまる。

弾性接線には手書きスロット `elastic_tangent(inc)` を設けた。`v0.1.0` では、予測子を override すると自動微分の値しか得られず、手書きの弾性接線を置く場所がなかった(`user_defined_tangent` は塑性ステップでしか呼ばれない)。

| integrator | 予測子が既定 | 予測子を override |
|---|---|---|
| `PythonNumericalIntegrator` | `elastic_stiffness(state_n)` | 予測子の自動微分 |
| `PythonAnalyticalIntegrator` | `elastic_stiffness(state_n)` | `elastic_tangent(inc)`。未実装ならエラー |
| `PythonIntegrator` | `elastic_stiffness(state_n)` | `elastic_tangent(inc)` があれば使い、なければ自動微分 |

driver の初期推定は、integrator の `elastic_tangent(stress_n, state_n)` から取る(4.7)。Python の integrator では、これはまだモデルの `elastic_stiffness(state_n)` を返す。予測子を override したモデルでは参照剛性だが、初期推定にしか使わないので、外側の反復は正しい応答に収束する。

### 4.5 手書き経路

`user_defined_return_mapping(inc)` は試行状態の全フィールドと Δε を受け取る。これで、予測子を override したモデルでも閉形式や手書き NR が書ける。`State` は読み取り専用なので、手書き NR の作業用 dict は `dict(inc.state_n)` で作る。

UMAT の構成要素すべてに手書きスロットが揃う。

| UMAT の要素 | 手書きスロット | 照合相手 |
|---|---|---|
| 試行状態 | `elastic_predictor` | 次元間突合、Fortran |
| 分岐 | `elastic_check` | 同上 |
| 弾性ステップの追従変数 | `elastic_update_state` | 同上 |
| 弾性接線 | `elastic_stiffness` / `elastic_tangent` | 予測子の自動微分 |
| 残差・ヤコビアンのブロック | 手導出ブロックの宣言(4.8) | `JacobianBlocks.part` |
| NR ループ | `user_defined_return_mapping` | 数値 NR |
| 塑性接線 | `user_defined_tangent` | 陰関数微分 |

### 4.6 結果オブジェクト

- **`StressUpdateResult` は `increment` を持つ。** `v0.1.0` の `_state_n` と `stress_trial` フィールドを置き換えた。`stress_trial` はプロパティとして残る。ヤコビアンの内省は `JacobianChecker(model).compute(result)` だけで済み、`state_n` や `strain_inc` を別に渡して別の残差で計算する取り違えが起きない。
- **`FortranIntegrator` の結果は `increment=None` である。** UMAT は試行状態を報告しない。`JacobianChecker.compare(python_result, fortran_result)` は、Python 側の結果の `increment` で両方の点を評価する。
- **integrator の契約に `increment(strain_inc, stress_n, state_n)` を加えた。** `SubSteppingIntegrator` が、分割前の増分の `Increment` を結果に載せるために使う。Fortran は `None` を返す。段 1 で残っていた `trial_stress` の有無による分岐は無くなった。
- **`DriverResult` に `converged` と `failed_step` を持たせる。** 非収束ステップの結果は配列に入れない(段 1 で実施)。

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
| `increment(strain_inc, stress_n, state_n)` | そのステップの `Increment`(段 2 で追加)。Fortran は `None` |
| `stress_update(strain_inc, stress_n, state_n)` | UMAT 1 回分 |

- integrator から `elastic_stiffness` は無くした。Python の integrator の `elastic_tangent` は、段 1 の時点ではモデルの `elastic_stiffness(state_n)` を返す。予測子の微分に切り替えるのは段 2(4.4)である。
- 段 1 の時点では `trial_stress` を契約に含めず、`SubSteppingIntegrator` がその有無で分岐していた。段 2 で `increment(strain_inc, stress_n, state_n)` を契約に加え、この分岐は無くなった(4.6)。Fortran を包んだとき `stress_trial=None` になる挙動は同じである。
- fitting は、`driver.run` が `converged=False` を返したとき `RuntimeError` にする。非収束を値で返す integrator を `driver_factory` が使っても、ペナルティとして扱われる。

### 4.8 手導出ブロックの宣言

手導出したヤコビアンのブロックを、モデル上で名前付きで宣言する(`core/material/hand_block.py`)。

```python
@hand_block("theta", "stress")                       # ∂R_θ/∂σ
def calc_ft_fs(self, state, dlambda): ...

@hand_block("theta", "dlambda", approx=2e-2, note="r_max を凍結して導出している")
def calc_ft_fl(self, state, dlambda, state_n): ...
```

- **名前**: 行は残差の行名、列は未知数名で、`JacobianBlocks.part[row][col]` と同じである。クラス定義時にレイアウトと照合し、存在しない名前はその場でエラーにする。
- **デコレータは記録を付けるだけ。** メソッドは変わらず、直接呼べる。`verified_against_fortran` と重ねられる。
- **引数は名前で埋める。** 当初は `(new, dlambda, inc)` に統一する案だったが、手導出ブロックは Fortran のルーチンに対応する利用者のコードで、引数リストもそれに合わせて書かれている(`calc_fy_fs(state)`、`calc_ft_fl(state, dlambda, state_n)` など)。署名を揃えさせると、既存の全ブロックと、それを直接呼ぶテストの書き換えになる。使える名前は `state`(評価点の状態)、`dlambda`、`state_n`、`inc` で、それ以外はクラス定義時にエラーにする。
- **近似を含むブロック**: 手書き NR が explicit 状態を凍結して扱うモデルでは、手導出と自動微分が一致しないブロックがある(Continuous YU の 3D / PS の θ / β / R 行)。`approx=` で許容値を、`note=` で理由を宣言する。理由の無い `approx` はエラーにする。
- **照合**: `JacobianChecker(model).check_hand_blocks(results)` が、宣言済みブロックを自動微分と履歴全体で比べる。最悪誤差をブロックごとに集計するので、1 つの誤りが他に隠れない。未導出のブロックの一覧と導出済みの数も返す。何も照合しなかった場合(宣言なし、塑性ステップなし)は合格ではなくエラーにする。
- **組み立て**: `JacobianChecker(model).hand(result)` は、宣言済みブロックを手導出、残りを自動微分で埋めた `JacobianBlocks` を返す。`.source[row][col]` に出所が残る。
- **対象外**: 手書き NR のループ本体と、その中の `calc_jacobian` の組み立ては汎用化していない。組み立て順も Fortran に移植する対象で、`project_theta` のようなモデル固有の処理もあるためである。

誤差の指標は、ブロックごとに `max|手導出 − 自動微分| / max(max|自動微分|, 1)` である。

**適用状況**

| モデル | ブロック | 近似の宣言 |
|---|---|---|
| 公開 `YUKinematicPS` | 16 / 16 | なし |
| internal `YUKinematicPS` | 16 / 16 | なし |
| internal `ContinuousYUKinematic3D` / `PS` | 25 / 25 | θ / β / R 行の 15 ブロック(実測の最悪 4.3e-3 に対し 2e-2) |
| internal `ContinuousYUKinematic1D` | 25 / 25 | なし(従来のテストは 2e-2 を許していたが、実測は 1e-10 以内) |

これらのモデルのテストにあった「どのメソッドがどのブロックか」の対応表(3 箇所)は、宣言に置き換えて削除した。

### 4.9 検証層の役割

| 役割 | 中身 | 寿命 |
|---|---|---|
| 内省 | `JacobianBlocks.part`、レイアウト、自動微分の接線 | 恒久 |
| テスト補助 | Crosscheck、`TangentChecker`、履歴生成、手導出ブロックの照合 | テストと同じ |
| 導出の記録 | `verified_against_fortran`、手導出ブロックの宣言 | モデルと同じ |

- **内省は integrator から取れる。** `integrator.jacobian(result)` が `JacobianBlocks` を返す(段 3 で実施)。行列の計算は `simulation/jacobian.py` に移した。`JacobianChecker.compute(result)` は同じものを返し、`.part[row][col]` の形は変えていない。
- **`JacobianChecker.compare` は 2 つの解を同じ自動微分残差で比べるもの**で、手導出と自動微分の比較ではない。後者は `check_hand_blocks` が担う(段 3 で追加)。
- **モジュールの移動は今回しない**(方針 5)。`manforge.verification` のパスは残し、中身の役割だけを整理する。

### 4.10 パラメータ宣言と `dimension`
`param_names` から属性を設定する処理を基底に持たせ、次元はクラス属性で固定する(段 4 で実施)。

```python
class ChabocheKinematic(MaterialModel):
    param_names = ["E", "nu", "sig_y0", "C", "gamma", "Qinf", "b"]
    ...

class ChabocheKinematic3D(ChabocheKinematic):
    dimension = SOLID_3D

ChabocheKinematic3D(E=..., nu=..., sig_y0=..., C=..., gamma=..., Qinf=..., b=...)
```

**パラメータ**

- **`param_names` が唯一の宣言である。** 基底の `__init__(*, dimension=None, **params)` が、その名前のキーワード引数を受け取り、同名の属性に入れる。モデルは `__init__` を書かない。
- **全パラメータが必須で、既定値は持てない。** UMAT の `PROPS` に既定値は無く、`service.run_simulation` も全キーを必須にしていた。コンストラクタだけが既定値を持つと、入口ごとに規則が違ってしまう。後から足したパラメータに既定値を付けて既存の呼び出しを黙って通す、という使い方もできなくなる。`ParamMeta.default` は入力フォームの初期値で、意味が違うので残す。
- **不足と余分は名前を挙げて `TypeError` にする。** 綴り違いは両方に出る(`missing ['b']; unexpected ['B']`)。
- **`param_names` はクラス定義時に検査する**(str の列、重複なし、属性名として有効、状態変数や `MaterialModel` の属性と衝突しない)。
- **パラメータ以外の属性にフックは足さない。** 定数はクラス属性、パラメータから導く値は `property` に書く。パラメータではない設定が要るモデルだけ `__init__` を override し、残りを `super().__init__(**params)` に渡す(`J2IsotropicDamageUnilateral` の `c`)。

**次元**

- **クラス本体に `dimension = X` と書いたクラスと、その子クラスは固定である。** 別の `dimension=` を渡すと `TypeError` になる。同じ値は通す。
- **書いていないクラスは、構築時の `dimension=` で選ぶ**(省略時は 3D)。`J2Isotropic(dimension=PLANE_STRAIN, ...)` はそのまま使える。
- 同じ値を通すので、モデルの再構築は `type(m)(dimension=m.dimension, **m.params)` の 1 形式になる。`fitting/objective.py` の `inspect.signature` による判定は削除した。

**実施結果**

- manforge: モデルの `__init__` 28 個を削除した(残るのは `J2IsotropicDamageUnilateral` の 1 個)。既定値 `p_D=0.0`、`h=0.2` を削除した。
- manforge-internal: `__init__` 32 個、約 500 行を削除した。`ContinuousYUKinematic` の既定値(`H`、`k_0`、`k_mc`、`eps_f`、`damage_law`)を削除した。
- 数値は変わらない。公開 66 ケース、manforge-internal 57 ケースとも、段 3 の結果とビット単位で一致した。
- **`param_names` の誤りが 1 件見つかった。** `ChabocheKinematicMulti` と `TeranishiKinematic` で、カンマが抜けて `"gamma3" "Qinf1"` が 1 つの名前になり、`C3` も抜けていた。コンストラクタが別に引数を持っていたので構築はでき、`model.params` を読むまで分からなかった。宣言が 1 箇所になったので、今後は構築時に分かる。

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
| 2 | `Increment`、フック署名、`elastic_tangent`、`isotropic_stiffness`、結果が `increment` を持つ | フック 29 定義の書き換え、剛性の override 4 つを削除 |
| 3 | 手導出ブロックの宣言、`integrator.jacobian`、照合の補助 | Continuous YU の 3 次元と `YUKinematicPS` に宣言を付け、テストの対応表を削除 |
| 4 | パラメータ宣言と `dimension` | コンストラクタ 32 個の削除、既定値に頼っていた呼び出しの明示、`param_names` の誤り 1 件の修正 |
| 5 | 入口と命名の整理 | なし(fitting 未使用) |
| 6 | ドキュメント基盤、公開範囲の絞り込み | import の書き換え |

- **段 2 が最大である。** フック定義は manforge のモデルに 20、manforge のテスト内モデルに約 65、manforge-internal に 29 ある。
- **段 2 の合格条件は数値の完全一致とする。** 署名の変更だけなので、既存モデルの結果は変わらないはずである。`v0.1.0` で代表的な履歴の結果を保存し、変更後とビット単位で比べる。実施結果: 公開モデル 66 ケース、manforge-internal の全モデル 57 ケース(応力・ひずみ・全状態量・接線・反復数)が完全一致した。段 1 も同じ方法で `v0.1.0` と完全一致することを確認した。
- **段 3 は段 2 の後に行う。** 手導出ブロックの署名が `inc` を使うためである。
- **段 4 は順序を入れ替えられる。** 段 2 より先に行えば、書き換えるコンストラクタが減る。

## 6. 未決事項

1. ~~文脈オブジェクトの名前~~ → `Increment`、引数名 `inc` に決定(4.1)。
2. ~~無損傷剛性の取り方~~ → `isotropic_stiffness()` に決定(4.4)。
3. ~~パラメータの既定値~~ → 持たない。全パラメータ必須に決定(4.10)。
4. ~~派生属性の初期化~~ → フックは足さない。定数はクラス属性、導出値は `property`、パラメータではない設定だけ `__init__` を override する(4.10)。調べた時点で、`__init__` でパラメータから計算している属性は無かった(YU の `P` / `T` は `dimension` から取る形になっていた)。
5. ~~次元を固定するクラスの判定~~ → クラス本体に `dimension` を書いたクラスとその子は固定、に決定(4.10)。
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
