# API 再設計 設計メモ

- 状態: 2026-11-18 に全体を合意。移行計画(5 章の段 0〜6)と未決事項(6 章)はすべて実施・決定済み。その後の全体レビューで見つかった点は 4.14、そこから決めた検査の形の統一は 4.15、名前の変更は 4.16、残っていた小さな不統一の解消は 4.17、モデルの比較は 4.18、モデルの置き方は 4.19、共通の検査は 4.20、近似になるステップの宣言は 4.21、UMAT のビルドとテストの足場は 4.22、UMAT の宣言は 4.23 にある。
- 基準点: returnmap `v0.1.0`(`10c586c`)、returnmap-internal `99bc291`。本文中の `file:line` はこの基準点のもの。
- 名前: このメモを書いた時点のパッケージ名は `manforge` だった。2026-11-20 に `returnmap` に改名した(4.16)。本文のパッケージ名と import パスは新しい名前に置き換えてある。基準点のタグ `v0.1.0` の中身は `manforge` のままである。
- 対象読者: returnmap と returnmap-internal の両方を触る人。

## 1. 背景

機能を後方互換のまま足し続けた結果、API の一貫性が落ちている。調査で確認した症状は次のとおり。

- **`=None` が「省略可」ではなく「古い呼び出し経路では渡ってこない」を意味している。** `strain_inc=None` が 26 箇所、`stress_trial=None` が 15 箇所ある。`stress_update` 経由では両方とも常に渡される。
- **試行状態が状態(複数フィールド)になったのに、下流のフックはベクトル 1 本しか受け取らない。** `elastic_predictor`(#101)は σ 以外の試行値も返せるが、`elastic_check` / `state_residual` / `user_defined_return_mapping` が受け取るのは `stress_trial` だけである。
- **同じ概念に層ごとに別のスイッチ・別の名前がある。** 非収束の扱い、`tol`、状態履歴の要求方法、`dimension` の渡し方など。
- **検証層に性格の違うものが混ざっている。** 内省(`JacobianBlocks.part`)、テスト補助(Crosscheck)、導出の記録(`verified_against_fortran`)。

根拠の一覧は付録 A にある。

## 2. 方針
1. **0.x は破壊的変更可。** returnmap と returnmap-internal を同時に直す。互換シムは置かない。
2. **UMAT に入る量には必ず手書きスロットがあり、自動微分はそれを照合する参照側に立つ。** 自動微分だけで得られる量を成果物(Fortran UMAT)の前提にしない。
3. **フックには引数を足し続けない。** ステップの文脈を 1 つのオブジェクトにして渡す。
4. **設計の重点は「コアが現在表現できること全体」で決める。** returnmap-internal での使用件数は移行コストの見積もりにだけ使う。
5. **公開 import パスは当面いじらない。** 深いモジュールパス(`returnmap.core.state` など)も公開扱いとする。絞り込みとモジュールの移動は別の機会に行う。→ 段 6a で、公開する import 元をパッケージ単位に決めた(4.12)。モジュールの移動はしていない。

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
| テスト補助 | `crosscheck`、`check_tangent`、履歴生成、手導出ブロックの照合 | テストと同じ |
| 導出の記録 | `verified_against_fortran`、手導出ブロックの宣言 | モデルと同じ |

- **内省は integrator から取れる。** `integrator.jacobian(result)` が `JacobianBlocks` を返す(段 3 で実施)。行列の計算は `simulation/jacobian.py` に移した。`.part[row][col]` の形は変えていない。同じものを返していた `JacobianChecker.compute(result)` は 4.15 で削除した。
- **`compare_jacobians`(当初は `JacobianChecker.compare`)は 2 つの解を同じ自動微分残差で比べるもの**で、手導出と自動微分の比較ではない。後者は `check_hand_blocks` が担う(段 3 で追加)。
- **モジュールの移動は今回しない**(方針 5)。`returnmap.verification` のパスは残し、中身の役割だけを整理する。

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

- returnmap: モデルの `__init__` 28 個を削除した(残るのは `J2IsotropicDamageUnilateral` の 1 個)。既定値 `p_D=0.0`、`h=0.2` を削除した。
- returnmap-internal: `__init__` 32 個、約 500 行を削除した。`ContinuousYUKinematic` の既定値(`H`、`k_0`、`k_mc`、`eps_f`、`damage_law`)を削除した。
- 数値は変わらない。公開 66 ケース、returnmap-internal 57 ケースとも、段 3 の結果とビット単位で一致した。
- **`param_names` の誤りが 1 件見つかった。** `ChabocheKinematicMulti` と `TeranishiKinematic` で、カンマが抜けて `"gamma3" "Qinf1"` が 1 つの名前になり、`C3` も抜けていた。コンストラクタが別に引数を持っていたので構築はでき、`model.params` を読むまで分からなかった。宣言が 1 箇所になったので、今後は構築時に分かる。

### 4.11 入口と命名の整理
段 5 で実施した。

- **`fit_params`**: `fit_params(model, targets, space, *, driver_factory=...)` の 1 形式にした。`targets` は `FitTarget` の列、`space` は `ParameterSpace` だけを受け取る。dict の省略形は明示的なコンストラクタに移した(実験カーブ 1 本は `FitTarget.stress_strain(strain, stress)`、パラメータは `ParameterSpace(free=, fixed=)`)。`coerce_targets`、`ParameterSpace.coerce`、`ExpData`、`FitConfig` は削除した。旧形式を渡すと、移行先を示す `TypeError` になる。
- **モデルの再構築**: `MaterialModel.with_params(**params)` を追加した。浅いコピーを作ってパラメータだけ差し替えるので、次元も、パラメータではない設定(`J2IsotropicDamageUnilateral` の `c`)も引き継がれ、`__init__` の形に依存しない。fitting はこれを使う。段 4 までは `type(model)(dimension=..., **params)` で作り直していて、設定が黙って既定値に戻っていた。導出値をインスタンスにキャッシュするモデルではコピーに古い値が残るので、導出値は `property` に書く、という 4.10 の規則が前提になる。
- **`nr_jacobian`**: 既定値を全層で `"exact"` に揃え、型を `Literal["exact", "broyden"]`(`_typing.NRJacobian`)にした。`service.run_simulation` の既定が `"broyden"` から変わるので、速度を優先する配備側は明示する必要がある。
- **`tol`**: 1 つの対象が 1 つの意味しか持たない所は `tol` のままにした(integrator と driver は自分の NR、`fit_params` は最適化器)。checker の合否は `tol` に揃えた(`JacobianChecker` と `check_bindings` の `rtol` を改名。`check_hand_blocks(tol=None)` は checker の `tol` を既定に使う)。2 つの意味が同居していた `CrosscheckStressDriver` だけ、driver に渡す分を `driver_tol` / `driver_rtol` / `driver_max_iter` に改名した。全層を役割名(`nr_tol`、`outer_tol` など)にする案は採らなかった。
- **状態履歴**: 要求そのものをなくした。`DriverResult` は全ステップの結果を持っているので、`result.history("ep")` で任意の状態量を後から取れる。`collect_state` 引数と `DriverResult.fields` は削除した(`FieldType` の指定を読む側は無かった)。fitting も同じ経路を使う。`service.run_simulation` の `include_fields` は、返す出力に何を入れるかの指定なので残した。当初の案は「名前の列に統一」だったが、要求と読み出しの 2 段階が残るので、こちらにした。
- **残骸の削除**(段 1 で実施): `mandel_factors_jnp` / `identity_jnp`、`returnmap.autodiff` パッケージ、`utils/tensor.py`、`utils/voigt.py`、`check_tangent`、`CrosscheckCaseResult` の `py_*` / `f_*`、`__all__` の `_PythonIntegratorBase`。当初は `autodiff` のうち `operators.py` と `backend.check_float64` だけを挙げていたが、それを除くと使う側が残らないためパッケージごと削除した。`utils/voigt.py` も同じ理由で参照が無くなったので削除した。

### 4.12 公開する import パスとドキュメント
段 6 で決めた。段 6a(import パス)と段 6b(ドキュメント基盤)に分けて進める。

**import パス(段 6a)**

同じ名前が 2〜3 通りのパスから取れていた(`Implicit` は `returnmap.core` と `returnmap.core.state`、`FortranModule` は `returnmap.simulation.integrator` と `returnmap.verification`)。README、例、returnmap-internal のあいだでも書き方が揃っていなかった。

- **公開の import 元はパッケージ単位の 7 つとする。** `returnmap.core` / `.models` / `.simulation` / `.fitting` / `.verification` / `.service` / `.utils`。
- **どの名前も 1 つのパッケージの `__all__` にだけ載せる。** 例外は置かない。`ParamMeta` は `returnmap.service` からも取れていたが、モデル上で宣言するものなので `returnmap.core` だけにした。配備側(returnmap-api)は `ModelDescriptor.param_meta` から値を読むので、クラス自体は通常要らない。要る場合は `returnmap.core` から取る。
- **`returnmap.core` はモデル定義に要るもの全部を持つ。** `MaterialModel`、`ParamMeta`、`hand_block`、`verified_against_fortran` を `core.material` から引き上げた。
- **`returnmap.verification` は検査だけを持つ。** 検査される側の宣言は、定義されている場所に置く。`FortranModule` と `JacobianBlocks` は `returnmap.simulation`、`verified_against_fortran` と `FortranBinding` は `returnmap.core`。`verification` からの再エクスポートは削除した(互換シムは置かない)。
- **`returnmap.utils` は smooth 近似と主値分解の関数を公開する。** これまで `__init__` は空で、`returnmap.utils.smooth` から取るしかなかった。
- **サブモジュールのパスは動き続けるが、文書には載せない。** モジュールの改名(`_state.py` など)で強制する案は採らなかった。約 30 モジュールの改名と、テスト配置の規約(`src` と 1:1)の見直しが要るためである。`src/` 内部の import と、非公開ヘルパを直接試すテストは、引き続きサブモジュールのパスを使う。
- 規則は `tests/unit/test_public_api.py` で確認する(載せた名前が存在する、二重に載っていない、各パッケージが単独で import できる)。

**ドキュメント(段 6b)**

- **ツールは Sphinx + MyST。** docstring は英語の NumPy 形式で、`:class:` などの相互参照が約 290 箇所ある。autodoc + napoleon ならそのままリンクになる。ガイドは Markdown(MyST)で書く。
- **ガイドは日本語、API リファレンスは英語。** 今ある資産(日本語の README、英語の docstring)をそのまま使う。
- **まずローカルビルドのみ。** `make docs` で HTML を作れる状態までを範囲とし、警告をエラー扱いにしてリンク切れを検出する。公開先(GitHub Pages など)は内容を確認してから別途決める。

実施結果(段 6b):

- **README を 600 行から 80 行に縮めた。** 概要、インストール、クイックスタート、`docs/` への案内だけを持つ。使い方の説明は `docs/` のガイド 9 ページに移した。`docs/index.md` は README の概要部分を include するので、同じ内容を 2 箇所に書かない。ガイドは素の Markdown なので、GitHub 上でもそのまま読める。
- **API リファレンスは公開パッケージごとに 1 ページ。** `automodule` が各パッケージの `__all__` に従うので、名前は import 元のパス(`returnmap.core.MaterialModel`)で 1 回だけ載る。定数(`NTENS`、`SOLID_3D` など)は docstring を持てないので、`docs/reference/*.md` に説明を書いた。
- **`make docs` は警告をエラー扱いにする。** 検出するもの: 解決できない相互参照、`docs/` 内のリンク切れ、`__all__` に載っているのにリファレンスに項目が無い名前(`docs/conf.py` のビルド後チェック)。最後のチェックで、docstring の無かった `YUKinematic1D` と、説明の無かった `StressDimension` の演算子 6 つが見つかった。
- **docstring の相互参照を公開パスに直した**(61 箇所)。非公開の名前への参照は literal にした。
- **私有の基底クラスを持つクラスは、継承したメソッドを明示して載せる。** `PythonIntegrator` など 3 つと Crosscheck ドライバ 2 つは、メソッドが `_PythonIntegratorBase` / `_CrosscheckDriverBase` にあり、そのままではリファレンスに `stress_update` すら出なかった。
- `StateField` を `returnmap.core` に追加した。モデルが `self.ep(...)` / `self.stress.update(...)` として触る型で、説明の置き場が無かった。

### 4.13 手書きの解の検査と crosscheck
移行計画(段 0〜6)の後に、残っていた未決事項 6 と 7 を決めて実施した。

**手書きの解の検査(未決事項 6)**

手書きの NR を持つモデルには、残差ベクトルを返す `calc_residual` がある。`state_residual`(自動微分側)と同じ代数をもう一度手で書いたもので、Fortran に移植される。この 2 つが食い違った前例がある。検査は「手書き経路と自動微分経路の履歴が一致する」ことだけで、食い違ったときにどの行が原因かは分からなかった。

- **残差は宣言しない。** 手導出ブロックと同じ形(`@hand_residual` で宣言し、値を照合する)は採らなかった。残差は根が同じなら行をスケールしてよいので、解の外で値まで一致させる照合は正しい書き換えを誤りと判定する。閉形式解(`J2Isotropic3D`)には残差ベクトルがそもそも無い。
- **解を自動微分側の残差に代入する。** `JacobianChecker.check_solution(results)` は、手書き経路が返した解を、そのステップの自動微分側の残差に代入する。手書きの行が食い違っていれば、その行だけが 0 にならない。explicit 状態は、報告された値を `update_state` の結果と比べる。
- **各ステップを個別に判定する。** そのステップの開始状態は手書き経路自身のものなので、誤差の累積に左右されない。
- **誤差の測り方。** 行ごとに `max|R| / (行を構成する項の大きさ + slack / tol)`。項の大きさは `max(|∂R/∂x|·|x|)`(成分ごとの後退誤差で、行をスケールしても変わらない)。`slack` はソルバ自身が報告した残差ノルムの 10 倍で、ソルバが止まった許容値を誤りとは判定しない(反復していない閉形式解には与えない)。実測では、閉形式解が 1e-16、手書き NR が 1e-12、Continuous YU(手書き NR の許容値が緩い)が 1e-9 で、既定の `tol=1e-8` に対して 1 桁以上の余裕がある。
- **動かなかった状態は合格にしない。** 検査したどのステップでも開始時の値から動かなかった explicit 状態は `not_exercised` に載る。その更新式は検査されていない。判定はその状態自身の値だけで行う(当初は未知数の最大値と比べていて、Pa 単位では動いている塑性ひずみを「動いていない」と誤判定した)。
- **対象は解だけ。** 収束しなかった結果は `ValueError`、分割されたステップ(`n_substeps > 1`)は対象外として数える。

**crosscheck(未決事項 7)**

2 つの integrator を同じ載荷履歴で進めて比べる、という 1 つのことに公開名が 7 つあった(`Comparator`、`CaseResult`、`ComparisonResult`、`CrosscheckCaseResult`、`CrosscheckResult`、`CrosscheckStrainDriver`、`CrosscheckStressDriver`)。`Comparator` の実装は Crosscheck だけだった。混合制御用は無かった。returnmap-internal の Fortran 突合テストは使っておらず、同じループを手で書いていた。

- **`crosscheck(integrator_a, integrator_b, load, *, driver_factory=...)` の 1 つにした。** 公開名は `crosscheck`、`CrosscheckResult`、`CrosscheckStep` の 3 つ。driver は載荷履歴から決まり(ひずみ履歴 → `StrainDriver`、応力履歴 → `StressDriver`)、それ以外は `driver_factory` で渡す(`fit_params` と同じ形)。混合制御も使える。driver 自身の設定は factory が持つので、`driver_tol` の受け渡しは無くなった。
- **`FortranIntegrator(parse_umat_failure=...)` を足した。** UMAT 自身の収束フラグを読む。これが無いと UMAT 側の非収束は見えず、収束しなかった反復値が解として比較される。internal が Crosscheck を使えなかった理由はこれだった。
- **`lockstep=True` を足した。** 既定では 2 つの経路を独立に進めるので、最初の食い違いの後はすべてずれ、長い履歴では正しい実装同士でも丸め誤差が累積する。lock-step では参照側だけを進め、各ステップで候補側に参照側の入力(ステップ開始時の応力と状態、ひずみ増分)を渡して出力を比べる。「同じ状態から同じ増分で一致するか」という、UMAT の移植が満たすべき条件そのものになる。internal のテストには「長い履歴には lock-step のハーネスが要る」というコメントがあった。
- **誤差の測り方を変えた。** これまでは応力が `|Δ| / (|σ| + 1)`(MPa を仮定した加算項)、状態量と接線が成分ごとの純粋な相対誤差で、0 に近い成分で丸め誤差が大きな相対誤差になった。配列全体で `max|b − a|` を取り、参照側がそれまでに到達した最大値に対する相対値にした。単位に依らない。
- **収束しなかったステップは合格にしない。**
- **internal の「両方を進めて比べる」テスト 10 本を `crosscheck` に置き換えた。** 手書きのループと `assert_allclose` の並びが消えた。Continuous YU の反転履歴は独立に進めると `q` が 1.5e-8 までずれるので、lock-step にした(6.5e-15)。

### 4.14 全体レビュー
未決事項がすべて決まった後、全体をレビューした(構造の調査と、文書と実装の突き合わせ、再設計で書き換えたコードの不具合探し)。

**誤った結果を黙って返していたもの(修正済み)**

- 手書きのリターンマッピングで Δλ < 0 と非有限の値が弾かれず、`run_simulation` が `converged=True` のまま返していた。自動微分経路と同じ条件で確かめるようにした。
- `StressDriver` が載荷の形を検査せず、`(N,)` の履歴が全成分にブロードキャストされて走っていた。
- `check_solution` が、Pa 単位では動いている塑性ひずみを「動いていない」と見なした。閾値が、ひずみを応力の大きさと比べていた。**単位に依らないと主張する判定は、同じ量どうしを比べていることを確認する。**
- `check_solution` が、非収束の反復値を合格にし、分割されたステップを誤って不合格にした。
- NaN は比較に必ず負けるので、「最悪値の更新」から漏れて合格になっていた(`check_hand_blocks`、`check_solution`、`crosscheck` の要約)。
- 固定した次元を子クラスで付け替えられた。
- `JacobianChecker.compare` が Fortran の結果(Δλ が NaN)と比べられず、成分ごとの相対誤差のため丸め誤差で不合格になった。

**文書と実装の食い違い(修正済み)**

- 「スカラー NR とベクトル NR の 2 経路」は存在しなかった。NR の未知数は常に `[σ, Δλ, implicit 状態]` で、explicit 状態は `update_state` で求める。README、ガイド、モデルの docstring を直した。
- 削除済みの名前や経路を説明する docstring が残っていた(`implicit_stress`、`method="auto"`、属性アクセスできる `State`、存在しないテストへのポインタ)。`verified_against_fortran(test=...)` のポインタは、存在することをテストで確認するようにした。

**片付け**

- バージョンを 0.2.0 にし、`CHANGELOG.md` に 0.1.0 からの移行の対応表をまとめた。
- 作業用スクリプト(`_scratch_yu*.py`)と未使用のコードを削除した。

**API の不統一(4.15 で解消)**

- 検査ごとに誤差の測り方と結果の形が違った(`TangentChecker`、`check_hand_blocks`、`check_bindings`)。
- driver の選び方が `crosscheck` と `fit_params` で違った。
- `FitResult.params` に補助変数が混ざり、fit 結果からモデルを直接作れなかった。
- `ReturnMappingResult` の `converged` の既定値が `True` だった。

**残っているもの**

- Python 3.10 / 3.11 では動作を確認していない(確認済みは 3.12 と 3.13)。
- 小さな不統一: `run_simulation(model_cls, params)` は次元とパラメータ以外の設定を渡せない。`DriverStep.i` と `CrosscheckStep.index`。`FitResult.n_iter` の中身は評価回数。→ 4.17 で解消した。
- Python の integrator が 3 クラスある(違いは経路の選び方だけ)。これはそのままにしている。

### 4.15 検査の形と誤差の測り方

全体レビュー(4.14)で見つかった不具合のうち 3 件は、検査ごとに誤差の測り方が違うことが原因だった。検査の入口と結果の形も揃っていなかった(クラスが 2 つ、関数が 2 つ。結果は `*Result`、`*Report`、素の dict)。次のとおり 1 つの形に揃えた。

**検査は関数で、報告を返す。**

```python
from returnmap.verification import (
    check_hand_blocks, check_solution, check_tangent,
    crosscheck, compare_jacobians, check_bindings,
)

report = check_hand_blocks(model, results)
report = check_solution(model, results)
report = check_tangent(integrator, strain_inc, stress_n, state_n)
assert report.passed, report.summary()
```

- `JacobianChecker` と `TangentChecker` は削除した。クラスが持っていた状態は `model` と `tol` だけで、検査のたびに作り直されていた。`check_solution` はヤコビアンの検査ではないのに `JacobianChecker` にあり、`compute` は `integrator.jacobian(result)` と同じものを返していた。
- 行列の取得は検査から分けた: 自動微分側は `integrator.jacobian(result)`、手導出側は `hand_jacobian(model, result)`(どちらも `JacobianBlocks`。後者は `returnmap.simulation` に置いた)。
- `check_tangent` の引数は `stress_update(strain_inc, stress_n, state_n)` と同じ順にした(`TangentChecker.check(stress, state, strain_inc)` は逆だった)。
- 段 1 で関数 `check_tangent` を削除してクラスに寄せたのとは逆の向きになる。当時は「同じことをする入口が 2 つある」ことを解消するのが目的で、どちらに寄せるかは既存の `JacobianChecker` に合わせた。検査が 6 つに増え、うち 2 つ(`crosscheck`、`check_bindings`)は関数でしか書けない形だったので、関数の側に揃えた。
- 命名: 計算の結果は `*Result`(`StressUpdateResult`、`DriverResult`、`FitResult`)、検査の結果は `*Report`。`JacobianComparisonResult` / `TangentCheckResult` / `CrosscheckResult` を `*Report` に改名し、`check_bindings` は素の dict をやめて `BindingReport` を返す。どの報告も `.passed` と `.summary()` を持つ。

**誤差の測り方は 1 つ。**

```
誤差 = max|候補 − 参照| / max|参照|      合格 ⇔ 誤差 <= tol
```

- 配列(ブロック、接線、応力ベクトル)全体で測る。成分ごとに割ると、厳密には 0 の成分に乗った丸め誤差が 100 % の誤差になる。それを避けるために分母に足していた定数(`TangentChecker` の `+ 1e-2`、`check_hand_blocks` の `max(·, 1)`、`check_bindings` の `+ 1e-14`)は単位を持つので、モデルを書いた単位で合否が変わっていた。
- 実装は `verification/_measure.py` の 1 箇所。NaN と `inf − inf` は最悪の誤差として扱う。
- 複数の点にまたがる検査は、差も参照の大きさも、検査した点全体で最大を取る(`check_hand_blocks` は履歴全体、`crosscheck` はそのステップまで)。あるステップで量がたまたま 0 に近くても、丸め誤差が大きな誤差にはならない。
- `check_hand_blocks` の旧い測り方(各ステップで `max|Δ| / max(max|ad|, 1)`)は、値が 1 より小さいブロックの誤差を小さく見せていた。公開の YU モデル(平面応力)では、`[beta][theta]` などで最大 1000 倍ほど小さく出ていた(どれも丸め誤差の範囲で、合否は変わらない)。値が小さいブロックだけでできた行は、実質的に絶対値で測られていたことになる。意図した近似(`approx=`)の宣言値は、この測り方で測り直す必要がある: 0 と置いて落としたブロックは 1(100 %)と読める。
- **厳密に 0 のブロック**には、割る相手が無い。自動微分は丸め誤差(1e-13 など)を返し、手導出は 0 を返すので、ブロック自身に対する比は 1 になる。そこで、ブロックの大きさを「その行と列から決まる大きさ」(行を構成する項の大きさ `Σ|∂R/∂x|·|x|` ÷ その列の未知数の大きさ。どちらも履歴全体の最大)と比べ、12 桁以上小さければ 0 と見なして、その大きさに対して測る。`crosscheck` が、0 から動いていない状態量を扱うのと同じ考え方で、定数も同じ(`NOISE_FLOOR = 1e-12`)。
- 検討して採らなかった案: すべてのブロックを「行と列から決まる大きさ」に対して測る(行と列をスケールしたヤコビアンでの誤差)。落とした項が行に与える影響を測るには適しているが、ブロック自身より大きい尺度で割ることになり、導出の誤りの検出が鈍る(小さいブロックの 1 % の誤りが通る。符号の誤りが 2 と読めない)。検査の目的は導出の確認なので、ブロック自身に対する比を採った。
- `check_solution` の残差の行だけは参照配列が無いので、行を構成する項の大きさに対する後退誤差で測る(4.13)。explicit 状態は上の測り方に従う。

**driver を省略できる入口は同じ規則で決める。** `fit_params` も `driver_factory` を省略でき、`crosscheck` と同じ規則(ひずみ履歴 → `StrainDriver`、応力履歴 → `StressDriver`)で決まる。規則は `simulation/driver.py` の `default_driver_for` の 1 箇所にある。`run_simulation(control=...)` は文字列で受け取る service 層の契約なので変えていない。

**fit 結果からモデルを作れる。** `FitResult.model_params` はモデルのパラメータだけ、`FitResult.model` はテンプレートにそれを入れたモデル(クラス、次元、パラメータ以外の設定を引き継ぐ)。`FitResult.params` は探索空間のすべての名前を持つままにした(比で探索したときの比の値を読むのに要る)。

**`ReturnMappingResult` に既定値を置かない。** `converged=True` が既定値だと、反復を使い切った手書きの NR が `converged` を書き忘れても成功になる。パラメータに既定値を置かないのと同じ理由で、`n_iterations`、`residual_history`、`converged` を必須にした。

### 4.16 名前を `returnmap` に変える

`manforge` という名前からは、何をする道具かが読み取れなかった。改名にあたって、道具の位置づけを整理した。

**全モデルが UMAT まで行くわけではない。** モデル定義 15 ファイル(公開と非公開の合計)のうち、手書きのリターンマッピングまで持つのは 4 ファイルで、残りは Python のモデルだけである。モデルごとに到達点を選ぶ。

| 到達点 | 中身 | 検査 |
|---|---|---|
| A. 比較・フィット用 | Python のモデルだけ(自動微分で解く) | 収束、`check_tangent` |
| B. 手書きのアルゴリズム | `user_defined_return_mapping`(と接線) | `check_solution`、`crosscheck` |
| C. UMAT | Fortran | `check_bindings`、`crosscheck(Python, UMAT)` |

- ヤコビアンの手導出(`@hand_block`)は、B で連立のまま NR で解く方法を選んだときだけ要る部品で、必須の段ではない。後退 Euler の方程式をスカラー方程式に縮約する方法(radial return 系)は根が同じなので、`check_solution` だけで検査できる。
- cutting plane は後退 Euler の根とは別の点に収束するので、`check_solution` の対象にならない。汎用の解法としては作らない(今のモデル定義は残差の形で、発展則の速度形を持たない)。カジュアルなモデルの解法は自動微分の NR で足りる。
- 到達点とは別に、公開範囲がある: 非公開リポジトリに置くだけ / `MODELS` に載せて API に出す / 本体に参照実装として入れる。非公開のモデルを本体に合流させることは前提にしない。

**名前。** A・B・C のどれも、中身はリターンマッピングを解いて結果を見ることなので、`returnmap` にした。`umat` を含む名前(`umatproof` など)は、到達点 A が多数派である実態より狭い。`matforge`(元の名前の誤植を直した形)は、「mat」が material / matrix / MATLAB のどれとも読め、「forge」が生成する道具と読めるので採らなかった。

**変えたもの。** 配布名、import パス(`returnmap.*`)、CLI(`returnmap build`)、Docker イメージ名(`returnmap-fortran`)、f2py の試験用モジュール名、文書。非公開リポジトリは `returnmap-internal`(import は `returnmap_internal`)。機能と計算結果は変わらない。

### 4.17 残っていた小さな不統一

全体レビュー(4.14)の「残っているもの」に挙げた不統一を解消した。どれも、同じものを指す名前や作り方が場所によって違っていたものである。

**service もテンプレートのモデルを受け取る。** `run_simulation(model_cls, params)` はクラスを受け取り、`model_cls(**params)` で構築していた。これでは、次元を固定していないクラスは既定の次元でしか走らず、パラメータ以外の設定(`J2IsotropicDamageUnilateral` の `c`)は既定値にしかならない。`fit_params` が 4.11 で解消したのと同じ問題である。同じ形にした:

- `run_simulation(model, params, load)` はテンプレートのモデルを受け取り、`model.with_params(**params)` で走らせる。テンプレートは書き換えない。
- `ModelRegistry.register(name, model)` はテンプレートを登録する。同じクラスを、別の名前で別の次元や設定として登録できる。`ModelDescriptor.ntens` は登録したモデルの次元から取る。
- **`params` は従来どおり `param_names` の全部を要求する。** テンプレートの値を既定値として使う形(渡した名前だけ置き換える)にすると、リクエストが名指ししていない値で計算が走る。パラメータに既定値を置かない(4.10)のと同じ理由で採らなかった。したがって、テンプレートが持つパラメータの値は構築のための仮の値である。`default_registry()` は `param_meta` の `default` を入れている。
- 非公開パッケージの `MODELS` は `dict[str, MaterialModel]` になる。

**名前を揃えた。**

| これまで | これから | 揃える相手 |
|---|---|---|
| `DriverStep.i` | `DriverStep.index` | `CrosscheckStep.index` |
| `FitResult.n_iter` | `FitResult.n_evaluations` | 中身(scipy の `nfev`)。反復回数ではない |
| `Observation(field="Stress")`、`x_field="Strain"` | `"stress"`、`"strain"` | `DriverResult.history("stress")`、状態量のキー、`control="strain"` |
| `SOLID_3D.name == "3D_SOLID"` | `"SOLID_3D"` | import する名前。他の 4 つの次元は元から一致していた |

- `Observation` の名前を揃えたのに合わせて、`DriverResult.history("strain")` を足した。`Observation.field` が受け取る名前と `history` が受け取る名前が、同じ 1 つの集合になった。`"strain"` が状態量のキーと同じ名前空間に入るので、状態量に `strain` という名前を付けるとクラス定義時に `TypeError` になるようにした(付けられると、名前で読む側で黙って全ひずみに置き換わる)。両リポジトリに該当するモデルは無い。
- 次元の `name` はエラーメッセージ(「`X` is fixed to ...」)に出るので、そのまま書ける名前にした。全次元について `returnmap.core` の同名の定数と一致することを、テストで確認している。

**`FieldHistory` のラベルを省略できるようにした。** `FieldHistory(type, name, data)` の `name` は必須の位置引数だったが、フレームワークはこれを読まない(両リポジトリの呼び出し 108 箇所は、`"Strain"`、`"eps"` などを形だけ渡していた)。`FieldHistory(type, data, *, name="")` にした。`name` をキーワード専用にしたのは、古い書き方(2 番目にラベル)がラベルを `data` として黙って受け取らず、`TypeError` になるようにするためである。

**そのままにしたもの。** Python の integrator の 3 クラス(`PythonIntegrator` / `PythonNumericalIntegrator` / `PythonAnalyticalIntegrator`)は、今回の対象にしていない。

### 4.18 モデルの比較 `compare_models`

到達点 A(4.16)のモデルは、Python のモデルを書いた時点で使い道がある: ほかの構成則と比べることである。比べたいのは応答の違い、計算コスト(収束回数、時間)、フィットの良さの 3 つで、毎回全部ではない。これまでは、モデルごとに driver を回して結果を集めるスクリプトを手で書いていた(returnmap-internal の `examples/compare_yu_variants.py`)。

```python
comparison = compare_models({"J2": j2, "AF": af, "OW": ow}, load)   # returnmap.simulation
comparison.runs["AF"].stress        # 応答(ModelRun は DriverResult と同じ読み方ができる)
print(comparison.summary())         # 1 モデル 1 行の表
```

- **載荷履歴だけで応答とコストが出る。フィットの良さは `targets=` を渡したときだけ。** 値は fitting の目的関数と同じもの(全ターゲットの残差の和)にした。フィットしたモデルでは `FitResult.residual` と一致するので、2 つの数字の関係を説明する必要がない。
- **何もフィットしない。** フィットは `fit_params` の仕事のままで、比較は渡されたモデルを走らせるだけである。
- **合否は付けない。** 計算の結果なので `ModelComparisonResult`(`*Report` ではない)。2 つの実装が一致するかを確かめるのは `crosscheck` である。
- **解けなかったモデルは行として返す。** 1 つのモデルが途中で収束しなくなっても、ほかの比較は続ける。素のモデルは `raise_on_nonconverged=False` の `PythonIntegrator` で積分する。渡された integrator が raise した場合(既定の `raise_on_nonconverged=True`、特異なヤコビアン)も同じ行にし、例外の文面を `ModelRun.error` に残す。対象にする例外は fitting の目的関数と同じ集合で、それ以外(載荷履歴の形の不一致など)は呼び出しの誤りとしてそのまま上げる。
- **反復数は塑性ステップで平均する。** 弾性ステップは 0 回なので、全ステップで平均すると降伏が遅いモデルほど安く見える。
- **時間は 1 回測るだけにした。** 繰り返して最良値を取る形にはしていない。見たいのはモデル間の数倍の差で、表の値は目安である。ひずみ制御では整合接線を求めない(遅延評価)ので、時間に含まれない。その driver が実際に必要とした計算の時間、という定義にした。
- **モデルの代わりに integrator を渡せる。** 当初の案は「名前 → モデル」だけだったが、解き方ごとのコスト(自動微分と手書き、`"exact"` と `"broyden"`、Python と UMAT)を比べるには integrator を選べる必要がある。UMAT は反復数と塑性の判定を報告しないので、その列は「不明」(`None`、表では `—`)にする。0 とは書かない。
- **driver の決め方は `fit_params`、`crosscheck` と同じ規則。** `driver_factory` は integrator を受け取るので、次元の違うモデルを 1 つの軸方向の履歴で走らせる(1D は `StrainDriver`、3D は `MixedDriver`)ことは factory の中で書ける。モデルごとの driver の指定は足していない。
- **`returnmap.simulation` に置いた。** `targets` の型(`FitTarget`)は `returnmap.fitting` のものだが、fitting は simulation の上に作られているので、import は関数の中で行う(パッケージの import の向きは変えていない)。

### 4.19 モデルは 1 ファミリ 1 ディレクトリ

使い方は 2 つある: 既存のモデルを使う・比べる(pip で入れれば足りる)ことと、独自の構成則を作って必要なら UMAT まで持っていくこと(returnmap-internal)。後者のモデルが本体に格上げされることもある。ところが、1 つのモデルに属するものが 5 箇所に散っていて(モデル本体、`tests/unit/models/`、`tests/integration/models/`、`tests/benchmarks/<model>/`、`fortran/`)、internal は別の並べ方だったので、格上げは「移動」ではなく「並べ直し」だった。

```
src/returnmap/models/<family>/
  __init__.py      # 公開するクラス
  model.py         # モデル本体(大きければ分ける)
  fortran/         # UMAT のソース(任意)
  tests/           # そのモデルのテスト(任意)
```

- **Python のモデルで止まってよい。** `fortran/` も `tests/` も任意で、到達点 A(4.16)のモデルはモデル本体だけで成立する。手導出まで進まないことは普通にある。
- **テストもモデルの隣に置く。** 「検証されたモデルである」ことは、検証がソースの近くにあって初めて示せる。そのモデルに固有の回帰テストも、そのモデルの話なので同じ場所に置く。パッケージの中に `tests/` を置くのは numpy や scipy と同じ慣習で、配布物に入れる(約 280 KB)。利用者は `pytest --pyargs returnmap.models.<family>` で、自分の環境で確かめられる。
- **ファミリのテストは、公開の import パスとそのファミリの中のものだけを import する。** これが守られていれば、ディレクトリごと動かせる。移す前のモデルのテスト 25 ファイルは、すでにこの規則を満たしていた(リポジトリのテスト用モジュールにも、ルートのフィクスチャにも依っていなかった)。
- **複数のファミリにまたがるテストは `tests/` に残す**(AF と OW の共通テスト、全参照モデルの `param_meta` の確認)。フレームワークのテストも今のまま。
- **`@verified_against_fortran(test=...)` のパスは、モデルのモジュールがあるディレクトリから見た相対パスにした。** リポジトリ直下からのパスだと、ファミリを動かすたびに全部書き直すことになる。
- **公開名は変えていない。** `from returnmap.models import ...` のまま。`returnmap.models.<family>` も import できる。
- **この段でしていないこと**: 定式化の `README.md`、共通の検査と検証の宣言、UMAT のビルドの入口(`abaqus_stubs.f90` をパッケージに入れる、リンク指定)、Fortran 用のテストの足場。次の段で足す。

### 4.20 共通の検査 `check_model` と検証の宣言

検査は 6 つの関数に分かれていて、どのモデルにどれを回すかはテストを書く人が決めていた。その結果、モデルによって検査の有無と書き方がまちまちだった(AF と OW の接線の確認は共通テストに、J2 損傷は各モデルのテストに、YU の手導出の確認は独立したファイルに)。「このモデルは検証されているか」を、1 箇所を見て答えられなかった。

```python
report = check_model(model)         # ModelReport: .passed / .summary() / report["tangent"]
```

- **何を検査するかは、モデルが宣言しているものから決まる。** 収束と接線は常に検査する。`@hand_block` があればブロックを、`user_defined_return_mapping` があれば解・軌跡・接線を検査する。宣言していないものは「not declared」で、不合格ではない — Python のモデルで止まることは、モデルの正当な到達点である(4.16 の A)。
- **載荷履歴は標準のものを持つ。** モデルの降伏ひずみに合わせた単軸の履歴と、直応力成分とせん断成分を同時に動かして向きを変える多軸の履歴。単軸だけでは、転置した外積やせん断の ×2 の誤りが見えない。降伏ひずみで測るので、モデルの単位とパラメータに依らない。
- **モデルに固有の履歴は、ファミリの `verification.py` に書く。** `check_model` は、モデルのモジュールの隣にあるその名前のモジュールを見つけて `histories(model)` を足す。登録の仕組みは置かなかった(名前の規則だけ)。同じファイルの `MODELS` が、そのファミリで検査にかけるモデルの一覧である。
- **塑性ステップが 1 つも無い履歴は不合格にする。** 何も検査せずに合格する形を残さない。これで `estimate_yield_strain` の不具合が見つかった: 降伏関数が応力以外の状態(有効応力)を見るモデルでは降伏点を見つけられず、単位を持つ値(剛性の逆数)に落ちていた。準片側損傷モデルの標準の履歴は、降伏に届いていなかった。
- **`Model.at_defaults()` を足した。** `param_meta` の目安の値でモデルを作る書き方が 5 箇所にあった(service の既定のレジストリ、テスト、非公開リポジトリの `MODELS`)。コンストラクタは全パラメータを要求するまま(4.10)で、目安の値を使うことを名指しする入口を別に置いた。
- **UMAT との突き合わせは入れていない。** コンパイルしたモジュールが要るので、ビルドの入口を整える段で足す。

**最初に回して分かったこと。** 全参照モデル 21 個のうち 19 個は通った。YU の 3D と平面応力は通らなかった: 手導出のヤコビアンと手書きの接線が、停留曲面のゲート(`smooth_heaviside(g_stag)`。`R` の成長を切り替える)を定数として扱っている。ゲートが 0 か 1 のステップでは厳密だが、収束点の `g_stag` が平滑化の幅に入ったステップでは、ゲートの微分のぶんだけ合わない(標準の多軸の履歴で、ゲートが 0.0015、0.99、0.39 のステップ。接線は最大 4.4e-2、平面応力の `[theta][beta]` ブロックはブロック自身の大きさだけずれる)。応力と状態は、そのステップでも自動微分と一致する(手書きの NR は同じ根に収束する)。既存のテストは、このステップを踏まない履歴を使っていたか、接線の許容値を 2e-2 に取っていた。扱いは 4.21。

### 4.21 近似になるステップを宣言する

意図した近似の宣言は、ブロックごとの許容値(`@hand_block(approx=...)`)だけだった。これは全ステップに効く。YU のゲートのように「切り替わるステップだけ合わない」導出にこれを使うと、`[theta][beta]` には `approx=1.0` が要り、どのステップで何が起きても通る検査になる。接線には宣言の手段そのものが無かった。

```python
def stagnation_gate_in_transition(self, state, state_n): ...      # ゲートが 0 でも 1 でもない

@hand_block("theta", "beta", approx_when="stagnation_gate_in_transition", note="...")
def calc_ft_fb(self, state, dlambda, state_n): ...

@hand_tangent(approx_when="stagnation_gate_in_transition", note="...")
def user_defined_tangent(self, new, dlambda, inc): ...
```

- **近似になるステップを、モデルのメソッドで言う。** そのステップは比較から外して数え、ほかのステップは厳密な許容値のまま検査する。許容値を決めなくて済む(4.4e-2 は今回の履歴での実測で、ゲートの微分が効く大きさは履歴しだいである)。`approx=` も付ければ、外さずにその値で判定する。
- **外した数を報告に出す。** 近似がどれだけ使われたかが見える。全ステップが条件に当たる宣言は、接線では「何も判定していない」として不合格にする。ブロックは「n ステップ中 n ステップが近似」と報告に出る。
- **条件のメソッドは名前で指定する。** 引数はブロックと同じ語彙(`state`、`dlambda`、`state_n`、`inc`)から名前で埋める。存在しない名前や知らない引数は、クラス定義時に `TypeError` になる。
- **接線の宣言 `@hand_tangent` を読むのは `check_model` だけ。** `check_tangent` と `crosscheck` は integrator を受け取るのでモデルの宣言を知らず、渡された許容値で判定する。
- **YU には宣言だけを付け、導出は変えていない**(ユーザーの判断)。標準の履歴では、塑性 80 ステップのうち 1〜3 ステップが該当する。その数が増えないことを、ファミリのテストで確認している。

### 4.22 UMAT のビルドとテストに要るものを本体が配る

独自のモデルを置くリポジトリ(returnmap-internal)は、UMAT まで行くために、本体と同じものを自前で持っていた: `abaqus_stubs.f90` の写し、f2py を直接呼ぶ Makefile、`fortran/` を `sys.path` に足す `conftest.py`、マーカーの宣言。本体の側でも、`returnmap build` はリンク指定を渡せなかったので、LAPACK を呼ぶ YU は CLI を使っていなかった。pip で入れた returnmap だけでは、UMAT まで行けなかった。

- **スタブをパッケージに入れた**(`src/returnmap/fortran/abaqus_stubs.f90`)。`returnmap build --abaqus-stubs` が一緒にコンパイルする。常に足す形にはしなかった: UMAT でないソース(ビルドの確認用の `test_basic.f90`)には要らず、何がコンパイルされるかはコマンドに書いてあるほうがよい。`returnmap stubs` はパスを出すだけのコマンドで、f2py やコンパイラを直接呼ぶ手順(ABAQUS の入口ファイルの確認など)のためにある。
- **`--link`** でリンクするライブラリを渡す。本体の Makefile は全部 `returnmap build` を呼ぶようにした(CLI が実際に使われる経路になった)。
- **pytest のプラグイン**(`pytest11` の entry point)。マーカーの登録と、ビルド先を `sys.path` に足すことだけを行う。entry point の名前はモジュール名と同じにした: 同じプラグインを `-p` や `conftest.py` の `pytest_plugins` で名指ししても、同じものとして扱われる(名前が違うと二重登録でエラーになる)。本体のルートの `conftest.py` はこの名指しだけを持つ — インストールせずに `PYTHONPATH` で動かす環境(Docker のイメージ)のためである。
- **ビルド先は `fortran/` のままにした。** 出力先を変えると、両リポジトリの手順と `.gitignore` に波及する。モデルごとの UMAT のソースはファミリのディレクトリにあるので、ルートの `fortran/` はビルド先と、ビルドそのものの確認用のソースだけを持つ。

### 4.23 UMAT を宣言する

UMAT まで行ったモデル(到達点 C)について、「どのソースをどうビルドするか」は Makefile に、「Python と突き合わせる」はモデルごとのテストに書いてあった。モデルが UMAT を持つことは、どこにも宣言されていなかった。

```python
# <family>/verification.py
UMATS = [
    Umat(YUKinematic3D, module="yu_kinematic_3d",
         sources=["fortran/yu_kinematic_3d.f90"], libraries=["lapack", "blas"]),
]
```

- **1 つの宣言をビルドと検査の両方が読む。** `returnmap build --family <package>` は宣言された UMAT を全部ビルドする(ファミリでも、ファミリを束ねるパッケージでもよい)。`check_model` は、ビルド済みの UMAT を Python のモデルと同じ載荷履歴で走らせて、全ステップで比べる。本体の Makefile の UMAT のターゲットは、宣言からビルドするだけになった。
- **未ビルドは不合格にしない。** gfortran の無い環境でも検査は通る必要がある。行は `not built` になり、ビルドのコマンドを示す。UMAT の検査を必ず走らせたい所(Docker での確認)では、`fortran` マーカーのテストが `report["UMAT"].status == "passed"` を確かめる。
- **比べる相手は `PythonIntegrator(model)`。** 手書きのリターンマッピングがあればそれと比べる(UMAT はその移植なので、宣言した近似も共有している)。無ければ自動微分と比べる。
- **ソースのパスは、モデルのモジュールがあるディレクトリからの相対パス。** `@verified_against_fortran(test=...)` のパスと同じ規則で、ファミリのディレクトリを動かしても切れない。
- **`check_bindings` は入れていない。** 対応づけた個々のサブルーチンは、入力を手で用意する必要があり、宣言から自動では回せない。
- **最初に回した結果**: J2 の 3D、YU の 3D と平面応力の UMAT は、標準の履歴(多軸の履歴と 2 % の反転繰返しを含む)で、応力・状態・接線とも 1e-15 で Python と一致した。既存のテストが使っていなかった履歴でも一致している。

## 5. 移行計画
各段で returnmap と returnmap-internal の PR を対にし、両方のテストが通る状態で終える。returnmap-internal は editable 依存で CI が無いので、returnmap 側の PR ごとに internal のテストを手元で走らせる。

| 段 | 内容 | internal への影響 |
|---|---|---|
| 0 | タグ `v0.1.0`、internal のテスト修正 | 完了 |
| 1 | 残骸の削除(#103)、非収束スイッチの一本化、`DriverResult.converged`、`StressIntegrator` の Protocol 化 | なし(未使用)。internal のテストは変更前後で同じ結果 |
| 2 | `Increment`、フック署名、`elastic_tangent`、`isotropic_stiffness`、結果が `increment` を持つ | フック 29 定義の書き換え、剛性の override 4 つを削除 |
| 3 | 手導出ブロックの宣言、`integrator.jacobian`、照合の補助 | Continuous YU の 3 次元と `YUKinematicPS` に宣言を付け、テストの対応表を削除 |
| 4 | パラメータ宣言と `dimension` | コンストラクタ 32 個の削除、既定値に頼っていた呼び出しの明示、`param_names` の誤り 1 件の修正 |
| 5 | 入口と命名の整理 | 例 1 本の `collect_state` を `history` に置き換え。`run_simulation` の既定が `"exact"` になる |
| 6a | 公開する import パスをパッケージ単位に揃える | import の書き換え |
| 6b | ドキュメント基盤(Sphinx + MyST、README からガイドへ) | なし |

- **段 2 が最大である。** フック定義は returnmap のモデルに 20、returnmap のテスト内モデルに約 65、returnmap-internal に 29 ある。
- **段 2 の合格条件は数値の完全一致とする。** 署名の変更だけなので、既存モデルの結果は変わらないはずである。`v0.1.0` で代表的な履歴の結果を保存し、変更後とビット単位で比べる。実施結果: 公開モデル 66 ケース、returnmap-internal の全モデル 57 ケース(応力・ひずみ・全状態量・接線・反復数)が完全一致した。段 1 も同じ方法で `v0.1.0` と完全一致することを確認した。
- **段 3 は段 2 の後に行う。** 手導出ブロックの署名が `inc` を使うためである。
- **段 4 は順序を入れ替えられる。** 段 2 より先に行えば、書き換えるコンストラクタが減る。

## 6. 未決事項

1. ~~文脈オブジェクトの名前~~ → `Increment`、引数名 `inc` に決定(4.1)。
2. ~~無損傷剛性の取り方~~ → `isotropic_stiffness()` に決定(4.4)。
3. ~~パラメータの既定値~~ → 持たない。全パラメータ必須に決定(4.10)。
4. ~~派生属性の初期化~~ → フックは足さない。定数はクラス属性、導出値は `property`、パラメータではない設定だけ `__init__` を override する(4.10)。調べた時点で、`__init__` でパラメータから計算している属性は無かった(YU の `P` / `T` は `dimension` から取る形になっていた)。
5. ~~次元を固定するクラスの判定~~ → クラス本体に `dimension` を書いたクラスとその子は固定、に決定(4.10)。
6. ~~手導出の残差も宣言するか~~ → 宣言しない。手書き経路の解を自動微分側の残差に代入する `check_solution` を足した(4.13)。
7. ~~Crosscheck ドライバの扱い~~ → `crosscheck` の 1 つの入口に作り直した。`FortranIntegrator` が UMAT の収束フラグを読めるようにし、lock-step の比較を足した(4.13)。
8. ~~ドキュメントのツールと言語~~ → Sphinx + MyST、ガイドは日本語で API リファレンスは英語、まずローカルビルドのみ、に決定(4.12)。

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
| returnmap-internal が import する returnmap のシンボルは 30 個 | 両リポジトリの import を集計 |
| returnmap-internal のフックで `strain_inc` を本体で使うものは 24 定義中 0 | AST で集計 |
| returnmap-internal の Fortran 突合テスト 5 本は `FortranIntegrator` / `Crosscheck*` を使わない | `tests/yu/benchmarks/test_*_vs_fortran.py` |
