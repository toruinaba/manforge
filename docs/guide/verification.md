# 検証ワークフロー

検査はどれも「関数が報告を返す」形をしている。報告は合否 `report.passed` と、その内訳の表 `report.summary()` を持つので、テストはどの検査でも同じ 1 行になる:

```python
assert report.passed, report.summary()
```

| 関数 | 比べるもの |
|---|---|
| `check_model(model)` | 下の検査のうち、そのモデルに当てはまるもの全部 |
| `check_hand_blocks(model, results)` | 手導出したヤコビアンのブロック ↔ 自動微分 |
| `check_solution(model, results)` | 手書きのリターンマッピングの解 ↔ 自動微分側の残差 |
| `check_tangent(integrator, strain_inc, stress_n, state_n)` | consistent tangent ↔ 中心差分 |
| `crosscheck(integrator_a, integrator_b, load)` | 2 つの実装 (同じ載荷履歴) |
| `compare_jacobians(model, result_a, result_b)` | 同じステップの 2 つの解でのヤコビアン |
| `check_bindings(model, fortran, cases)` | Fortran に対応づけたメソッド ↔ そのサブルーチン |

**誤差の測り方は全検査で同じ** — 配列 (ブロック、接線、応力ベクトル) 全体の最大差を、参照側の最大値で割る: `max|候補 − 参照| / max|参照|`。合格は `誤差 <= tol`。成分ごとに割らないので、厳密には 0 の成分に乗った丸め誤差を 100 % の誤差とは数えない。分母に単位を持つ定数を置かないので、モデルを MPa で書いても Pa で書いても同じ許容値が同じ意味を持つ。 `check_solution` の残差の行だけは、比べる相手の配列が無いので、行を構成する項の大きさに対して測る (手順 4)。

## まとめて検査する — `check_model`

`check_model` は、モデルが持っているものに応じて当てはまる検査を全部走らせ、何を検査したかを報告する。

```python
from returnmap.verification import check_model

report = check_model(YUKinematic1D.at_defaults())     # at_defaults: param_meta の目安の値で作る
assert report.passed, report.summary()
print(report.summary())
# YUKinematic1D: every check that applies passes
#   ok   converges                 71 steps over 2 histories, 57 plastic
#   ok   tangent                   4.0e-09 (tol 1e-05) at 8 plastic steps
#   —    hand blocks               no @hand_block on the model
#   —    hand-written solution     no user_defined_return_mapping on the model
#   —    hand-written vs autodiff  no user_defined_return_mapping on the model
#   —    hand-written tangent      no user_defined_return_mapping on the model
#   histories: uniaxial (24 plastic steps), reversed 2 % (33 plastic steps)
```

| 行 | 検査するもの | 対象になる条件 |
|---|---|---|
| `converges` | 自動微分の経路が、どの載荷履歴でも最後まで収束する (Δλ ≥ 0、有限の値、塑性ステップがある) | 常に |
| `tangent` | その接線 ↔ 中心差分 (`check_tangent`、履歴ごとに数ステップ) | 常に |
| `hand blocks` | 手導出ブロック ↔ 自動微分 (`check_hand_blocks`) | `@hand_block` がある |
| `hand-written solution` | 手書きの解 ↔ 自動微分側の残差 (`check_solution`) | `user_defined_return_mapping` がある |
| `hand-written vs autodiff` | 手書きの経路 ↔ 自動微分の経路。応力・状態・接線を全ステップで (`crosscheck`) | 同上 |
| `hand-written tangent` | 手書きの接線 ↔ 中心差分 | `user_defined_tangent` がある |
| `UMAT` | コンパイルした UMAT ↔ Python のモデル。同じ履歴を全ステップで (`crosscheck`) | UMAT が宣言されていて、ビルドしてある |

- **Python のモデルだけで止まってよい。** 自動微分で解くだけのモデルは上の 2 行で検査され、それで合格する。`—` の行は「宣言していない」で、不合格ではない。手導出や手書きの解法を足すと、その行が検査されるようになる。
- **失敗した行の内訳** は `report["hand blocks"].reports` などから取れる (元の検査の報告が入っている)。
- **UMAT は、宣言してあってビルド済みなら突き合わせる。** 宣言はファミリの `verification.py` の `UMATS` ([Fortran UMAT クロス検証](fortran.md))。未ビルドなら行は `not built` になり、不合格ではない (gfortran の無い環境でも検査は通る)。比べる相手は `PythonIntegrator(model)` — 手書きのリターンマッピングがあればそれ (UMAT はその移植)、無ければ自動微分。
- **この検査で分からないこと**: 定式化そのものが意図どおりか。どの行も「同じモデルの 2 つの導出」を比べているので、全部が共有する仮定 (単位、符号の規約、定式化) は素通りする。`verified_against_fortran` で対応づけた個々のサブルーチンは、入力を手で用意する必要があるので `check_bindings` で別に検査する。

**載荷履歴** — 何も渡さなければ、モデルの降伏ひずみに合わせた標準の履歴 2 本を使う: 単軸の引張 → 除荷 → 圧縮と、直応力成分とせん断成分を同時に動かして途中で向きを変える多軸の履歴 (転置した外積やせん断の ×2 の誤りは、単軸では見えない)。

そのモデルの機構が標準の履歴では働かないときは、ファミリの `verification.py` に `histories(model)` を書いて履歴を足す。`check_model` は、モデルのモジュールの隣にあるその名前のモジュールを自分で見つける (登録は要らない):

```python
# src/returnmap/models/yu_kinematic/verification.py
MODELS = [YUKinematic3D.at_defaults(), YUKinematicPS.at_defaults(), YUKinematic1D.at_defaults()]

def histories(model):
    """2 % の反転繰返し — C_1 → C_2 の切り替えと停留曲面は、降伏ひずみの 5 倍では働かない。"""
    return {"reversed 2 %": FieldHistory.cyclic_strain(
        [0.02, -0.02, 0.02], n_per_segment=12, ntens=model.ntens)}
```

`MODELS` は、そのファミリのテスト (`tests/test_verification.py`) が `check_model` にかけるモデルの一覧。`check_model(model, histories)` と履歴を渡したときは、渡したものだけを使う。

## 検査を 1 つずつ使う

新しいモデルを実装するときの推奨手順:

1. **数値収束の確認** — `PythonNumericalIntegrator` で残差が落ちることを確認。`result.residual_history` で NR 収束履歴 (二次収束なら桁が倍々で落ちる) を見られる
2. **ヤコビアンのブロック検証** — AD の残差ヤコビアンを名前付きブロックで取り出し、手導出の参照にする:

```python
integrator = PythonIntegrator(model)
result = integrator.stress_update(deps, stress_n, state_n)
jac = integrator.jacobian(result)   # result が自分の Increment を持っている
jac.part["alpha"]["stress"]    # ∂R_alpha/∂σ
jac.part["dlambda"]["dlambda"]
```

   手導出したブロックは、モデル上で `hand_block` を付けて「どのブロックか」を宣言する。メソッド自体は変わらず、直接呼ぶことも `verified_against_fortran` と重ねることもできる:

```python
from returnmap.core import hand_block

class MyModel(MaterialModel):
    @hand_block("alpha", "stress")                 # ∂R_alpha/∂σ
    def dRa_ds(self, state, dlambda): ...

    @hand_block("alpha", "dlambda", approx=2e-2,   # 意図した近似は許容値と理由を宣言
                note="r_max を凍結して導出している")
    def dRa_dl(self, state, dlambda, state_n): ...
```

   引数は名前で埋められる (`state` = 評価点の状態、`dlambda`、`state_n`、`inc`)。Fortran の引数リストに合わせた手書きのまま宣言できる。

   **意図した近似は、モデル上で宣言する** (テストの許容値を緩めない — 理由が残らないため)。宣言の仕方は 2 つある:

   | 宣言 | 意味 | 向く場合 |
   |---|---|---|
   | `approx=2e-2` | 全ステップで、自動微分との差がこの値まで | 導出から落とした項が、どのステップにもある |
   | `approx_when="メソッド名"` | そのメソッドが真を返すステップだけ近似。ほかのステップは厳密な許容値のまま | 切り替えを定数として微分している (切り替わるステップだけ合わない) |

   `approx_when` のステップは比較から外して数える (報告に「78 ステップ中 3 ステップは近似」と出る)。`approx=` も付けると、外さずにその値で判定する。どちらも `note=` (理由) が必須。条件のメソッドは、ブロックと同じく引数を名前で受け取り、真偽を返す。手書きの接線 (`user_defined_tangent`) は `hand_tangent` で同じように宣言する:

```python
from returnmap.core import hand_block, hand_tangent

class MyModel(MaterialModel):
    def gate_in_transition(self, state, state_n):       # 条件: ゲートが 0 でも 1 でもない
        ...

    @hand_block("theta", "beta", approx_when="gate_in_transition",
                note="ゲートを定数として微分している")
    def dRt_db(self, state, dlambda, state_n): ...

    @hand_tangent(approx_when="gate_in_transition", note="同上")
    def user_defined_tangent(self, new, dlambda, inc): ...
```

   組み込みの YU (3D と平面応力) がこの形で、停留曲面のゲートが切り替わるステップを宣言している。接線の宣言を読むのは `check_model` で、`check_tangent` と `crosscheck` は渡された許容値で判定する (integrator を受け取るので、モデルの宣言を知らない)。

   宣言済みブロックは履歴全体で AD と照合する:

```python
from returnmap.verification import check_hand_blocks

report = check_hand_blocks(model, driver.run(load).step_results)
assert report.passed, report.summary()
# hand blocks: 16/25 derived, checked at 57 plastic steps — all within tolerance
#   ok   [alpha][stress] dRa_ds: 2.092e-16 (tol 1e-08, worst at point 18)
#   ...
#   not derived: [stress][stress], [stress][dlambda], ...
```

   最悪誤差をブロックごとに集計するので、1 つの誤りが他に隠れない。未導出のブロックも一覧に出る。ブロックの誤差は、履歴全体での最大差を、そのブロックの履歴全体での最大値で割ったもの — 符号の誤りは 2、0 と置いて落としたブロックは 1 と読める (あるステップでブロックがたまたま 0 に近くても、丸め誤差が大きな誤差にはならない)。厳密に 0 のブロックは、自動微分が丸め誤差 (1e-13 など) を返すので、割る相手が無い。その行と列から決まる大きさ (行を構成する項の大きさ ÷ 未知数の大きさ) より 12 桁以上小さいブロックは 0 と見なし、その大きさに対して測る。`hand_jacobian(model, result)` (`returnmap.simulation`) は、宣言済みブロックを手導出、残りを AD で埋めた行列を返す。**履歴は直ひずみとせん断を同時に動かすこと** — 外積の転置や剪断 ×2 の取り違えは、単軸・純せん断では見えない。

3. **FD tangent チェック** — consistent tangent を中心差分と比較:

```python
from returnmap.verification import check_tangent

report = check_tangent(PythonIntegrator(model), strain_inc, stress_n, state_n)
assert report.passed, report.summary()
# tangent check: agrees with central differences
#   ok   ddsdde: 2.167e-10 (tol 1e-05, worst at [0][1])
```

   引数は `stress_update(strain_inc, stress_n, state_n)` と同じ順。必要なのは `stress_update` だけなので、手書きの接線 (`PythonAnalyticalIntegrator`) にも Fortran UMAT (`FortranIntegrator`) にも使える。`report.rel_err_matrix` でどの成分がずれているかを見られる。

4. **手書きの解の検査** — 手書きのリターンマッピング (`user_defined_return_mapping`: 閉形式解や、自前の残差で回す手書き NR) が返した解を、自動微分側の残差に代入する:

```python
results = StrainDriver(PythonAnalyticalIntegrator(model)).run(load).step_results
report = check_solution(model, results)
assert report.passed, report.summary()
# solution check: 57 plastic steps — 1 row off
#   ok   [stress]: 3.100e-13 (tol 1e-08, worst at point 12)
#   ok   [dlambda]: 8.000e-14 (tol 1e-08, worst at point 12)
#   FAIL [theta]: 2.400e-02 (tol 1e-08, worst at point 31)
#   ok   [beta]: 1.200e-13 (tol 1e-08, worst at point 30)
#   ok   R (explicit state): 0.000e+00 (tol 1e-08, worst at point 0)
```

   手書きの残差の行が自動微分側と食い違っていると、その行だけが 0 にならないので、**どの行が原因かが分かる**。各ステップを個別に判定するので、誤差の累積に左右されない。比べるのは根だけなので、行のスケールを変えた正しい書き換えは合格する。残差ベクトルを持たない閉形式解にも使える。explicit 状態は、報告された値を `update_state` の結果と比べる。履歴中に一度も動かなかった状態は `not exercised` として一覧に出る (その更新式は検査されていない)。

   誤差は「行を構成する項の大きさ」に対する相対値で、ソルバ自身が報告した残差ノルムの 10 倍までは許容する (ソルバが止まった許容値を誤りとは判定しない。反復していない閉形式解には許容幅を与えない)。

   検査の対象は「解」だけである。収束しなかったステップの結果を渡すと `ValueError` になる。`SubSteppingIntegrator` が増分を分割したステップ (`n_substeps > 1`) は、分割前の増分の残差の根ではないので対象から外し、`left out: N sub-stepped steps` として数だけ報告する。

5. **実装同士のクロスチェック** — 閉形式 vs NR、または Python vs Fortran を、同じ載荷履歴で別々に進めてステップごとに比べる:

```python
from returnmap.verification import crosscheck, generate_strain_history

load = FieldHistory(FieldType.STRAIN, generate_strain_history(model))
report = crosscheck(PythonNumericalIntegrator(model), PythonAnalyticalIntegrator(model), load)
assert report.passed, report.summary()
# crosscheck: 35 steps — all agree
#   ok   stress: 1.492e-16 (tol 1e-06)
#   ok   state[ep]: 3.837e-16 (tol 1e-06)
#   ok   tangent: 4.036e-16 (tol 1e-05)
```

   driver は載荷履歴から決まる (ひずみ履歴 → `StrainDriver`、応力履歴 → `StressDriver`)。それ以外は `driver_factory` で渡す — 3D モデルの単軸応力なら `driver_factory=lambda i: MixedDriver(i, prescribed_strain_idx=[0])`。driver 自身の設定 (外側 NR の許容値など) も factory が持つ。食い違ったときは `report.first_failure` が最初のステップで、両方の生の結果 (`.result_a` / `.result_b`) を持つ。`compare_jacobians(model, step.result_a, step.result_b)` に渡すと、2 つの解でのヤコビアンをブロックごとに比べられる。

   2 つの経路は独立に進むので、最初の食い違いの後はすべてのステップがずれる。手書きの解のどの行が原因かを知りたいときは 4 の `check_solution` を使う。

6. **Fortran との対応の検査** — `verified_against_fortran` で Fortran のサブルーチンに対応づけたメソッドを、同じ引数で両側から呼んで比べる:

```python
from returnmap.verification import check_bindings

report = check_bindings(model, FortranModule("j2_isotropic_3d"), {
    "elastic_stiffness": ((model.initial_state(),), (model.E, model.nu)),
})
assert report.passed, report.summary()
# bindings: 1 checked — all agree
#   ok   elastic_stiffness ↔ j2_isotropic_3d_elastic_stiffness: 0.000e+00 (tol 1e-10)
```

   対応づけたのに `cases` に入れなかったメソッドは `not checked` として一覧に出る。

通しの実例: `examples/example_custom_model_workflow.py`
