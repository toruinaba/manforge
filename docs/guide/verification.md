# 検証ワークフロー

新しいモデルを実装するときの推奨手順:

1. **数値収束の確認** — `PythonNumericalIntegrator` で残差が落ちることを確認。`result.residual_history` で NR 収束履歴 (二次収束なら桁が倍々で落ちる) を見られる
2. **ヤコビアンのブロック検証** — AD の残差ヤコビアンを名前付きブロックで取り出し、手導出の参照にする:

```python
from manforge.verification import JacobianChecker

result = PythonIntegrator(model).stress_update(deps, stress_n, state_n)
jac = JacobianChecker(model).compute(result)   # result が自分の Increment を持っている
jac.part["alpha"]["stress"]    # ∂R_alpha/∂σ
jac.part["dlambda"]["dlambda"]
```

   手導出したブロックは、モデル上で `hand_block` を付けて「どのブロックか」を宣言する。メソッド自体は変わらず、直接呼ぶことも `verified_against_fortran` と重ねることもできる:

```python
from manforge.core import hand_block

class MyModel(MaterialModel):
    @hand_block("alpha", "stress")                 # ∂R_alpha/∂σ
    def dRa_ds(self, state, dlambda): ...

    @hand_block("alpha", "dlambda", approx=2e-2,   # 意図した近似は許容値と理由を宣言
                note="r_max を凍結して導出している")
    def dRa_dl(self, state, dlambda, state_n): ...
```

   引数は名前で埋められる (`state` = 評価点の状態、`dlambda`、`state_n`、`inc`)。Fortran の引数リストに合わせた手書きのまま宣言できる。宣言済みブロックは履歴全体で AD と照合する:

```python
report = JacobianChecker(model).check_hand_blocks(driver.run(load).step_results)
assert report.passed, report.summary()
# hand blocks: 16/25 derived, checked at 57 plastic steps — all within tolerance
#   ok   [alpha][stress] dRa_ds: 2.092e-16 (tol 1e-08, worst at point 18)
#   ...
#   not derived: [stress][stress], [stress][dlambda], ...
```

   最悪誤差をブロックごとに集計するので、1 つの誤りが他に隠れない。未導出のブロックも一覧に出る。`JacobianChecker(model).hand(result)` は、宣言済みブロックを手導出、残りを AD で埋めた行列を返す。**履歴は直ひずみとせん断を同時に動かすこと** — 外積の転置や剪断 ×2 の取り違えは、単軸・純せん断では見えない。

3. **FD tangent チェック** — consistent tangent を中心差分と比較:

```python
from manforge.verification import TangentChecker
r = TangentChecker(PythonIntegrator(model)).check(stress_n, state_n, strain_inc)
print(r.passed, r.max_rel_err)
```

4. **手書きの解の検査** — 手書きのリターンマッピング (`user_defined_return_mapping`: 閉形式解や、自前の残差で回す手書き NR) が返した解を、自動微分側の残差に代入する:

```python
results = StrainDriver(PythonAnalyticalIntegrator(model)).run(load).step_results
report = JacobianChecker(model).check_solution(results)
assert report.passed, report.summary()
# solution check: 57 plastic steps — 1 row off
#   ok   [stress]: 3.100e-13 (tol 1e-08, worst at point 12)
#   ok   [dlambda]: 8.000e-14 (tol 1e-08, worst at point 12)
#   FAIL [theta]: 2.400e-02 (tol 1e-08, worst at point 31)
#   ok   [beta]: 1.200e-13 (tol 1e-08, worst at point 30)
#   ok   R (explicit state): 0.000e+00 (tol 1e-08, worst at point 0)
```

   手書きの残差の行が自動微分側と食い違っていると、その行だけが 0 にならないので、**どの行が原因かが分かる**。各ステップを個別に判定するので、誤差の累積に左右されない。比べるのは根だけなので、行のスケールを変えた正しい書き換えは合格する。残差ベクトルを持たない閉形式解にも使える。explicit 状態は、報告された値を `update_state` の結果と比べる。履歴中に一度も動かなかった状態は `not exercised` として一覧に出る (その更新式は検査されていない)。

   誤差は「行を構成する項の大きさ」に対する相対値で、ソルバ自身が報告した残差ノルムの 10 倍までは許容する (ソルバが止まった許容値を誤りとは判定しない)。

5. **実装同士のクロスチェック** — 閉形式 vs NR、または Python vs Fortran を、同じ載荷履歴で別々に進めてステップごとに比べる:

```python
from manforge.verification import crosscheck, generate_strain_history

load = FieldHistory(FieldType.STRAIN, "eps", generate_strain_history(model))
result = crosscheck(PythonNumericalIntegrator(model), PythonAnalyticalIntegrator(model), load)
assert result.passed, result.summary()
# crosscheck: 40 steps — all agree
#   ok   stress: 2.100e-13 (tol 1e-06)
#   ok   state[ep]: 1.100e-16 (tol 1e-06)
#   ok   tangent: 3.400e-12 (tol 1e-05)
```

   driver は載荷履歴から決まる (ひずみ履歴 → `StrainDriver`、応力履歴 → `StressDriver`)。それ以外は `driver_factory` で渡す — 3D モデルの単軸応力なら `driver_factory=lambda i: MixedDriver(i, prescribed_strain_idx=[0])`。driver 自身の設定 (外側 NR の許容値など) も factory が持つ。食い違ったときは `result.first_failure` が最初のステップで、両方の生の結果 (`.result_a` / `.result_b`) を持つ。

   2 つの経路は独立に進むので、最初の食い違いの後はすべてのステップがずれる。手書きの解のどの行が原因かを知りたいときは 4 の `check_solution` を使う。

通しの実例: `examples/example_custom_model_workflow.py`
