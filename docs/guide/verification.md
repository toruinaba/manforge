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

4. **実装同士のクロスチェック** — 閉形式 vs NR、または Python vs Fortran を全履歴で比較:

```python
from manforge.verification import CrosscheckStrainDriver, generate_strain_history

cc = CrosscheckStrainDriver(PythonNumericalIntegrator(model), PythonAnalyticalIntegrator(model))
cr = cc.run(FieldHistory(FieldType.STRAIN, "eps", generate_strain_history(model)))
print(cr.passed, cr.max_stress_rel_err)
```

通しの実例: `examples/example_custom_model_workflow.py`
