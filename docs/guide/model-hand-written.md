# 手書きの解法を照合する

UMAT に書く代数 (リターンマッピングと整合接線) を Python で書き、自動微分を基準として検査する手順を示す ([新しいモデルを足す](adding-a-model.md) の段階 B)。前提は [モデルを書く](model-python.md) である。

コードは [`examples/model_family/`](https://github.com/toruinaba/returnmap/tree/main/examples/model_family) のものである。

## 1. 手書きのリターンマッピング

`user_defined_return_mapping(inc)` に、1 ステップの塑性補正を書く。J2 のリターンは半径方向であるため、1 ステップはスカラーの方程式 `g(Δλ) = q − 3μ Δλ − σ_y(p_n + Δλ) = 0` に帰着する (`q` は試行応力の Mises 応力)。この恒等式は直応力 3 成分がそろう次元でのみ成り立つため、クラスを 3D に固定する。

```{literalinclude} ../../examples/model_family/src/mymodels/voce/model.py
:language: python
:start-at: "class J2Voce3D(J2Voce):"
:end-before: "    def user_defined_tangent"
```

| 項目 | 内容 |
|---|---|
| 引数 `inc` | 1 ステップの入力。`inc.trial["stress"]` は試行応力、`inc.state_n` はステップ開始時の状態、`inc.strain_inc` はひずみ増分 |
| 戻り値 | `ReturnMappingResult`。応力、状態、`dlambda`、反復回数、残差の履歴、`converged` を持つ |
| `converged` | 必須であり、既定値は無い。反復の上限で終了した場合に、成功と報告しないためである |
| 結果の検査 | Δλ < 0 と非有限の値は、`converged=True` と報告されていても非収束として扱われる |

## 2. 手書きの接線

`user_defined_tangent(new, dlambda, inc)` に、整合接線を書く。`new` はステップ終了時の状態である。

```{literalinclude} ../../examples/model_family/src/mymodels/voce/model.py
:language: python
:pyobject: J2Voce3D.user_defined_tangent
```

## 3. 検査の報告

`verification.py` の `MODELS` に `J2Voce3D.at_defaults()` を加えると、手書きの経路の検査が実行される。

```none
J2Voce3D: every check that applies passes
  ok   converges                 59 steps over 2 histories, 47 plastic
  ok   tangent                   4.6e-10 (tol 1e-05) at 8 plastic steps
  —    hand blocks               no @hand_block on the model
  ok   hand-written solution     4.9e-13 (tol 1e-08) over 3 rows at 47 plastic steps
  ok   hand-written vs autodiff  stress 3.3e-12, state 4.2e-12 (tol 1e-06), tangent 4.8e-12 (tol 1e-05) over 2 histories
  ok   hand-written tangent      4.6e-10 (tol 1e-05) at 8 plastic steps
  —    UMAT                      module voce_3d is not built — returnmap build --family mymodels.voce
  histories: uniaxial (24 plastic steps), multiaxial (23 plastic steps)
```

| 行 | 検査の内容 |
|---|---|
| `hand-written solution` | 手書きの解を、フレームワークの残差 (自動微分側の方程式) に代入すると 0 になる |
| `hand-written vs autodiff` | 2 つの経路を同じ履歴で進め、応力・状態・接線が全ステップで一致する |
| `hand-written tangent` | 手書きの接線が、応力更新の中心差分と一致する |
| `UMAT` | 段階 C の検査 ([UMAT を確かめる](model-umat.md))。ビルドしていない環境では `—` であり、不合格には数えない |

基準は自動微分である。2 つの経路が一致しない場合は、手導出の側を先に調べる。

## 4. 行の内訳

報告の行は名前で取り出せる。行は、検査したステップごとの報告を持つ。

```{literalinclude} ../../examples/model_family/report.py
:language: python
:start-after: "# %% One row in detail"
```

```none
passed — 4.6e-10 (tol 1e-05) at 8 plastic steps
['multiaxial[0]', 'multiaxial[1]', 'multiaxial[2]', 'multiaxial[3]', 'uniaxial[0]', 'uniaxial[1]', 'uniaxial[2]', 'uniaxial[3]']
tangent check: agrees with central differences
  ok   ddsdde: 2.857e-11 (tol 1e-05, worst at [2][2])
```

不合格の行は、同じ方法で、誤差が最大のステップと成分まで追える。

## 5. ヤコビアンのブロックの宣言

手書きの解法が複数の未知数とヤコビアンを持つモデルでは、手導出したブロックを `@hand_block(row, col)` で宣言する。宣言したブロックは、自動微分のヤコビアンとブロックごとに照合され、報告の `hand blocks` の行になる。

| 場合 | 書き方 |
|---|---|
| 厳密なブロック | `@hand_block(row, col)` |
| 意図して項を省いたブロック | `approx=` (全ステップ)、`approx_when=` (条件に該当するステップのみ)。テストの許容値は変更しない |
| 近似を含む接線 | `@hand_tangent` |

書き方と検査の内容は [検証ワークフロー](verification.md) を参照。

## 関連ページ

- 次の段階: [UMAT を確かめる](model-umat.md)
- 各検査の測定内容と許容値: [検証ワークフロー](verification.md)
- `user_defined_return_mapping` などのフックの契約: [モデルを作る](writing-models.md)
