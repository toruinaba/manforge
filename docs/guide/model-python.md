# モデルを書く

降伏関数と状態の発展則を Python で書き、共通の検査に合格させるまでの手順を示す ([新しいモデルを足す](adding-a-model.md) の段階 A)。リターンマッピングと整合接線は自動微分で解かれるため、この段階のモデルは、応答の計算、比較、パラメータの同定にそのまま使用できる。

コードは [`examples/model_family/`](https://github.com/toruinaba/returnmap/tree/main/examples/model_family) のものである。モデルと docstring は英語で書いてある (API リファレンスの規約に合わせている)。

## 1. モデルの定義

例は J2 + Voce 硬化である。降伏応力は `σ_y(p) = σ_y0 + Q (1 − exp(−b p))`、`p` は累積塑性ひずみである。

```{literalinclude} ../../examples/model_family/src/mymodels/voce/model.py
:language: python
:pyobject: J2Voce
```

```{literalinclude} ../../examples/model_family/src/mymodels/voce/model.py
:language: python
:pyobject: J2Voce1D
```

| 記述 | 内容 |
|---|---|
| `param_names` | パラメータの宣言。同名のキーワード引数で構築し、同名の属性 (`self.E` など) で参照する。順序は UMAT の `PROPS` の順である。`__init__` は書かない |
| `param_meta` | 各パラメータの説明、単位、目安の値、範囲。任意。書くと `J2Voce.at_defaults()` が使え、検証の宣言、調整用のノートブック、service の入力フォームがこれを参照する |
| `ep = Explicit(...)` | 状態変数の宣言。閉形式で更新できる状態は `Explicit`、残差を解く状態は `Implicit` で宣言する |
| `yield_function(state)` | 降伏関数 f。必須である |
| `update_state(new, dlambda, inc)` | `Explicit` の状態の更新。`inc.state_n` はステップ開始時の状態、`dlambda` は塑性乗数の増分である |
| `dimension = UNIAXIAL_1D` | 次元の固定。書かないクラスは、構築時の `dimension=` で次元を選ぶ |

- **次元ごとの式は書かない。** `vonmises` などの演算子が次元の違いを吸収するため、物理は親クラスに 1 度だけ書く。次元を固定したクラスは `dimension` のみを持つ。
- **流れ則は既定で関連流れである。** 流れ方向は降伏関数の勾配から自動微分で求められる。
- その他のフック (`Implicit` の状態、非関連流れ、弾性の変更など) は [モデルを作る](writing-models.md) を参照。

## 2. 検証の宣言

検査の対象とするモデルを、ファミリの `verification.py` の `MODELS` に並べる。

```{literalinclude} ../../examples/model_family/src/mymodels/voce/verification.py
:language: python
:start-at: "MODELS = ["
:end-at: "]"
```

`J2Voce3D` は段階 B で追加するクラスである ([手書きの解法を照合する](model-hand-written.md))。

## 3. テスト

テストは、どのファミリでも同じ内容である。

```{literalinclude} ../../examples/model_family/src/mymodels/voce/tests/test_verification.py
:language: python
:start-at: "import pytest"
:end-at: "assert report.passed, report.summary()"
```

```bash
uv sync --extra dev
uv run pytest
```

## 4. 検査の報告

`check_model` は、標準の載荷履歴でモデルを計算し、モデルに該当する検査をすべて実行する。報告は `report.summary()` で表示する。

```{literalinclude} ../../examples/model_family/report.py
:language: python
:start-after: "# %% The report of every model"
:end-before: "# %%"
```

`J2Voce1D` の報告は次のとおり。

```none
J2Voce1D: every check that applies passes
  ok   converges                 35 steps over 1 histories, 24 plastic
  ok   tangent                   7.2e-10 (tol 1e-05) at 4 plastic steps
  —    hand blocks               no @hand_block on the model
  —    hand-written solution     no user_defined_return_mapping on the model
  —    hand-written vs autodiff  no user_defined_return_mapping on the model
  —    hand-written tangent      no user_defined_return_mapping on the model
  —    UMAT                      no UMAT declared for J2Voce1D in mymodels.voce.verification
  histories: uniaxial (24 plastic steps)
```

| 行 | 検査の内容 |
|---|---|
| `converges` | 自動微分の経路が、すべての載荷履歴で最後まで収束する |
| `tangent` | 整合接線が、応力更新の中心差分と一致する |
| `—` の行 | モデルが宣言していない項目。不合格には数えない |

| 項目 | 内容 |
|---|---|
| 標準の載荷履歴 | 単軸の載荷・除荷・逆載荷と、方向を変える多軸の履歴 (多成分のモデルのみ)。ひずみの範囲は降伏ひずみの 5 倍までである |
| 履歴の追加 | 標準の履歴では到達しない挙動を持つモデル (大きなひずみからの反転で作用する硬化則など) は、`verification.py` に `histories(model)` を定義する |
| 弾性域の無いモデル | 履歴の尺度とするひずみを渡す ([検証ワークフロー](verification.md) の `check_model`) |

## 関連ページ

- 試験データに対するパラメータの同定: [試験データに合わせる](tutorial.md)
- 他のモデルとの比較: [モデルを比べる](comparing.md)
- 次の段階: [手書きの解法を照合する](model-hand-written.md)
