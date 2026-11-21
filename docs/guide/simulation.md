# 応答を計算する

モデルに載荷履歴を与え、応力・ひずみ・状態量の履歴を計算する手順を示す。フィッティング、モデルの比較、検証は、いずれも内部でこの計算を行う。

コードは [`examples/example_response.py`](https://github.com/toruinaba/returnmap/blob/main/examples/example_response.py) の各節であり、上から順に実行できる。

## 1. 最小の計算

Armstrong-Frederick モデルによる、単軸の引張・圧縮の繰返し。

```{literalinclude} ../../examples/example_response.py
:language: python
:start-after: "# %% 1. 最小の計算"
:end-before: "# %%"
```

```text
ステップ数           : 150
最初の折返しでの応力 : 394.9 MPa
最後の累積塑性ひずみ : 0.0650
```

計算は次の 4 つで構成される。

| 要素 | 上の例 | 役割 |
|---|---|---|
| モデル | `AFKinematic1D` | 構成則。降伏条件、硬化則、材料パラメータを持つ |
| 載荷履歴 | `FieldHistory` | ステップごとのひずみ (または応力) の値 |
| integrator | `PythonIntegrator` | 1 ステップ分の応力更新を解く。Abaqus による UMAT の呼び出し 1 回に相当する |
| driver | `StrainDriver` | integrator を載荷履歴に沿って繰り返し呼ぶ |

結果 `result` は、応力 `result.stress`、ひずみ `result.strain`、状態量の履歴 `result.history(名前)` を持つ (4 節)。

単位系は規定しない。パラメータを MPa で与えた場合、応力は MPa で返る。

## 2. モデルの選択

組み込みモデルは `returnmap.models` から import する。クラス名の末尾は応力状態を表す。

| 末尾 | 応力状態 | 成分数 |
|---|---|---|
| `1D` | 単軸応力 | 1 |
| `3D` | 3 次元 | 6 |
| `PS` | 平面応力 | 3 |

単軸試験の計算には `1D` を用いる。モデルの一覧、定式化、パラメータの定義、平面ひずみの指定方法は [組み込みモデルと次元選択](models.md) を参照。

```{literalinclude} ../../examples/example_response.py
:language: python
:start-after: "# %% 2. モデルの選択"
:end-before: "# %%"
```

```text
E         Young's modulus                   MPa   210000.0  (1000.0, 1000000.0)
nu        Poisson's ratio                              0.3  (0.0, 0.49)
sigma_y0  Initial yield stress              MPa      250.0  (1.0, 5000.0)
C_k       Kinematic hardening modulus (AF)  MPa    20000.0  (0.0, 1000000.0)
gamma     Dynamic recovery coefficient (AF)          100.0  (0.0, 10000.0)
{'E': 206000.0, 'nu': 0.3, 'sigma_y0': 250.0, 'C_k': 40000.0, 'gamma': 100.0}
```

| 名前 | 内容 |
|---|---|
| `Model.param_meta` | 各パラメータの説明、単位、目安の値、範囲 |
| `Model(**params)` | 全パラメータをキーワードで指定して構築する。既定値は無い。名前の不足・誤りは `TypeError` |
| `Model.at_defaults()` | 目安の値で構築する |
| `model.with_params(...)` | 指定したパラメータだけを置き換えた新しいモデルを返す。元のモデルは変更されない |
| `model.params` | 現在のパラメータ値 (dict) |

## 3. 載荷履歴の作成

`FieldHistory` は、種類 (ひずみ / 応力) と、ステップごとの値の配列を持つ。

- 値は累積値であり、増分ではない。
- 初期値 0 は配列に含めない。配列の先頭は、第 1 ステップ終了時の値である。
- 配列の形は `(ステップ数, 成分数)`。1 成分の履歴は 1 次元配列でもよい。

```{literalinclude} ../../examples/example_response.py
:language: python
:start-after: "# %% 3. 載荷履歴の作成"
:end-before: "# %%"
```

```text
(100,) (150, 1)
```

生成関数は次のとおり。応力履歴の生成関数は、名前の `_strain` を `_stress` に置き換えたものである。

| 関数 | 形 |
|---|---|
| `FieldHistory.cyclic_strain(peaks, n_per_segment=20)` | 0 から始め、`peaks` の値を順に直線で結ぶ |
| `FieldHistory.triangular_strain(amplitude, n_cycles)` | 一定振幅の三角波 (`mean=` で平均値) |
| `FieldHistory.sine_strain(amplitude, n_cycles, n_per_cycle=80)` | 正弦波 |
| `FieldHistory.decaying_cyclic_strain(amplitude, n_cycles, decay=0.5)` | 振幅が指数的に変化する繰返し |

多成分のモデル (`3D`、`PS`) では、次を指定する。

| 項目 | 指定 |
|---|---|
| 成分数 | 生成関数に `ntens=model.ntens` を渡す |
| 載荷する成分 | `component=` (既定は 0 = 11 成分)。他の成分は 0 になる |
| 複数成分の同時載荷 | 形 `(ステップ数, 成分数)` の配列を `FieldHistory` に直接渡す |
| 成分の順序 | `[11, 22, 33, 12, 13, 23]` |
| せん断成分 | ひずみは工学ひずみ (γ = 2ε)、応力はテンソル成分 ([概念](../concepts.md) の Voigt 規約) |

### 刻みの影響

1 ステップは後退 Euler 法で積分される。刻みが粗いと、応答に誤差が残る。1.5 % の単調引張を、ステップ数を変えて計算した結果を示す。

```{literalinclude} ../../examples/example_response.py
:language: python
:start-after: "# %% 3b. 刻みの影響"
:end-before: "# %%"
```

```text
    5 ステップ: 387.4 MPa
   50 ステップ: 394.9 MPa
 1000 ステップ: 395.9 MPa
```

1000 ステップの結果に対する差は、5 ステップで 2 %、50 ステップで 0.3 % である。刻みは、細かくしても結果が変わらない範囲で選ぶ。

## 4. 結果の参照

```{literalinclude} ../../examples/example_response.py
:language: python
:start-after: "# %% 4. 結果の参照"
:end-before: "# %%"
```

```text
(150, 1) (150, 1)
['stress', 'alpha', 'ep']
(150,) (150, 1)
True 3 5.37e-04
```

| 属性 | 形 | 内容 |
|---|---|---|
| `result.stress` | `(ステップ数, 成分数)` | 応力。`1D` のモデルでも 2 次元である (曲線は `result.stress[:, 0]`) |
| `result.strain` | `(ステップ数, 成分数)` | ひずみ |
| `result.history(名前)` | スカラーは `(ステップ数,)`、テンソルは `(ステップ数, 成分数)` | 状態量の履歴。指定できる名前は `model.state_names` |
| `result.step_results[i]` | — | i 番目のステップの記録。`is_plastic` (塑性か)、`n_iterations` (反復回数)、`dlambda` (塑性乗数の増分) など |
| `result.converged`、`result.failed_step` | — | 非収束の有無と位置 (6 節) |

`AFKinematic1D` の状態量は、`ep` (累積塑性ひずみ) と `alpha` (背応力) である。

### ステップ単位の実行

`iter_run` は、1 ステップ進むごとに結果を返す。条件を判定して途中で打ち切る場合に用いる。

```{literalinclude} ../../examples/example_response.py
:language: python
:start-after: "# %% 4b. ステップ単位の実行"
:end-before: "# %%"
```

```text
降伏したステップ: 4 (ひずみ 0.0015)
```

### 作図

応力-ひずみ曲線は、`result.strain` と `result.stress` をそのまま描く。matplotlib は `pip install "returnmap[examples]"` で入る。

```{literalinclude} ../../examples/example_response.py
:language: python
:start-after: "# %% 4c. 作図"
:end-before: "# %%"
```

## 5. 応力制御と混合制御

driver は、載荷履歴で与える量に応じて選ぶ。

| 与える量 | driver |
|---|---|
| 全成分のひずみ | `StrainDriver` |
| 全成分の応力 | `StressDriver` |
| 一部の成分はひずみ、残りの成分は応力 (既定は 0) | `MixedDriver` |

### 応力制御

```{literalinclude} ../../examples/example_response.py
:language: python
:start-after: "# %% 5. 応力制御"
:end-before: "# %%"
```

```text
400 MPa でのひずみ: 1.66 %
```

`StressDriver` は、目標応力を満たすひずみを、ステップごとに反復で求める。計算時間はひずみ制御より長い。

### 3D モデルでの単軸応力

3D のモデルで引張試験を計算する場合は、`MixedDriver` を用いる。`StrainDriver` に ε11 のみを与えると、他のひずみ成分が 0 に拘束された単軸ひずみ状態となり、単軸応力とは異なる応力になる。

```{literalinclude} ../../examples/example_response.py
:language: python
:start-after: "# %% 5b. 3D モデルでの単軸応力"
:end-before: "# %%"
```

```text
単軸ひずみ: σ11 =  2817.4 MPa, σ22 =  2453.8 MPa
単軸応力  : σ11 =   394.9 MPa, σ22 =     0.0 MPa
```

- `prescribed_strain_idx=[0]` は、成分 0 (11 成分) をひずみで与える指定である。残りの成分は、応力が 0 になるように解かれる。
- 載荷履歴には、ひずみで与える成分のみを書く (この例では 1 列)。
- 単軸応力の結果 (394.9 MPa) は、1 節の `1D` のモデルと一致する。

## 6. 非収束時の扱い

1 ステップの増分が大きい場合、応力更新の反復が収束しないことがある。既定の動作は `RuntimeError` の送出である。

```{literalinclude} ../../examples/example_response.py
:language: python
:start-after: "# %% 6. 非収束時の扱い"
:end-before: "# %%"
```

```text
既定では例外になる
False 6 6
```

`raise_on_nonconverged=False` で作った integrator では、例外を送出せず、収束したステップまでの結果を返す。

| 属性 | 内容 |
|---|---|
| `result.converged` | 全ステップが収束した場合に `True` |
| `result.failed_step` | 収束しなかったステップの番号 (0 始まり)。全ステップが収束した場合は `None` |
| `result.stress` など | 収束したステップのみを含む (この例では 6 ステップ) |

### 対処

```{literalinclude} ../../examples/example_response.py
:language: python
:start-after: "# %% 6b. 対処"
```

```text
True
True [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1]
```

| 方法 | 内容 |
|---|---|
| 刻みを細かくする | 載荷履歴のステップ数を増やす。精度の面でも有効である (3 節) |
| 増分の自動分割 | `SubSteppingIntegrator` で integrator を包む。収束しなかった増分のみを 2、4、8 … 分割して解く。分割数は `n_substeps` に記録される (この例では 7 番目のステップが 2 分割)。収束する増分の結果は変わらない |

integrator は、非収束を報告する前に、失敗したステップを別の初期値から 1 回解き直す。詳細は [概念](../concepts.md) の「非収束との付き合い方」を参照。

## 関連ページ

- 試験データに対するパラメータの同定: [試験データに合わせる](tutorial.md)
- モデル、パラメータ、解法の比較: [モデルを比べる](comparing.md)
- 各モデルの定式化とパラメータ: [組み込みモデルと次元選択](models.md)
