# 応答を計算する

既存のモデルに載荷履歴を与えて、応力とひずみの履歴を計算する。試験データに合わせるときも、モデルを比べるときも、自分のモデルを確かめるときも、中で行うのはこの計算である。

このページのコードは [`examples/example_response.py`](https://github.com/toruinaba/returnmap/blob/main/examples/example_response.py) の各節で、上から順にそのまま実行できる。

## 1. 最小の計算

単軸の引張と圧縮の繰返しを、Armstrong-Frederick モデルで計算する。

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

計算に要るものは 3 つある。

- **モデル** — 構成則。降伏条件と硬化則を持ち、材料パラメータを与えて作る。`AFKinematic1D` は移動硬化のモデルで、末尾の `1D` は単軸応力を表す。
- **載荷履歴** — ステップごとのひずみ (または応力) の値。ここでは 0 → +1.5 % → −1.5 % → +1.5 % を、区間ごとに 50 ステップで進める。
- **解き方** — `PythonIntegrator` は、1 ステップ分の応力の更新を解く。Abaqus が UMAT を 1 回呼ぶのに当たる。`StrainDriver` は、それを載荷履歴に沿って繰り返す。この文書では前者を integrator、後者を driver と呼ぶ。

`result` が計算の結果で、応力 `result.stress`、ひずみ `result.strain`、状態量の履歴 `result.history(名前)` を持つ。単位は決められていない。パラメータを MPa で与えれば、応力は MPa で返る。

## 2. モデルを選ぶ

組み込みのモデルは `returnmap.models` にある。クラス名の末尾が、応力の状態を表す。

| 末尾 | 応力の状態 | 成分の数 |
|---|---|---|
| `1D` | 単軸応力 | 1 |
| `3D` | 3 次元 | 6 |
| `PS` | 平面応力 | 3 |

**単軸の試験を計算するなら `1D` を使う。** どんなモデルがあるか、それぞれの式とパラメータの意味は [組み込みモデルと次元選択](models.md) にある。平面ひずみの使い方もそこにある。

```{literalinclude} ../../examples/example_response.py
:language: python
:start-after: "# %% 2. モデルを選ぶ"
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

- **`param_meta`** は、各パラメータの意味、単位、目安の値、ありうる範囲を持つ。モデルに何を与えればよいかは、これを見れば分かる。
- **パラメータはすべて必須**で、既定値は無い。キーワードで渡す。足りない名前や綴りの違う名前があれば、その名前を挙げたエラーになる。
- **`at_defaults()`** は、目安の値を入れたモデルを作る。値にこだわらず、まず動かしたいときに使う。
- **`with_params(...)`** は、一部の値だけを変えた新しいモデルを返す。元のモデルは変わらない。パラメータを振るときに使う。

## 3. 載荷履歴を作る

載荷履歴は `FieldHistory` で、ひずみか応力かの種類と、ステップごとの値の配列を持つ。値は**その時点までの累積値**で、増分ではない。計算は 0 から始まるので、配列には最初のステップの終わりの値から書く。

```{literalinclude} ../../examples/example_response.py
:language: python
:start-after: "# %% 3. 載荷履歴を作る"
:end-before: "# %%"
```

```text
(100,) (150, 1)
```

配列をそのまま渡すほかに、よく使う形には作る関数がある。応力の履歴を作る関数は、名前の `_strain` を `_stress` に替えたものである。

| 関数 | 形 |
|---|---|
| `FieldHistory.cyclic_strain(peaks, n_per_segment=20)` | 0 から始めて、`peaks` の値を順に直線で結ぶ |
| `FieldHistory.triangular_strain(amplitude, n_cycles)` | 一定振幅の三角波 (`mean=` で平均値) |
| `FieldHistory.sine_strain(amplitude, n_cycles, n_per_cycle=80)` | 正弦波 |
| `FieldHistory.decaying_cyclic_strain(amplitude, n_cycles, decay=0.5)` | 振幅が指数的に変わる繰返し |

成分が複数あるモデル (`3D`、`PS`) では、次の 2 点に注意する。

- **成分の数を合わせる。** 作る関数には `ntens=model.ntens` を渡す。動かす成分は `component=` で選び (既定は 0、つまり 11 成分)、ほかの成分は 0 になる。複数の成分を同時に動かす履歴は、形が `(ステップ数, 成分の数)` の配列を自分で作って渡す。
- **せん断ひずみは工学ひずみで与える。** 成分の並びは `[11, 22, 33, 12, 13, 23]` で、ひずみのせん断成分は γ = 2ε である。応力のせん断成分はテンソルの成分そのままである ([概念](../concepts.md) の Voigt 規約)。

### 刻みを変えて確かめる

**刻みの細かさは結果に効く。** 1 ステップの中は後退 Euler 法で積分するので、刻みが粗いと応答に誤差が残る。同じ 1.5 % の引張を、刻みを変えて計算すると次のようになる。

```{literalinclude} ../../examples/example_response.py
:language: python
:start-after: "# %% 3b. 刻みを変えて確かめる"
:end-before: "# %%"
```

```text
    5 ステップ: 387.4 MPa
   50 ステップ: 394.9 MPa
 1000 ステップ: 395.9 MPa
```

5 ステップでは 2 % ずれ、50 ステップで 0.3 % になる。使う刻みを決めるときは、細かくしても結果が動かないことを一度確かめる。

## 4. 結果を読む

```{literalinclude} ../../examples/example_response.py
:language: python
:start-after: "# %% 4. 結果を読む"
:end-before: "# %%"
```

```text
(150, 1) (150, 1)
['stress', 'alpha', 'ep']
(150,) (150, 1)
True 3 5.37e-04
```

- **応力とひずみ** — `result.stress` と `result.strain` は、形が `(ステップ数, 成分の数)` の配列である。`1D` のモデルでも 2 次元なので、曲線を取り出すには `result.stress[:, 0]` と書く。
- **状態量** — `result.history(名前)` で取り出す。使える名前は `model.state_names` にある。上の例では `ep` が累積塑性ひずみ、`alpha` が背応力である。スカラーの状態量は長さがステップ数の配列、テンソルの状態量は応力と同じ形になる。
- **ステップごとの詳細** — `result.step_results[i]` が i 番目のステップの記録で、塑性だったか (`is_plastic`)、反復の回数 (`n_iterations`)、塑性乗数の増分 (`dlambda`) などを持つ。

### 1 ステップずつ受け取る

`run` の代わりに `iter_run` を使うと、ステップを 1 つ進めるたびに結果を受け取れる。途中で条件を見て止めたいときに使う。

```{literalinclude} ../../examples/example_response.py
:language: python
:start-after: "# %% 4b. 1 ステップずつ受け取る"
:end-before: "# %%"
```

```text
降伏したステップ: 4 (ひずみ 0.0015)
```

### 図にする

応力-ひずみ曲線は、2 つの配列をそのまま描けばよい。図を描くための matplotlib は `pip install "returnmap[examples]"` で一緒に入る。

```{literalinclude} ../../examples/example_response.py
:language: python
:start-after: "# %% 4c. 図にする"
:end-before: "# %%"
```

## 5. 応力で載荷する、3D のモデルで単軸応力にする

ここまでは、すべての成分のひずみを与えた。与えるものが違えば、driver を替える。

| 与えるもの | driver |
|---|---|
| すべての成分のひずみ | `StrainDriver` |
| すべての成分の応力 | `StressDriver` |
| 一部の成分はひずみ、残りの成分は応力 (既定は 0) | `MixedDriver` |

### 応力で載荷する

```{literalinclude} ../../examples/example_response.py
:language: python
:start-after: "# %% 5. 応力で載荷する"
:end-before: "# %%"
```

```text
400 MPa でのひずみ: 1.66 %
```

`StressDriver` は、与えた応力になるひずみを、ステップごとに反復で求める。そのぶん、ひずみを与える計算より時間がかかる。

### 3D のモデルで単軸応力にする

**3D のモデルで引張試験を計算するときは、`MixedDriver` を使う。** ε11 だけを与えてほかのひずみを 0 のままにすると、横方向が拘束された単軸ひずみになり、引張試験とは違う応力が出る。

```{literalinclude} ../../examples/example_response.py
:language: python
:start-after: "# %% 5b. 3D のモデルで単軸応力にする"
:end-before: "# %%"
```

```text
単軸ひずみ: σ11 =  2817.4 MPa, σ22 =  2453.8 MPa
単軸応力  : σ11 =   394.9 MPa, σ22 =     0.0 MPa
```

`prescribed_strain_idx=[0]` は「成分 0 (11 成分) はひずみで与える」という指定で、残りの成分は応力が 0 になるように解かれる。載荷履歴には、ひずみで与える成分だけを書く (ここでは 1 列)。結果は 1 節の `1D` のモデルと同じ 394.9 MPa になる。

## 6. 解けなかったとき

1 ステップの増分が大きいと、応力の更新の反復が収束しないことがある。そのときは、既定では `RuntimeError` になる。

```{literalinclude} ../../examples/example_response.py
:language: python
:start-after: "# %% 6. 解けなかったとき"
:end-before: "# %%"
```

```text
既定では例外になる
False 6 6
```

integrator を `raise_on_nonconverged=False` で作ると、例外の代わりに、解けたところまでの結果が返る。

- `result.converged` が `False` なら、途中で止まっている。
- `result.failed_step` は、解けなかったステップの番号 (0 始まり) である。最後まで解ければ `None` になる。
- `result.stress` などの配列には、完了したステップだけが入る。上の例では 6 ステップ分である。

### 刻みを細かくする、増分を分割する

```{literalinclude} ../../examples/example_response.py
:language: python
:start-after: "# %% 6b. 刻みを細かくする、増分を分割する"
```

```text
True
True [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1]
```

- **刻みを細かくする。** まずこれを試す。3 節のとおり、精度のためにも細かい刻みが要る。
- **増分を分割する。** `SubSteppingIntegrator` で integrator を包むと、解けなかった増分だけを 2、4、8 … と分割して解き直す。上の例では 7 番目のステップだけが 2 分割されている (`n_substeps`)。解けている増分には何もしないので、結果は変わらない。

integrator 自身も、失敗したステップを別の初期値から 1 回解き直してから、解けなかったと報告する。この仕組みと、どういうときに解けなくなるかは [概念](../concepts.md) の「非収束との付き合い方」にある。

## 次に読むもの

- 試験の記録に合うパラメータを探す: [チュートリアル](tutorial.md)
- 複数のモデルやパラメータを並べて比べる: [モデルを比べる](comparing.md)
- 各モデルの式とパラメータ: [組み込みモデルと次元選択](models.md)
