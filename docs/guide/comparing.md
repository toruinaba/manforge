# モデルを比べる

複数のモデル、パラメータ、解法を同じ載荷履歴で計算し、応答・計算コスト・フィットの良さを並べる手順を示す。入口は `compare_models` の 1 つである。計算の基本は [応答を計算する](simulation.md) を前提とする。

コードは [`examples/example_compare_models.py`](https://github.com/toruinaba/returnmap/blob/main/examples/example_compare_models.py) の各節であり、上から順に実行できる (約 15 秒)。

実行時間 (`ms/step`) とフィッティングの結果は、環境によって変わる。

## 1. 応答と計算コスト

`compare_models` は、名前とモデルの対応 (dict) と載荷履歴を受け取り、各モデルを同じ履歴で計算する。

```{literalinclude} ../../examples/example_compare_models.py
:language: python
:start-after: "# %% 1. 応答と計算コスト"
:end-before: "# %%"
```

```none
model    steps  plastic  NR it/plastic  ms/step  converged
J2     120/120      110           1.00    0.021  yes
AF     120/120      111           3.00    3.211  yes
OW     120/120      111           2.17    3.731  yes
```

| 表の列 | `ModelRun` の属性 | 内容 |
|---|---|---|
| `steps` | `n_steps` | 完了したステップ数 / 載荷履歴のステップ数 |
| `plastic` | `n_plastic_steps` | 塑性であったステップ数 |
| `NR it/plastic` | `iterations_per_plastic_step` | 応力更新の反復回数の、塑性ステップ 1 つあたりの平均 |
| `outer it/step` | `n_outer_iterations` | 応力更新の呼び出し回数 (応力制御と混合制御の外側の反復)。ひずみ制御では常に 1 であり、列を表示しない |
| `ms/step` | `time_per_step`、`elapsed` | 実行時間 |

- **反復回数の平均は塑性ステップで取る。** 弾性ステップの反復は 0 回であり、全ステップで平均すると、降伏の遅いモデルほど小さい値になるためである。
- **実行時間は 1 回の計測値である。** 数倍の差を判別する目的の値であり、数 % の差は有意でない。driver が必要とした計算の時間であるため、ひずみ制御では整合接線の計算を含まず、応力制御と混合制御では含む。
- **UMAT (`FortranIntegrator`) は反復回数と塑性の判定を報告しない。** その 2 列は `—` になる。
- **渡したモデルをそのまま計算する。** フィッティングは行わない。同定したモデルを比較する場合は、`fit_params` の結果の `FitResult.model` を渡す (4 節)。

## 2. 応答の参照

`comparison.runs[名前]` は各モデルの結果 (`ModelRun`) である。応力、ひずみ、状態量の参照方法は driver の結果と同じである ([応答を計算する](simulation.md) の 4 節)。

```{literalinclude} ../../examples/example_compare_models.py
:language: python
:start-after: "# %% 2. 応答の参照"
:end-before: "# %%"
```

```text
J2: 最大応力 384.4 MPa
AF: 最大応力 419.6 MPa
OW: 最大応力 264.1 MPa
(120, 1)
DriverResult
```

| 属性 | 内容 |
|---|---|
| `run.stress`、`run.strain` | 応力、ひずみ。形は `(ステップ数, 成分数)` |
| `run.history(名前)` | 状態量の履歴 |
| `run.result` | driver の結果 (`DriverResult`)。`step_results` などはここから参照する |

## 3. 非収束のモデル

途中で収束しなくなったモデルは、例外にならず、表の 1 行になる。他のモデルの計算は継続する。

```{literalinclude} ../../examples/example_compare_models.py
:language: python
:start-after: "# %% 3. 非収束のモデル"
:end-before: "# %%"
```

```none
model                   steps  plastic  NR it/plastic  ms/step  converged
AF                    120/120      111           3.00    3.141  yes
OW (absurd recovery)    3/120        0              —    4.151  no — stopped at step 3
```

```text
False False 3 3
```

| 属性 | 内容 |
|---|---|
| `comparison.converged` | 全モデルが最後まで収束した場合に `True` |
| `run.converged`、`run.failed_step` | そのモデルの収束と、収束しなかったステップの番号 (0 始まり) |
| `run.stress` など | 完了したステップのみを含む (この例では 3 ステップ) |

呼び出しの誤り (載荷履歴の形がモデルと合わない、など) は例外になる。

## 4. フィットの良さ

`targets=` に試験の記録 (`FitTarget`) を渡すと、各モデルの残差が `fit residual` の列に並ぶ。

```{literalinclude} ../../examples/example_compare_models.py
:language: python
:start-after: "# %% 4. フィットの良さ"
:end-before: "# %%"
```

```none
model          steps  plastic  NR it/plastic  ms/step  fit residual  converged
J2           120/120      110           1.00    0.018          6708  yes
J2 (fitted)  120/120      108           1.00    0.018          5533  yes
AF           120/120      111           3.00    3.204         8.266  yes
OW           120/120      111           2.17    3.680     1.004e+04  yes
```

```text
True
```

- **残差の定義は `fit_params` の目的関数と同じである** (全ターゲットの残差の和)。同定したモデルでは、`run.fit_residual` は `FitResult.residual` と一致する。応力の記録に対しては応力の差の二乗平均であり、平方根が RMS 誤差になる。
- **ターゲットは、それぞれ自身の載荷履歴で計算される。** `load` は応答とコストを比較するための履歴であり、ターゲットの履歴と同じでなくてよい。
- **ターゲットを計算できなかったモデル**は、`fit_residual` が `None` になり、理由が `fit_error` に入る。

この例では、記録を AF の応答から作っているため、AF の残差はノイズの分散 (標準偏差 3 MPa の二乗) にほぼ等しい。J2 (等方硬化) はバウシンガー効果を表現できないため、フィッティング後も残差が大きい。

## 5. パラメータスタディ

`with_params` で作ったモデルを並べると、パラメータを振った比較になる。

```{literalinclude} ../../examples/example_compare_models.py
:language: python
:start-after: "# %% 5. パラメータスタディ"
:end-before: "# %%"
```

```text
gamma=50   最大応力  468.5 MPa   RMS  36.2 MPa
gamma=100  最大応力  419.6 MPa   RMS   2.9 MPa
gamma=200  最大応力  348.6 MPa   RMS  42.6 MPa
gamma=400  最大応力  300.0 MPa   RMS  74.4 MPa
```

- 収束しない値が含まれていても、計算は止まらず、その行が非収束として残る (3 節)。
- 2 つのパラメータの格子は、dict のキーを値の組にすれば同じ形で書ける。

## 6. 解法の比較

モデルの代わりに integrator を渡すと、解法を指定できる。比較できるのは、自動微分と手書きのリターンマッピング、ヤコビアンの `"exact"` と `"broyden"`、Python のモデルと UMAT などである。

```{literalinclude} ../../examples/example_compare_models.py
:language: python
:start-after: "# %% 6. 解法の比較"
:end-before: "# %%"
```

```none
model                    steps  plastic  NR it/plastic  outer it/step  ms/step  converged
AF 1D                  120/120      111           3.00           1.00    3.304  yes
AF 3D, exact Jacobian  120/120      111           3.00           2.85   26.172  yes
AF 3D, Broyden         120/120      111           3.24           2.85   14.117  yes
```

| 引数 | 内容 |
|---|---|
| `driver_factory` | integrator を受け取り、driver を返す関数。上の例は、1 つの軸方向の履歴で、次元の異なるモデルを単軸応力で計算する |
| `driver_factory` を省略 | 載荷履歴から決まる。ひずみ履歴は `StrainDriver`、応力履歴は `StressDriver` (`fit_params`、`crosscheck` と同じ規則) |
| `nr_jacobian="broyden"` | integrator ではなくモデルを渡した項目のすべてに適用される。モデルは `PythonIntegrator` で積分される |

### 解き直しの有無

自動微分の応力更新は、失敗したステップを別の初期値から 1 回解き直す ([概念](../concepts.md) の「非収束との付き合い方」)。`nr_restart=False` の integrator は解き直しを行わない。両者を並べると、解き直しの効果を確認できる。

```{literalinclude} ../../examples/example_compare_models.py
:language: python
:start-after: "# %% 6b. 解き直しの有無"
:end-before: "# %%"
```

```none
model              steps  plastic  NR it/plastic  restarts  ms/step  converged
no restart           0/9        0              —         0  110.581  no — stopped at step 0
restart (default)    9/9        9          12.67         1   31.152  yes
```

- `restarts` は、2 回目の試行で収束したステップの数である。解き直しが 1 つも無い場合、この列は表示しない。
- `NR it/plastic` は 1 回目の試行の反復を含む。解き直しのコストはこの値に現れる。
- ステップごとの値は `run.result.step_results[i].n_restarts` と `.residual_history` (2 回分が連続して入る) で参照する。
- 1 回目の試行で収束するステップの結果は、どちらの integrator でもビット単位で一致する。差が生じるのは、解き直したステップとそれ以降である。

2 つの実装の一致を判定するのは、`compare_models` ではなく [`crosscheck`](verification.md) である (ステップごとに誤差を測り、合否を返す)。

## 7. 作図

```{literalinclude} ../../examples/example_compare_models.py
:language: python
:start-after: "# %% 7. 作図"
```

## 関連ページ

- 試験データに対するパラメータの同定: [試験データに合わせる](tutorial.md)
- 2 つの実装の一致の検査: [検証ワークフロー](verification.md)
- 非収束の仕組み: [概念](../concepts.md)
