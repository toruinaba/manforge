# 試験データに合わせる

単軸の引張・圧縮試験の記録に対して、既存のモデルのパラメータを同定する手順を示す。手動の調整で初期値を定め、`fit_params` で最適化する。計算の基本 (モデル、載荷履歴、結果) は [応答を計算する](simulation.md) を前提とする。

| | 場所 | 内容 |
|---|---|---|
| コード | [`examples/example_identification.py`](https://github.com/toruinaba/returnmap/blob/main/examples/example_identification.py) | このページの各節。実行時間は約 50 秒 |
| ノートブック | [`examples/notebook_tuning.py`](https://github.com/toruinaba/returnmap/blob/main/examples/notebook_tuning.py) | 3 節の調整をスライダーで行う |
| 試験データ | [`examples/data/cyclic_tension_compression.csv`](https://github.com/toruinaba/returnmap/blob/main/examples/data/cyclic_tension_compression.csv) | 合成した記録。Armstrong-Frederick モデルの応答に、標準偏差 3 MPa のノイズを加えた 300 点 |

作図には matplotlib を用いる (`pip install "returnmap[examples]"`)。

## 1. 試験データの読み込み

必要な列は、ひずみと応力の 2 つである。

```{literalinclude} ../../examples/example_identification.py
:language: python
:start-after: "# %% 1. 試験データの読み込み"
:end-before: "# %%"
```

```text
記録 300 点、調整に用いる点 100 点
```

調整の間は記録を間引く。自動微分で解くモデルは 1 ステップに数 ms を要し、300 点の計算 1 回は約 1 秒、3 点おきの 100 点では約 0.3 秒である。全点は 5 節で用いる。

## 2. モデルの選択

単軸試験には `1D` のモデルを用いる ([応答を計算する](simulation.md) の 2 節)。必要なパラメータは `AFKinematic1D.param_meta` で確認する。

```{literalinclude} ../../examples/example_identification.py
:language: python
:start-after: "# %% 2. モデルの選択"
:end-before: "# %%"
```

## 3. 候補の比較

パラメータを変えた候補を同じひずみ履歴で計算し、記録との差を比べる。

```{literalinclude} ../../examples/example_identification.py
:language: python
:start-after: "# %% 3. 候補の比較"
:end-before: "# %%"
```

```text
J2, H=5000             RMS  110.4 MPa
AF start               RMS   49.5 MPa
AF yield up            RMS   19.7 MPa
AF yield up, C_k up    RMS   33.2 MPa
AF by hand             RMS   13.7 MPa
```

| 名前 | 内容 |
|---|---|
| `FitTarget.stress_strain(strain, stress)` | 比較の対象。ひずみ履歴と、それに対する応力の記録 |
| `model.with_params(...)` | 指定したパラメータだけを置き換えたモデル |
| `compare_models(models, load, targets=[...])` | 各モデルを `load` で計算し、`targets` に対する残差を求める ([モデルを比べる](comparing.md)) |
| `run.fit_residual` | 応力の差の二乗平均。平方根は応力の RMS 誤差 [MPa] であり、記録のノイズ (この例では約 3 MPa) と直接比較できる |
| `run.strain`、`run.stress`、`run.history(名前)` | 各候補の応答。記録に重ねて描く場合に用いる |

モデルの種類も同じ表で比較できる。J2 (等方硬化) はバウシンガー効果を表現できないため、パラメータを調整してもこの記録には合わない。

### ノートブックによる調整

`examples/notebook_tuning.py` は、同じ調整をスライダーで行う [marimo](https://marimo.io) のノートブックである。

```bash
pip install marimo        # リポジトリを clone している場合: uv sync --extra notebook
marimo edit examples/notebook_tuning.py
```

| 機能 | 内容 |
|---|---|
| モデルの選択 | 選んだモデルの `param_meta` からスライダーを生成する |
| 再計算 | スライダーを動かすたびに計算し、記録に重ねた図と RMS 誤差を表示する |
| Fit from here | スライダーの値を初期値、スライダーの範囲を探索範囲として、フィッティングを実行する |
| コードの表示 | その時点の値でモデルを構築するコードを表示する |

## 4. フィッティング

手動で調整した値を初期値として、`fit_params` で最適化する。

```{literalinclude} ../../examples/example_identification.py
:language: python
:start-after: "# %% 4. フィッティング"
:end-before: "# %%"
```

```none
収束 True、評価 52 回、RMS 10.5 MPa
{'E': 206000.0, 'nu': 0.3, 'sigma_y0': 279.8, 'C_k': 29365.5, 'gamma': 164.6}
```

| 引数 | 内容 |
|---|---|
| 第 1 引数 (モデル) | クラス、次元、パラメータ以外の設定を与えるテンプレート。パラメータの値は `ParameterSpace` から取る |
| 第 2 引数 (ターゲット) | `FitTarget` の列 |
| 第 3 引数 (`ParameterSpace`) | モデルの全パラメータを、`free` (探索する) か `fixed` (固定する) で与える |
| `nr_jacobian="broyden"` | 自動微分で解くモデルの評価を速くする。既定は `"exact"` |
| `tol` | 最適化の収束判定。ノイズのある記録に対して、既定の値まで詰める必要は無い |
| `progress` | 評価ごとに呼ばれる関数。引数は (評価回数, その残差, 最良の残差) |

| 結果 | 内容 |
|---|---|
| `fit.model` | 同定したパラメータを入れたモデル |
| `fit.model_params` | 同定したパラメータ (dict) |
| `fit.residual` | 最終の残差 (`run.fit_residual` と同じ定義) |
| `fit.success`、`fit.n_evaluations` | 最適化の収束、評価回数 |

- **範囲の指定**: `free` の範囲には、ありうる値の幅を有限の上下限で与える。探索は各変数を範囲の幅で割って行うため、範囲は桁の異なるパラメータ間の尺度になる。`param_meta` の `bounds` は外枠であり、手動の調整で得た見当まで狭める。
- **範囲の端**: 同定した値が範囲の端にある場合は、範囲またはモデルの選択を見直す。
- **計算時間**: 評価 1 回が 0.5 秒以上かかる場合、`SlowFitWarning` が実測時間と短縮の手段を示す。手書きの解法を持たないモデル (AF、OW、損傷モデルなど) は自動微分で解くため、1 ステップに 2〜3 ms を要する。

複数の試験の同時フィッティング、応力制御の試験、パラメータ間の制約は [パラメータフィッティング](fitting.md) を参照。

## 5. 全点での確認と仕上げ

間引いた記録に対して同定したパラメータは、全点の記録に対する最適値ではない。3 点おきでは 1 ステップのひずみ増分が 3 倍になり、その積分誤差がパラメータに吸収されるためである ([応答を計算する](simulation.md) の「刻みの影響」)。

```{literalinclude} ../../examples/example_identification.py
:language: python
:start-after: "# %% 5. 全点での確認"
:end-before: "# %%"
```

```none
AF by hand           100 点:  13.7 MPa   300 点:   7.1 MPa
AF fitted (thinned)  100 点:  10.5 MPa   300 点:   6.0 MPa
```

全点に対する RMS 誤差は 6.0 MPa であり、ノイズの 3 MPa に達していない。間引いた記録での結果を初期値として、全点で再度フィッティングする。

```{literalinclude} ../../examples/example_identification.py
:language: python
:start-after: "# %% 5b. 全点でのフィッティング"
:end-before: "# %%"
```

```none
収束 True、評価 48 回、RMS 2.9 MPa
{'E': 206000.0, 'nu': 0.3, 'sigma_y0': 284.9, 'C_k': 25971.8, 'gamma': 140.1}
```

RMS 誤差はノイズの大きさ (約 3 MPa) まで下がる。この記録の生成に用いた値は `sigma_y0=285`、`C_k=26000`、`gamma=140` であり、同定した値はこれと一致する。

記録の点数が数千点以上の場合、全点でのフィッティングは計算時間の面で現実的でない。その場合は、粗い履歴で計算した応答を全点の記録と曲線どうしで比較する `metric="pcm"` を用いる ([パラメータフィッティング](fitting.md) の「サンプリングが一致しない実験データ」)。

## 6. 結果の保存

```{literalinclude} ../../examples/example_identification.py
:language: python
:start-after: "# %% 6. 結果の保存"
:end-before: "# %%"
```

## 7. 作図

```{literalinclude} ../../examples/example_identification.py
:language: python
:start-after: "# %% 7. 作図"
```

## 関連ページ

- 同定した値の周りでのパラメータスタディ、モデル間の比較: [モデルを比べる](comparing.md)
- フィッティングの設定 (複数の試験、制約、比較の方法): [パラメータフィッティング](fitting.md)
- モデルの一覧と定式化: [組み込みモデルと次元選択](models.md)
- 3D のモデル、応力制御での計算: [応答を計算する](simulation.md)
