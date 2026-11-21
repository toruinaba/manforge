# チュートリアル: 試験データにモデルを合わせる

既存のモデルを使って、引張・圧縮試験の記録に合うパラメータを探す。手でパラメータを動かして当たりを付け、最後に自動で仕上げる。モデルを自作する必要は無い。

通しのスクリプトは `examples/example_identification.py` (約 50 秒)、手で合わせる部分のノートブックは `examples/notebook_tuning.py` にある。ここで使う試験データ `examples/data/cyclic_tension_compression.csv` は合成したもので (Armstrong-Frederick モデルの応答にノイズを足した 300 点)、試験機が書き出す CSV の代わりである。

## 1. インストール

```bash
pip install "returnmap[examples]"
```

`[examples]` は図を描くための matplotlib を足す。計算だけなら要らない。

## 2. 試験データを読む

ひずみと応力の 2 列があればよい。読み込みは numpy で足りる。

```python
import numpy as np

record = np.loadtxt("examples/data/cyclic_tension_compression.csv", delimiter=",", skiprows=1)
strain_all, stress_all = record[:, 0], record[:, 1]     # ひずみ [-]、応力 [MPa]

strain, stress = strain_all[::3], stress_all[::3]       # 調整の間は 3 点おきに使う
```

**調整の間は間引く。** 閉形式の解を持たないモデルは 1 ステップに数 ms かかるので、300 点そのままだと 1 回の計算が 1 秒近くになる。3 点おきなら 0.3 秒で、パラメータを動かして見比べるのに向く。全点は最後の仕上げで使う (6 節)。

## 3. モデルを選ぶ

単軸の試験には `*1D` のモデルを使う。応力は単軸応力になる (3D モデルにひずみ ε11 だけを与えると、横を拘束した単軸**ひずみ**になり、引張試験にならない。3D モデルで単軸応力にする方法は [シミュレーション実行](simulation.md) の混合制御)。

```python
import returnmap.models as models
from returnmap.models import AFKinematic1D

[name for name in models.__all__ if name.endswith("1D")]
# ['J2Isotropic1D', 'J2IsotropicDamage1D', ..., 'AFKinematic1D', 'ChabocheKinematic1D', ..., 'YUKinematic1D']

for name, meta in AFKinematic1D.param_meta.items():      # 何を与えるモデルか
    print(name, meta.doc, meta.unit, meta.default, meta.bounds)
# E         Young's modulus                    MPa  210000.0  (1000.0, 1000000.0)
# nu        Poisson's ratio                         0.3       (0.0, 0.49)
# sigma_y0  Initial yield stress               MPa  250.0     (1.0, 5000.0)
# C_k       Kinematic hardening modulus (AF)   MPa  20000.0   (0.0, 1000000.0)
# gamma     Dynamic recovery coefficient (AF)       100.0     (0.0, 10000.0)
```

`param_meta` は各パラメータの意味・単位・目安の値・ありうる範囲を持つ。モデルの構築は全パラメータをキーワードで渡す (既定値は無い)。

```python
ELASTIC = dict(E=206_000.0, nu=0.3)          # ヤング率は記録の弾性勾配から読む
start = AFKinematic1D(**ELASTIC, sigma_y0=250.0, C_k=20_000.0, gamma=100.0)
```

## 4. 手で合わせる

パラメータを変えた候補を並べ、同じひずみ履歴で走らせて、記録との差を見る。`with_params` は一部のパラメータだけ変えたモデルを作り、`compare_models` の `targets=` は各候補を記録に対して採点する。

```python
from returnmap.fitting import FitTarget
from returnmap.models import J2Isotropic1D
from returnmap.simulation import FieldHistory, FieldType, compare_models

load = FieldHistory(FieldType.STRAIN, strain)
target = FitTarget.stress_strain(strain, stress)        # 「このひずみでこの応力になるはず」

candidates = {
    "J2, H=5000": J2Isotropic1D(**ELASTIC, sigma_y0=300.0, H=5_000.0),
    "AF start": start,
    "AF yield up": start.with_params(sigma_y0=290.0),
    "AF yield up, C_k up": start.with_params(sigma_y0=290.0, C_k=26_000.0),
    "AF by hand": start.with_params(sigma_y0=290.0, C_k=26_000.0, gamma=135.0),
}
tuned = compare_models(candidates, load, targets=[target])
for name, run in tuned.runs.items():
    print(f"{name:22s} RMS {np.sqrt(run.fit_residual):6.1f} MPa")
```

```text
J2, H=5000             RMS  110.4 MPa
AF start               RMS   49.5 MPa
AF yield up            RMS   19.7 MPa
AF yield up, C_k up    RMS   33.2 MPa
AF by hand             RMS   13.7 MPa
```

- **残差の平方根は、応力の RMS 誤差 [MPa] として読める。** `fit_residual` は応力の差の二乗平均なので、記録のノイズ (ここでは約 3 MPa) と直接比べられる。
- **図にするなら** `run.strain[:, 0]` と `run.stress[:, 0]` を記録に重ねる。`run.history("alpha")` などで状態量も取れる。
- **モデルの種類も同じ表で比べられる。** J2 (等方硬化) はバウシンガー効果を表せないので、どう合わせてもこの記録には合わない。

### スライダーで動かす (marimo)

`examples/notebook_tuning.py` は、同じことをスライダーで行う [marimo](https://marimo.io) のノートブックである。モデルを選ぶと、そのモデルの `param_meta` からスライダーが作られる。動かすたびに計算し直して、記録に重ねた図と RMS 誤差を出す。合ってきたら「Fit from here」で、スライダーの値を初期値に、スライダーの範囲を探索範囲にしてフィットする。その時点の値でモデルを作るコードも表示するので、スクリプトに持ち帰れる。

```bash
pip install marimo        # リポジトリを clone しているなら: uv sync --extra notebook
marimo edit examples/notebook_tuning.py
```

## 5. 自動で合わせる

手で合わせた値を初期値にして、`fit_params` で仕上げる。

```python
from returnmap.fitting import ParameterSpace, fit_params

def space_around(model):
    """AF の 3 パラメータを、model の値から探す。"""
    return ParameterSpace(
        free={                               # 名前: (初期値, (下限, 上限))
            "sigma_y0": (model.sigma_y0, (150.0, 450.0)),
            "C_k": (model.C_k, (5_000.0, 80_000.0)),
            "gamma": (model.gamma, (20.0, 500.0)),
        },
        fixed=ELASTIC,
    )

by_hand = candidates["AF by hand"]
fit = fit_params(
    by_hand, [target], space_around(by_hand),
    nr_jacobian="broyden",                   # 自動微分で解くモデルでは速くなる
    tol=1e-6,                                # ノイズのある記録に 1e-12 まで詰めても意味が無い
    progress=print,                          # 評価のたびに (回数, その残差, 最良の残差)
)
fit.model_params     # {'E': 206000.0, 'nu': 0.3, 'sigma_y0': 279.8, 'C_k': 29365.5, 'gamma': 164.6}
fit.model            # その値を入れたモデル
```

- **範囲は「ありうる値の幅」を書く。** 探索は各変数を範囲の幅で割って行うので、範囲が桁の違うパラメータどうしの物差しになる。`param_meta` の `bounds` はありうる値の外枠なので、手で合わせて見当が付いた幅まで狭める。
- **時間がかかるときは警告が出る。** 1 回の評価に 0.5 秒以上かかると、`SlowFitWarning` が実測の時間と理由、短くする手段を知らせる。AF や OW、損傷モデルの 1D は閉形式の解を持たず、自動微分で解くので 1 ステップ 2〜3 ms かかる (J2 の 1D と 3D、YU の 3D と平面応力は手書きの解法を持つ)。上の例は 52 回の評価で十数秒。
- **結果は `fit.model`。** `fit.success`、`fit.n_evaluations`、`fit.residual` も見る。パラメータが範囲の端に張り付いていたら、範囲かモデルを疑う。

フィッティングの詳しい使い方 (複数の試験の同時フィット、応力制御の試験、パラメータ間の制約) は [パラメータフィッティング](fitting.md)。

## 6. 全点で確かめて仕上げる

間引いた記録に合わせたパラメータは、全点の記録に対しては最良ではない。3 点おきにすると 1 ステップのひずみ増分が 3 倍になり、増分が大きいぶんの積分誤差をパラメータが吸収するためである。

```python
load_all = FieldHistory(FieldType.STRAIN, strain_all)
target_all = FitTarget.stress_strain(strain_all, stress_all)

check = compare_models({"by hand": by_hand, "fitted (thinned)": fit.model},
                       load_all, targets=[target_all])
```

```text
                     間引いた 100 点    全 300 点
AF by hand                13.7 MPa      7.1 MPa
AF fitted (thinned)       10.5 MPa      6.0 MPa
```

全点でも 6 MPa 残っていて、ノイズの 3 MPa まで下がっていない。間引いたフィットの結果を初期値にして、全点でもう一度フィットする。

```python
final = fit_params(fit.model, [target_all], space_around(fit.model),   # 範囲は同じ、初期値だけ変える
                   nr_jacobian="broyden", tol=1e-6)
np.sqrt(final.residual)      # 2.9 MPa — ノイズの大きさ
final.model_params           # sigma_y0=284.9, C_k=25971.8, gamma=140.1
```

この記録は `sigma_y0=285`、`C_k=26000`、`gamma=140` で作ってあるので、元の値に戻っている。48 回の評価で 20〜30 秒かかる。

**記録の点数がずっと多いとき** (数千点以上) は、全点でのフィットが現実的でなくなる。そのときは、シミュレーションを粗い履歴で走らせて、全点の記録と曲線どうしで比べる `metric="pcm"` を使う ([パラメータフィッティング](fitting.md) の「サンプリングが一致しない実験データ」)。

## 7. パラメータを振る

合わせた値のまわりでパラメータを振るのも、同じ `compare_models` でできる。

```python
study = compare_models(
    {f"gamma={g:g}": final.model.with_params(gamma=g) for g in (80.0, 110.0, 140.0, 200.0)},
    load_all, targets=[target_all],
)
for name, run in study.runs.items():
    print(f"{name:10s} peak {run.stress[:, 0].max():6.1f} MPa   RMS {np.sqrt(run.fit_residual):5.1f} MPa")
```

```text
gamma=80   peak  528.8 MPa   RMS  43.0 MPa
gamma=110  peak  492.0 MPa   RMS  19.6 MPa
gamma=140  peak  459.6 MPa   RMS   2.9 MPa
gamma=200  peak  413.1 MPa   RMS  28.3 MPa
```

途中で解けなくなる値があっても止まらず、表にその行が出る ([モデルを比べる](comparing.md))。2 つのパラメータの格子も、辞書のキーを組にすれば同じ書き方になる。

## 8. 結果を残す

```python
run = study.runs["gamma=140"]
np.savetxt("af_fitted.csv",
           np.column_stack([run.strain[:, 0], run.stress[:, 0], run.history("ep")]),
           delimiter=",", header="strain,stress,ep", comments="")
```

## 次に読むもの

- 使えるモデルと次元の選び方: [組み込みモデルと次元選択](models.md)
- 3D や応力制御で走らせる: [シミュレーション実行](simulation.md)
- モデルどうしの比較と計算コスト: [モデルを比べる](comparing.md)
- 自分のモデルを足して、検査して、UMAT まで進める: [新しいモデルを足す](adding-a-model.md)
- モデルに書けるフックの一覧: [モデルを作る](writing-models.md)
