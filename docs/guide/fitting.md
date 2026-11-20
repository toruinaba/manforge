# パラメータフィッティング

```python
from returnmap.fitting import FitTarget, ParameterSpace, fit_params

space = ParameterSpace(
    free={
        "sigma_y0": (200.0, (100.0, 500.0)),  # (初期値, (下限, 上限))
        "H":        (500.0, (0.0,  5000.0)),
    },
    fixed={"E": 210_000.0, "nu": 0.3},
)
result = fit_params(
    model,                                              # テンプレート
    [FitTarget.stress_strain(strain_exp, stress_exp)],  # 実験カーブ 1 本 = ターゲット 1 つ
    space,
    method="L-BFGS-B",          # / "Nelder-Mead" / "differential_evolution"
    nr_jacobian="broyden",      # 閉形式フックの無いモデルでは数倍高速 (既定は "exact")
    max_iter=1000,
)
print(result.model_params, result.residual)
fitted = result.model           # フィット結果を入れたモデル (テンプレートと同じクラス・次元・設定)
```

**範囲は探索の物差しでもある。** `"L-BFGS-B"` と `"Nelder-Mead"` は、各自由変数を範囲 (上下限) の幅で割った変数で探索する。硬化係数 (1e4 台) と回復係数 (1e2 台) のように桁の違うパラメータを同じ重みで動かすためで、結果や履歴は書いた単位のまま返る。範囲は「ありうる値の幅」を有限の上下限で書く (上下限の無い変数は、初期値の大きさで割る)。

呼び出しの形は 1 つ — モデル、ターゲットの列、`ParameterSpace`。`space` はモデルの全パラメータを (free / fixed / derived のどれかで) 与える。`model` はテンプレートで、クラス・次元・パラメータ以外の設定 (`J2IsotropicDamageUnilateral` の `c` など) を与える。各評価のモデルは `model.with_params(**params)` で作られ、テンプレート自身のパラメータ値は使われない。`max_iter` / `tol` は最適化器の設定で、return mapping の NR のものではない。

**driver** — `driver_factory` を省略すると、ターゲットの載荷履歴から決まる (ひずみ履歴 → `StrainDriver`、応力履歴 → `StressDriver`。`crosscheck` と同じ規則)。それ以外は渡す — 3D モデルで単軸応力の実験に合わせるなら `driver_factory=lambda i: MixedDriver(i, prescribed_strain_idx=[0])`。優先順位は、ターゲット自身の `driver_factory`、`fit_params` に渡したもの、載荷履歴から決まるもの、の順。

**結果** — `result.model_params` はモデルのパラメータ (`param_names` と同じキー)、`result.model` はそれを入れたモデル。`result.params` は探索空間のすべての名前を持つので、下の `B_ratio` のような補助変数も含む (`model.with_params(**result.params)` は `TypeError` になる)。

**非収束の扱い** — 探索空間には return mapping に実根が無いパラメータ値が普通に含まれる。そのため評価中の非収束は例外ではなく**有限の大ペナルティ** (1e30) としてスコアされ、最適化は続行する (`inf` にしないのは L-BFGS-B の有限差分勾配を汚染しないため)。`FitResult.n_failed` に失敗評価数、`failure_message` / `failure_params` に最初の失敗の再現情報が入る。**全評価が失敗**した場合のみ raise し、メッセージが 2 ケースを区別する: 初期値自体が解けない「fitting cannot start」(初期値を見直す; データ間引きで Δε が広がると起きやすい) と、初期値は解けたが探索が全滅した「no point the optimiser tried was」(bounds を狭める)。

**相関パラメータ — `ParameterSpace`** — `Y < B` のような順序制約は、比を最適化変数にして下限 1.0 を課すと不等式制約なしで表現できる:

```python
from returnmap.fitting import ParameterSpace

space = ParameterSpace(
    free={"Y": (200.0, (50.0, 600.0)),
          "B_ratio": (1.5, (1.0, 5.0))},        # B/Y >= 1 ⇔ Y <= B
    fixed={"E": 210_000.0, "nu": 0.3},
    derived={"B": lambda p: p["Y"] * p["B_ratio"]},
)
result = fit_params(model, targets, space, max_iter=200)
result.params["B_ratio"]   # 探索した比
result.model_params["B"]   # 復元済み。B_ratio は入らない
```

**複数カーブ同時フィット — `FitTarget` / `Observation`** — 2 階層になっているのは計算コストが違うため。`FitTarget` = **1 回の解析** (載荷履歴 1 本 + driver 1 つ、目的関数 1 評価あたり driver run 1 回)。`Observation` = **1 本の実験カーブ** (同一ターゲット上の複数 observation は解析結果を共有、追加コストほぼゼロ):

```python
from returnmap.fitting import FitTarget, Observation
from returnmap.simulation import FieldHistory, FieldType

targets = [
    FitTarget(                       # ひずみ制御: σ11 と ep を同じ解析から比較
        load=FieldHistory(FieldType.STRAIN, strain_hist),
        observations=[
            Observation(values=sigma11_exp, component=0),
            Observation(values=ep_exp, field="ep", weight=1e8),  # スケール差は weight で調整
        ],
    ),
    FitTarget(                       # 応力制御実験: 応力履歴なので StressDriver で回る
        load=FieldHistory(FieldType.STRESS, stress_hist),
        observations=[Observation(values=strain_exp, field="strain", component=0)],
    ),
]
result = fit_params(model, targets, space)
```

ターゲットごとに driver を変えたいときは `FitTarget(..., driver_factory=...)` で渡す。

各 observation の残差二乗和は**データ点数で割られる** (長いカーブが支配するのを防ぐ)。デフォルトの比較は点対点 MSE で、**実験とシミュレーションのサンプリングが一致している必要がある**。

**サンプリングが一致しない実験データ — `metric="pcm"`** — 実験が細かく計測され、シミュレーションを粗い増分で回す場合は、点対点でなく曲線間面積で比較する Partial Curve Mapping を使う:

```python
Observation(values=stress_exp, x_values=strain_exp, metric="pcm", component=0)
```

点数・x 範囲が違ってよく、ヒステリシスループのような多価カーブも扱える。

**実測データのノイズ** — PCM は 2 本のカーブを弧長で対応づける。ノイズのある点列を結んだ折れ線は元のカーブより長く、点数が多いほど長くなる (応力範囲の 0.4 % のノイズで、2000 点なら 1.5 倍、20000 点なら 10 倍)。そのため実測カーブの弧長は、点列から見積もったノイズの大きさの 10 倍の長さの物差しで測る (カーブ自体は間引かず、全点と比べる)。ノイズの無いカーブでは何も変わらない。フィルタや間引きを事前にかける必要は無い。限界: 物差しは降伏の角や載荷反転の先端をまたぐので、弧長が少し短く出る (範囲の 0.4 % のノイズで 1 % 前後、1 % を超えるノイズで数 %)。このとき真のパラメータでも PCM の値は 0 にならず 0.01〜0.05 程度になり、それより小さい差は見分けられない。

注意: 粗い増分は降伏コーナーを弦で切るため**複数パラメータの同時フィットにバイアス**が乗る (実測: 30 増分で H に 12% 誤差、400 増分で消失)。単一パラメータの掃引は不偏。詳細は `fitting/metrics.py` の docstring。
