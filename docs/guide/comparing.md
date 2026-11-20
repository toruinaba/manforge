# モデルを比べる

複数の構成則を同じ載荷履歴で走らせて、応答・計算コスト・フィットの良さを並べる。Python のモデルだけあれば使える (手書きのリターンマッピングや UMAT は要らない)。

```python
from returnmap.simulation import FieldHistory, compare_models

load = FieldHistory.cyclic_strain([0.015, -0.015, 0.015], n_per_segment=40)
comparison = compare_models({"J2": j2, "AF": af, "OW": ow}, load)   # 名前 → モデル
print(comparison.summary())
```

```text
model    steps  plastic  NR it/plastic  ms/step  converged
J2     120/120      110           1.00    0.020  yes
AF     120/120      111           3.00    3.256  yes
OW     120/120      111           2.17    3.739  yes
```

何もフィットしない — 渡したモデルをそのまま走らせる。フィットしたモデルどうしを比べるなら、それぞれ `fit_params` にかけて `FitResult.model` を渡す。

## 応答

`comparison.runs[名前]` が各モデルの結果 (`ModelRun`) で、`DriverResult` と同じ読み方ができる。

```python
for name, run in comparison.runs.items():
    plt.plot(run.strain[:, 0], run.stress[:, 0], label=name)

comparison.runs["AF"].history("alpha")     # 状態量の履歴
comparison.runs["AF"].result               # DriverResult そのもの (step_results など)
```

## 計算コスト

| 表の列 | `ModelRun` の属性 | 中身 |
|---|---|---|
| `steps` | `n_steps` | 完了したステップ数 / 載荷履歴のステップ数 |
| `plastic` | `n_plastic_steps` | 塑性だったステップ数 |
| `NR it/plastic` | `iterations_per_plastic_step` | リターンマッピングの NR 反復数を、塑性ステップ 1 つあたりにしたもの |
| `outer it/step` | `n_outer_iterations` | 応力更新の回数 (`StressDriver` / `MixedDriver` の外側 NR)。ひずみ制御では常に 1 なので列を出さない |
| `ms/step` | `time_per_step`、`elapsed` | 実行時間 |

- **反復数は塑性ステップで平均する。** 弾性ステップは反復が 0 なので、全ステップで平均すると、降伏が遅いモデルほど安く見える。
- **時間は 1 回測っただけの値。** 数倍の差を見分けるためのもので、数 % の差は読まない。その driver が実際に必要とした計算の時間なので、ひずみ制御では整合接線の計算を含まず (使わないので求めない)、応力制御と混合制御では含む。
- UMAT (`FortranIntegrator`) は反復数と塑性の判定を報告しないので、その 2 列は `—` になる。

## 解けなかったモデル

途中で収束しなくなったモデルは例外にならず、表の 1 行になる。ほかのモデルの比較はそのまま続く。

```text
model                   steps  plastic  NR it/plastic  ms/step  converged
AF                    120/120      111           3.00    3.227  yes
OW (absurd recovery)    3/120        0              —    1.843  no — stopped at step 3
```

`run.converged` / `run.failed_step` で分かり、`run.stress` などには完了したステップまでが入っている。全モデルが完走したかは `comparison.converged`。呼び出しの誤り (載荷履歴の形がモデルと合わない、など) は従来どおり例外になる。

## フィットの良さ

実験カーブ (`FitTarget`) を渡すと、各モデルの残差が `fit residual` の列に並ぶ。

```python
from returnmap.fitting import FitTarget

target = FitTarget.stress_strain(strain_exp, stress_exp)
scored = compare_models({"J2": fit.model, "AF": af, "OW": ow}, load, targets=[target])
scored.runs["J2"].fit_residual      # fit.residual と同じ値
```

- 値は `fit_params` の目的関数と同じもの (全ターゲットの残差の和) なので、フィットしたモデルでは `FitResult.residual` と一致する。
- ターゲットは `fit_params` のときと同じく、それぞれ自分の載荷履歴で走る。`load` は応答とコストを比べるための履歴で、ターゲットの履歴と同じでなくてよい。
- ターゲットを走らせられなかったモデルは `fit_residual` が `None` になり、理由が `fit_error` に入る。

## 解き方を比べる、次元の違うモデルを比べる

モデルの代わりに integrator を渡すと、解き方を選べる — 自動微分と手書きのリターンマッピング、`"exact"` と `"broyden"`、Python のモデルと UMAT。

```python
from returnmap.simulation import MixedDriver, PythonIntegrator, StrainDriver

def uniaxial_stress(integrator):                 # 次元によらず単軸応力
    if integrator.ntens == 1:
        return StrainDriver(integrator)
    return MixedDriver(integrator, prescribed_strain_idx=[0])

compare_models(
    {
        "AF 1D": af,
        "AF 3D, exact Jacobian": PythonIntegrator(af_3d),
        "AF 3D, Broyden": PythonIntegrator(af_3d, nr_jacobian="broyden"),
    },
    load,
    driver_factory=uniaxial_stress,
)
```

```text
model                    steps  plastic  NR it/plastic  outer it/step  ms/step  converged
AF 1D                  120/120      111           3.00           1.00    3.177  yes
AF 3D, exact Jacobian  120/120      111           3.00           2.85   26.059  yes
AF 3D, Broyden         120/120      111           3.24           2.85   14.247  yes
```

`driver_factory` を省略すると載荷履歴から決まる (ひずみ履歴 → `StrainDriver`、応力履歴 → `StressDriver`。`fit_params`、`crosscheck` と同じ規則)。factory は integrator を受け取るので、上のように 1 つの軸方向の履歴で次元の違うモデルを走らせられる。素のモデルは `PythonIntegrator` で積分され、`compare_models(..., nr_jacobian="broyden")` はその全部に効く。

2 つの実装が**一致するか**を確かめるのは `compare_models` ではなく [`crosscheck`](verification.md) の役割である (ステップごとに誤差を測り、合否を返す)。

実行例: `examples/example_compare_models.py`
