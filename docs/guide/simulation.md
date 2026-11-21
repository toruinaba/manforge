# シミュレーション実行

モデルに載荷履歴を与えて、応力・ひずみ・状態量の履歴を計算する。integrator が 1 ステップの応力更新 (UMAT 1 回分) を行い、driver がそれを履歴に沿って進める。

```python
from returnmap.models import AFKinematic3D
from returnmap.simulation import (
    StrainDriver, StressDriver, PythonIntegrator, FieldHistory, FieldType)

model = AFKinematic3D.at_defaults()
load = FieldHistory.cyclic_strain([0.01, -0.01, 0.01], n_per_segment=40, ntens=model.ntens)

# ひずみ制御
result = StrainDriver(PythonIntegrator(model)).run(load)
result.stress    # np.ndarray (N, ntens)
result.strain    # np.ndarray (N, ntens)

# 状態変数の履歴 — 事前の指定は不要。結果から名前で取り出す
result.history("ep")       # np.ndarray (N,)
result.history("alpha")    # np.ndarray (N, ntens)

# 応力制御 (目標応力を外側 NR で満たすひずみを解く)
result = StressDriver(PythonIntegrator(model)).run(
    FieldHistory(FieldType.STRESS, stress_target))
```

**混合制御** — 一部成分はひずみ指定・残りは応力指定 (例: 3D モデルで ε11 を駆動し σ22..σ23 = 0 の単軸応力試験を再現):

```python
from returnmap.simulation import MixedDriver
driver = MixedDriver(PythonIntegrator(model), prescribed_strain_idx=[0])
result = driver.run(load_eps11)   # load.data は (N, 1) — ε11 のみ
```

**iter_run** — ステップごとに `DriverStep` を受け取り、途中分岐・打切りができる:

```python
for step in StrainDriver(PythonIntegrator(model)).iter_run(load):
    if step.result.is_plastic:
        print(f"Plasticity onset at step {step.index}")
        break
```

## 載荷履歴を作る

載荷履歴は `FieldHistory` で、ひずみか応力の種類 (`FieldType`) と、ステップごとの値の配列を持つ。配列は**累積値** (増分ではない) で、形は `(ステップ数, ntens)` である。1 成分だけの履歴は 1 次元の配列でもよい。

```python
import numpy as np
from returnmap.simulation import FieldHistory, FieldType

load = FieldHistory(FieldType.STRAIN, np.linspace(0.0, 0.02, 101)[1:])   # 配列をそのまま渡す
```

よく使う形は、作る関数がある (応力の履歴は `_strain` を `_stress` に替えた名前)。

| 関数 | 形 |
|---|---|
| `FieldHistory.cyclic_strain(peaks, n_per_segment=20)` | 0 から始めて、`peaks` の値を順に直線で結ぶ |
| `FieldHistory.triangular_strain(amplitude, n_cycles)` | 一定振幅の三角波 (`mean=` で平均値) |
| `FieldHistory.sine_strain(amplitude, n_cycles, n_per_cycle=80)` | 正弦波 |
| `FieldHistory.decaying_cyclic_strain(amplitude, n_cycles, decay=0.5)` | 振幅が指数的に変わる繰返し |

- **多成分のモデル**には `ntens=model.ntens` を渡す。動かす成分は `component=` で選び (既定は 0 = 11 成分)、ほかの成分は 0 になる。複数の成分を同時に動かす履歴は、配列を自分で作って渡す。
- **成分の並びとせん断の扱い**は [概念](../concepts.md) の Voigt 規約に従う。ひずみのせん断成分は工学ひずみ (γ = 2ε) で与える。
- **`*1D` のモデルは単軸応力**である。3D のモデルにひずみの 11 成分だけを与えると、横方向が拘束された単軸ひずみになる。3D のモデルで単軸応力の試験を再現するには、上の混合制御を使う。
- **刻みの細かさは精度に効く。** 1 ステップは後退 Euler で積分するので、粗い刻みは応答に誤差を残す。刻みを変えて結果が動かないことを確かめるとよい。

## 解けなかったとき

リターンマッピングが収束しなかったステップがあると、既定では `RuntimeError` になる。値として受け取るには、integrator を `raise_on_nonconverged=False` で作る。

```python
result = StrainDriver(PythonIntegrator(model, raise_on_nonconverged=False)).run(load)
result.converged       # False なら途中で止まっている
result.failed_step     # 止まったステップの番号 (0 始まり)。完走なら None
len(result.stress)     # 完了したステップの数。解けなかったステップは入らない
```

解けないのは、たいてい 1 ステップの増分が大きすぎるときである。フレームワークは 2 つの手当てを持つ。

- **解き直し (既定で有効)**: 自動微分の NR は、失敗したステップを塑性側の初期値から 1 回解き直す。`result.step_results[i].n_restarts` に残る。
- **増分の分割 (選んで使う)**: `SubSteppingIntegrator(PythonIntegrator(model))` は、解けなかった増分を 2、4、8 … 分割して解き直す。

どちらでも解けないときは、載荷履歴の刻みを細かくする。詳しくは [概念](../concepts.md) の「非収束との付き合い方」を参照。

## ステップごとの情報

各ステップの詳細 (`dlambda`, `n_iterations`, NR 残差履歴 `residual_history`) は `result.step_results[i]` / `step.result` から参照できる。`StressDriver`/`MixedDriver` では `step.n_outer_iter` / `step.residual_inf` で外側 NR を確認できる。
