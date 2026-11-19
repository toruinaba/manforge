# シミュレーション実行

```python
from manforge.simulation import (
    StrainDriver, StressDriver, PythonIntegrator, FieldHistory, FieldType)

# ひずみ制御
result = StrainDriver(PythonIntegrator(model)).run(load)
result.stress    # np.ndarray (N, ntens)
result.strain    # np.ndarray (N, ntens)

# 状態変数の履歴 — 事前の指定は不要。結果から名前で取り出す
result.history("ep")       # np.ndarray (N,)
result.history("alpha")    # np.ndarray (N, ntens)

# 応力制御 (目標応力を外側 NR で満たすひずみを解く)
result = StressDriver(PythonIntegrator(model)).run(
    FieldHistory(FieldType.STRESS, "Stress", stress_target))
```

**混合制御** — 一部成分はひずみ指定・残りは応力指定 (例: 3D モデルで ε11 を駆動し σ22..σ23 = 0 の単軸応力試験を再現):

```python
from manforge.simulation import MixedDriver
driver = MixedDriver(PythonIntegrator(model), prescribed_strain_idx=[0])
result = driver.run(load_eps11)   # load.data は (N, 1) — ε11 のみ
```

**iter_run** — ステップごとに `DriverStep` を受け取り、途中分岐・打切りができる:

```python
for step in StrainDriver(PythonIntegrator(model)).iter_run(load):
    if step.result.is_plastic:
        print(f"Plasticity onset at step {step.i}")
        break
```

各ステップの詳細 (`dlambda`, `n_iterations`, NR 残差履歴 `residual_history`) は `result.step_results[i]` / `step.result` から参照できる。`StressDriver`/`MixedDriver` では `step.n_outer_iter` / `step.residual_inf` で外側 NR を確認できる。
