# 概念

## 三層構造

```
MaterialModel  ──defines──▶  yield_function / update_state / state_residual / flow
      │
      ▼
  Integrator   ──wraps──▶  stress_update(strain_inc, stress_n, state_n) → StressUpdateResult
      │                    increment(strain_inc, stress_n, state_n)     → Increment
      ▼
    Driver     ──steps──▶  run(load) → DriverResult
                            iter_run(load) → Iterator[DriverStep]
```

**import 元は 7 つのパッケージ** — どの名前も 1 つのパッケージから取る:

| パッケージ | 中身 |
|---|---|
| `manforge.core` | モデルを書く: `MaterialModel`、状態変数の宣言 (`Implicit` / `Explicit` / `NTENS` / `SCALAR`)、次元 (`SOLID_3D` 等)、`hand_block`、`verified_against_fortran`、結果型 |
| `manforge.models` | 参照モデル (J2 / 損傷 / AF / OW / YU) |
| `manforge.simulation` | integrator、driver、載荷履歴 (`FieldHistory`)、`FortranModule` |
| `manforge.fitting` | `fit_params`、`FitTarget`、`ParameterSpace` |
| `manforge.verification` | 検査: `check_hand_blocks`、`check_solution`、`check_tangent`、`crosscheck`、`compare_jacobians`、`check_bindings` |
| `manforge.service` | WebAPI 向けの契約層 |
| `manforge.utils` | モデル用の補助関数 (smooth 近似、主値分解) |

その下のサブモジュール (`manforge.core.state`、`manforge.simulation.driver` など) はコードの置き場所で、文書化されたインターフェースではない。

## Integrator の種類

| クラス | 内部ソルバ | 用途 |
|--------|----------|------|
| `PythonIntegrator(model)` | auto (user_defined があれば使用、なければ NR) | 通常用途 |
| `PythonNumericalIntegrator(model)` | numerical_newton 固定 | NR 強制 (検証時) |
| `PythonAnalyticalIntegrator(model)` | user_defined 固定 | 閉形式解と比較する場合 |
| `FortranIntegrator.from_model(fortran, subroutine, model)` | Fortran f2py | UMAT クロス検証 |
| `SubSteppingIntegrator(integrator)` | 上記をラップ | 大きい Δε で収束しない場合 |

**`nr_jacobian="broyden"`** (コンストラクタ kwarg、デフォルト `"exact"`) — NR ヤコビアンをステップ初回だけ autodiff で厳密計算し、以降は Broyden rank-1 更新で済ませる。残差が減らないと厳密再計算に自動フォールバック。実測で YU 級モデル 2〜4 倍高速 (反復 1〜2 回増)。**consistent tangent は常に厳密**なので精度への影響はない。閉形式フックを持たないモデル (AF/OW 等) の長い履歴・フィッティングで推奨。

## Voigt 規約と FlowVector

3D ソリッドの応力・ひずみは 6 成分 `[11, 22, 33, 12, 13, 23]` (ABAQUS UMAT 規約)。剪断成分の扱いが**量の種類で異なる 2 層契約**になっている:

- **応力系** (σ, s, 背応力 α など): **physical shear** — テンソル成分そのもの
- **ひずみ系** (ε, ε_p, Δε): **engineering shear** (γ12 = 2ε12) — ABAQUS `DSTRAN`・弾性剛性 `σ = C:ε`・driver 入力すべてこちら

塑性流れ方向 n は同じ方向に 2 つの Voigt 表現 (剪断 ×2 差) があり、取り違えると**単軸/二軸では消える silent error** になる。そこで `flow(state)` は生の ndarray ではなく `FlowVector` を返し、使用側が規約を名指しする:

```python
n = model.flow(state)
C @ n.strain_like          # engineering shear — Δε_p = Δλ·n、C@n、残差のひずみ次元
n.stress_like              # physical shear — 背応力など応力系内部変数の発展則
n * 2.0                    # TypeError — 規約を名指ししない演算は拒否される
```

オーバーライドの方法は [モデルを作る](guide/writing-models.md) の「flow のオーバーライド」、詳細は API リファレンスの [`FlowVector`](reference/core.md) を参照。

## 非収束との付き合い方

材料点の return mapping は収束しないことがあり、フレームワークは層ごとに異なる扱いを用意している。

**収束判定はスケール不変** — NR の許容値は `tol × max(1, ‖σ_trial‖_∞)` (driver 外側 NR は `tol + rtol·‖σ_target‖_∞`)。MPa でも Pa でも同じ挙動になる。

**`tol` は「その対象の許容値」** — integrator と driver では自分の NR の収束、検査 (`check_hand_blocks` / `check_solution` / `check_tangent` / `compare_jacobians` / `check_bindings`) では合否 (誤差の測り方は全検査で同じ — [検証ワークフロー](guide/verification.md))、`fit_params` では最適化器の収束を指す。2 つを同じ引数リストに並べない: `crosscheck` の `stress_tol` / `state_tol` / `tangent_tol` は合否で、driver の外側 NR の設定は `driver_factory` が作る driver が持つ。

**Δλ < 0 は非収束扱い** — 残差系は Δλ < 0 側にも根を持つが、負の塑性乗数は KKT 条件違反 (塑性ひずみの巻き戻り)。残差ノルムはこの根を区別できないため、フレームワークが符号で弾く。大きい Δε で σ_trial が許容枝の外に出ると起きる。手書きのリターンマッピング (`user_defined_return_mapping`) の結果も同じ条件で確かめる — 閉形式解はどんな入力にも何かの値を返すので、Δλ < 0 や非有限の値 (`inf` / `nan`) は `converged=True` と報告されていても非収束として扱う。

**増分の自動分割** — `SubSteppingIntegrator(integrator, max_substeps=16)` は非収束を検知して同じ増分を 2 → 4 → 8 分割で再試行する。opt-in のラッパで、収束する増分では結果・反復数・接線が一切変わらない。詳細は `simulation/integrator/substepping.py` の docstring。

**非収束を値として受け取る** — 既定では `RuntimeError`。スイッチは integrator の `raise_on_nonconverged` 1 つだけで、driver は自分の外側 NR (`StressDriver` / `MixedDriver`) の失敗にも同じ設定を使う。`PythonIntegrator(model, raise_on_nonconverged=False)` にすると:

- `iter_run` は `DriverStep.converged=False` の step を yield して停止する。
- `run` は完了した step だけを返し、`result.converged=False` / `result.failed_step` で知らせる。失敗した step は非収束イテレートなので `result.stress` などの配列には入らない。

**フィッティング中の非収束はペナルティ** — 探索空間には解の存在しないパラメータ値が普通に含まれるため、`fit_params` は非収束の評価を例外にせず有限の大ペナルティとして扱い、最適化を続行する ([パラメータフィッティング](guide/fitting.md))。
