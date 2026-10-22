# manforge

Fortran UMAT (Abaqus) の検証・フィッティング用 Python フレームワーク。降伏関数と硬化則を実装するだけで、return mapping / consistent tangent (autograd 自動微分) / パラメータフィッティング / Fortran クロス検証がすべて使える。

- autograd による **consistent tangent 自動計算** — 手動でアルゴリズム接線を導出する必要なし
- **汎用 NR ソルバ** — スカラー Δλ / ベクトル `[σ, Δλ, q]` の 2 経路を自動選択。Broyden 更新 (`nr_jacobian="broyden"`) で数倍高速化
- **多次元対応** — 3D ソリッド / 平面ひずみ / 平面応力 / 1D を `StressDimension` で統一管理
- **組み込みモデル** — J2 等方硬化 / Armstrong-Frederick / Ohno-Wang を同梱
- **サービス契約層** — WebAPI などへの組み込み用に plain な入出力の `ModelRegistry` / `run_simulation` を提供

詳細な API 仕様は各モジュールの docstring が正。本 README は入口と判断基準を提供する。

---

## 1. Installation

現状は PyPI 未公開のため、ソースからの導入が正:

```bash
git clone <this-repo> && cd manforge
uv sync --extra dev          # テスト依存含む
uv sync --all-extras         # matplotlib / Fortran ビルド依存含む

# Fortran モジュールのビルド (gfortran 必須、無ければ Docker: make docker-build)
uv run manforge build fortran/abaqus_stubs.f90 fortran/j2_isotropic_3d.f90 --name j2_isotropic_3d
```

PyPI 公開後は `pip install manforge` (+ extras: `[dev]` / `[examples]` / `[fortran]`)。

---

## 2. Quick Start

```python
import numpy as np
from manforge.models import J2Isotropic3D
from manforge.simulation import StrainDriver, PythonIntegrator
from manforge.simulation.types import FieldHistory, FieldType

model = J2Isotropic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)

strain_history = np.zeros((100, 6))
strain_history[:, 0] = np.linspace(0.0, 5e-3, 100)  # ε11 を増加
load = FieldHistory(FieldType.STRAIN, "Strain", strain_history)

result = StrainDriver(PythonIntegrator(model)).run(load)
print(f"Final σ11: {result.stress[-1, 0]:.1f} MPa")
```

繰返し載荷は `FieldHistory` のコンストラクタ群で生成できる:

```python
load = FieldHistory.cyclic_strain(peaks=[5e-3, -5e-3, 5e-3], n_per_segment=25, ntens=6)
# 他: triangular_strain / sine_strain / decaying_cyclic_strain (+ 各 stress 版)
```

実行できる使用例は `examples/` を参照。

---

## 3. Concepts

### 3.1 三層構造

```
MaterialModel  ──defines──▶  yield_function / update_state / state_residual / flow
      │
      ▼
  Integrator   ──wraps──▶  stress_update(strain_inc, stress_n, state_n) → StressUpdateResult
      │                    return_mapping(stress_trial, state_n)         → ReturnMappingResult
      ▼
    Driver     ──steps──▶  run(load) → DriverResult
                            iter_run(load) → Iterator[DriverStep]
```

### 3.2 Integrator の種類

| クラス | 内部ソルバ | 用途 |
|--------|----------|------|
| `PythonIntegrator(model)` | auto (user_defined があれば使用、なければ NR) | 通常用途 |
| `PythonNumericalIntegrator(model)` | numerical_newton 固定 | NR 強制 (検証時) |
| `PythonAnalyticalIntegrator(model)` | user_defined 固定 | 閉形式解と比較する場合 |
| `FortranIntegrator.from_model(fortran, subroutine, model)` | Fortran f2py | UMAT クロス検証 |
| `SubSteppingIntegrator(integrator)` | 上記をラップ | 大きい Δε で収束しない場合 |

**`nr_jacobian="broyden"`** (コンストラクタ kwarg、デフォルト `"exact"`) — NR ヤコビアンをステップ初回だけ autodiff で厳密計算し、以降は Broyden rank-1 更新で済ませる。残差が減らないと厳密再計算に自動フォールバック。実測で YU 級モデル 2〜4 倍高速 (反復 1〜2 回増)。**consistent tangent は常に厳密**なので精度への影響はない。閉形式フックを持たないモデル (AF/OW 等) の長い履歴・フィッティングで推奨。

### 3.3 Voigt 規約と FlowVector

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

詳細 (オーバーライド方法含む) は § 5.3 と `core/state.py` の docstring。

### 3.4 非収束との付き合い方

材料点の return mapping は収束しないことがあり、フレームワークは層ごとに異なる扱いを用意している。

**収束判定はスケール不変** — NR の許容値は `tol × max(1, ‖σ_trial‖_∞)` (driver 外側 NR は `tol + rtol·‖σ_target‖_∞`)。MPa でも Pa でも同じ挙動になる。

**Δλ < 0 は非収束扱い** — 残差系は Δλ < 0 側にも根を持つが、負の塑性乗数は KKT 条件違反 (塑性ひずみの巻き戻り)。残差ノルムはこの根を区別できないため、フレームワークが符号で弾く。大きい Δε で σ_trial が許容枝の外に出ると起きる。

**増分の自動分割** — `SubSteppingIntegrator(integrator, max_substeps=16)` は非収束を検知して同じ増分を 2 → 4 → 8 分割で再試行する。opt-in のラッパで、収束する増分では結果・反復数・接線が一切変わらない。詳細は `simulation/integrator/substepping.py` の docstring。

**非収束を値として受け取る** — 既定では `RuntimeError`。`PythonIntegrator(model, raise_on_nonconverged=False)` にすると driver が `DriverStep.converged=False` の step を yield して停止し、呼び出し側が部分履歴を使って判断できる。

**フィッティング中の非収束はペナルティ** — 探索空間には解の存在しないパラメータ値が普通に含まれるため、`fit_params` は非収束の評価を例外にせず有限の大ペナルティとして扱い、最適化を続行する (§ 4.3)。

---

## 4. モデルを使う

### 4.1 組み込みモデルと次元選択

モデルファミリごとに親クラス (物理) + 次元特化の具象クラスがある:

```python
from manforge.models import (
    J2Isotropic3D, J2IsotropicPS, J2Isotropic1D,    # J2 等方硬化
    AFKinematic3D, AFKinematicPS, AFKinematic1D,    # Armstrong-Frederick
    OWKinematic3D, OWKinematicPS, OWKinematic1D,    # Ohno-Wang
)
model = AFKinematic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, C_k=8_000.0, gamma=80.0)
```

| 次元 | NTENS | 想定要素 |
|------|-------|---------|
| `*3D` (`SOLID_3D`) | 6 | C3D8 等 3D ソリッド |
| — (`PLANE_STRAIN`) | 4 | CPE4 / CAX4 |
| `*PS` (`PLANE_STRESS`) | 3 | CPS4 / シェル |
| `*1D` (`UNIAXIAL_1D`) | 1 | 1D トラス / フィッティング用 |

平面ひずみは専用クラスが無いため親クラスに直接 `dimension=` を渡す:

```python
from manforge.models import J2Isotropic
from manforge.core.dimension import PLANE_STRAIN
model_pe = J2Isotropic(dimension=PLANE_STRAIN, E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)
```

### 4.2 シミュレーション実行

```python
from manforge.simulation import StrainDriver, StressDriver, PythonIntegrator
from manforge.simulation.types import FieldHistory, FieldType

# ひずみ制御
result = StrainDriver(PythonIntegrator(model)).run(load)
result.stress    # np.ndarray (N, ntens)
result.strain    # np.ndarray (N, ntens)

# 状態変数も収集
result = StrainDriver(PythonIntegrator(model)).run(
    load, collect_state={"ep": FieldType.STRAIN})
result.fields["ep"].data   # np.ndarray (N,)

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

### 4.3 パラメータフィッティング

```python
from manforge.simulation.driver import StrainDriver
from manforge.fitting import fit_params

fit_config = {
    "sigma_y0": (200.0, (100.0, 500.0)),  # (初期値, (下限, 上限))
    "H":        (500.0, (0.0,  5000.0)),
}
result = fit_params(
    model,
    driver_factory=lambda i: StrainDriver(i),
    exp_data={"strain": strain_exp, "stress": stress_exp},
    fit_config=fit_config,
    fixed_params={"E": 210_000.0, "nu": 0.3},
    method="L-BFGS-B",          # / "Nelder-Mead" / "differential_evolution"
    nr_jacobian="broyden",      # 閉形式フックの無いモデルでは数倍高速
    max_iter=1000,
)
print(result.params, result.residual)
```

**非収束の扱い** — 探索空間には return mapping に実根が無いパラメータ値が普通に含まれる。そのため評価中の非収束は例外ではなく**有限の大ペナルティ** (1e30) としてスコアされ、最適化は続行する (`inf` にしないのは L-BFGS-B の有限差分勾配を汚染しないため)。`FitResult.n_failed` に失敗評価数、`failure_message` / `failure_params` に最初の失敗の再現情報が入る。**全評価が失敗**した場合のみ raise し、メッセージが 2 ケースを区別する: 初期値自体が解けない「fitting cannot start」(初期値を見直す; データ間引きで Δε が広がると起きやすい) と、初期値は解けたが探索が全滅した「no point the optimiser tried was」(bounds を狭める)。

**相関パラメータ — `ParameterSpace`** — `Y < B` のような順序制約は、比を最適化変数にして下限 1.0 を課すと不等式制約なしで表現できる:

```python
from manforge.fitting import ParameterSpace

space = ParameterSpace(
    free={"Y": (200.0, (50.0, 600.0)),
          "B_ratio": (1.5, (1.0, 5.0))},        # B/Y >= 1 ⇔ Y <= B
    fixed={"E": 210_000.0, "nu": 0.3},
    derived={"B": lambda p: p["Y"] * p["B_ratio"]},
)
result = fit_params(model, driver_factory, exp_data, space, max_iter=200)
result.params["B"]   # 復元済み
```

**複数カーブ同時フィット — `FitTarget` / `Observation`** — 2 階層になっているのは計算コストが違うため。`FitTarget` = **1 回の解析** (載荷履歴 1 本 + driver 1 つ、目的関数 1 評価あたり driver run 1 回)。`Observation` = **1 本の実験カーブ** (同一ターゲット上の複数 observation は解析結果を共有、追加コストほぼゼロ):

```python
from manforge.fitting import FitTarget, Observation
from manforge.simulation.driver import StressDriver

targets = [
    FitTarget(                       # ひずみ制御: σ11 と ep を同じ解析から比較
        load=FieldHistory(FieldType.STRAIN, "Strain", strain_hist),
        observations=[
            Observation(values=sigma11_exp, component=0),
            Observation(values=ep_exp, field="ep", weight=1e8),  # スケール差は weight で調整
        ],
    ),
    FitTarget(                       # 応力制御実験: このターゲットだけ driver が違う
        load=FieldHistory(FieldType.STRESS, "Stress", stress_hist),
        observations=[Observation(values=strain_exp, field="Strain", component=0)],
        driver_factory=lambda i: StressDriver(i),
    ),
]
result = fit_params(model, lambda i: StrainDriver(i), targets, fit_config,
                    fixed_params={"E": 210_000.0, "nu": 0.3})
```

各 observation の残差二乗和は**データ点数で割られる** (長いカーブが支配するのを防ぐ)。デフォルトの比較は点対点 MSE で、**実験とシミュレーションのサンプリングが一致している必要がある**。

**サンプリングが一致しない実験データ — `metric="pcm"`** — 実験が細かく計測され、シミュレーションを粗い増分で回す場合は、点対点でなく曲線間面積で比較する Partial Curve Mapping を使う:

```python
Observation(values=stress_exp, x_values=strain_exp, metric="pcm", component=0)
```

点数・x 範囲が違ってよく、ヒステリシスループのような多価カーブも扱える。注意: 粗い増分は降伏コーナーを弦で切るため**複数パラメータの同時フィットにバイアス**が乗る (実測: 30 増分で H に 12% 誤差、400 増分で消失)。単一パラメータの掃引は不偏。詳細は `fitting/metrics.py` の docstring。

---

## 5. モデルを作る

### 5.1 Explicit 型 (J2 型) — closed-form 更新

状態変数を `Explicit` として宣言し、`update_state` に閉形式更新を書く。NR はスカラー Δλ のみ解く:

```python
from manforge.core import Explicit, SCALAR
from manforge.core.material import MaterialModel
from manforge.core.dimension import SOLID_3D, StressDimension

class MyModel(MaterialModel):
    param_names = ["E", "nu", "sigma_y0", "H"]
    ep = Explicit(shape=SCALAR, doc="equivalent plastic strain")

    def __init__(self, dimension: StressDimension = SOLID_3D, *, E, nu, sigma_y0, H):
        super().__init__(dimension=dimension)
        self.E = E; self.nu = nu; self.sigma_y0 = sigma_y0; self.H = H

    def yield_function(self, state):
        return self.vonmises(state["stress"]) - (self.sigma_y0 + self.H * state["ep"])

    def update_state(self, dlambda, state_new, state_n, *, stress_trial=None, strain_inc=None):
        return [self.ep(state_n["ep"] + dlambda)]
```

`elastic_stiffness` は `self.E`/`self.nu` から自動生成 (状態依存の場合のみ override)。物理ロジックは次元非依存に書けるので、親クラスに 1 度実装し、次元特化クラスはデフォルト `dimension` の指定のみを担う — 参照実装: `src/manforge/models/j2_isotropic.py`。

### 5.2 Implicit 型 (Ohno-Wang 型) — 後方 Euler 残差

閉形式で解けない発展則は状態変数を `Implicit` として宣言し、`state_residual` に残差を書く。`stress = Implicit(shape=NTENS)` を宣言すると σ も NR 未知数になる (ベクトル NR):

```python
from manforge.core import Implicit, NTENS, SCALAR

class MyImplicitModel(MaterialModel):
    param_names = ["E", "nu", "sigma_y0", "C_k", "gamma"]
    stress = Implicit(shape=NTENS, doc="Cauchy stress (NR unknown)")
    alpha  = Implicit(shape=NTENS, doc="backstress tensor")
    ep     = Implicit(shape=SCALAR, doc="equivalent plastic strain")

    def yield_function(self, state):
        return self.vonmises(state["stress"] - state["alpha"]) - self.sigma_y0

    def state_residual(self, state_new, dlambda, state_n, *, stress_trial, strain_inc=None):
        R_stress = self.default_stress_residual(state_new, dlambda, stress_trial)
        R_alpha = ...   # 後方 Euler 残差, shape (ntens,)
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        return [self.stress(R_stress), self.alpha(R_alpha), self.ep(R_ep)]
```

参照実装: `src/manforge/models/ow_kinematic.py`。Explicit と Implicit の混在 (partial-implicit) も可。粘塑性などで Δλ 行の残差自体を差し替えたい場合は `self.dlambda(R_dl)` を返す (省略時は `yield_function = 0`)。

### 5.3 flow のオーバーライド — 非関連流れ・手導出

デフォルトの流れ方向は関連流れ (`∂f/∂σ` を autodiff)。塑性ポテンシャル g ≠ f や手導出式を使う場合は `flow` を override する (**`default_stress_residual` は触らない** — 残差の代数は不変)。返り値は計算した規約を `strain_flow` / `stress_flow` で名指しする:

```python
def flow(self, state):                       # 教科書形 (physical shear) で導出した場合
    s_xi = self.dev(state["stress"]) - state["alpha"]
    return self.stress_flow(1.5 * s_xi / self.vonmises_norm(s_xi))

def flow(self, state):                       # engineering shear で導出済みの場合
    return self.strain_flow(my_n_voigt)
```

生の ndarray を返すと境界で `TypeError`。モデル内で ×2/÷2 の変換算術を書かないことが設計意図 (§ 3.3)。

### 5.4 弾塑性分岐のカスタム — elastic_check / elastic_update_state

既定の弾性判定は `yield_function(state_trial) <= 0`。**弾性域が降伏関数の内部と一致しないモデル** (subloading-surface 系が典型: 分岐が 2 条件の論理和になり、1 つの滑らかな関数のゼロ等値面で書けない) のみ `elastic_check(state_n, *, stress_trial, strain_inc=None) -> bool` を override する。弾性ステップでも動く状態量 (応力に追従する正規降伏比など) は `elastic_update_state` に書く。古典的塑性では両方ともデフォルトで正しい。詳細な設計理由は `core/material/base.py` の docstring。

### 5.5 user_defined フック — 閉形式ソルバ

閉形式の return mapping が存在する場合は `user_defined_return_mapping(self, stress_trial, C, state_n)` を実装すると `PythonIntegrator` が自動使用する (`None` を返すと汎用 NR にフォールバック)。consistent tangent の閉形式は `user_defined_tangent`。`J2Isotropic3D` が両フックの参照実装。

### 5.6 検証ワークフロー

新しいモデルを実装するときの推奨手順:

1. **数値収束の確認** — `PythonNumericalIntegrator` で残差が落ちることを確認。`result.residual_history` で NR 収束履歴 (二次収束なら桁が倍々で落ちる) を見られる
2. **ヤコビアンのブロック検証** — 手導出式がある場合、AD 値と要素ごとに照合:

```python
from manforge.verification import JacobianChecker

result = PythonIntegrator(model).stress_update(deps, stress_n, state_n)
jac = JacobianChecker(model).compute(result, state_n)
jac.part["alpha"]["stress"]    # ∂R_alpha/∂σ — 手導出と npt.assert_allclose で比較
jac.part["dlambda"]["dlambda"]
```

3. **FD tangent チェック** — consistent tangent を中心差分と比較:

```python
from manforge.verification import TangentChecker
r = TangentChecker(PythonIntegrator(model)).check(stress_n, state_n, strain_inc)
print(r.passed, r.max_rel_err)
```

4. **実装同士のクロスチェック** — 閉形式 vs NR、または Python vs Fortran を全履歴で比較:

```python
from manforge.verification import CrosscheckStrainDriver, generate_strain_history

cc = CrosscheckStrainDriver(PythonNumericalIntegrator(model), PythonAnalyticalIntegrator(model))
cr = cc.run(FieldHistory(FieldType.STRAIN, "eps", generate_strain_history(model)))
print(cr.passed, cr.max_stress_rel_err)
```

通しの実例: `examples/example_custom_model_workflow.py`

### 5.7 Fortran UMAT クロス検証

**Build → Load → Integrate → Crosscheck**:

```bash
uv run manforge build fortran/abaqus_stubs.f90 fortran/j2_isotropic_3d.f90 --name j2_isotropic_3d
uv run manforge list   # 確認
```

```python
from manforge.simulation import FortranIntegrator
from manforge.verification import FortranModule, CrosscheckStrainDriver

fortran = FortranModule("j2_isotropic_3d")
fc_int = FortranIntegrator.from_model(fortran, "j2_isotropic_3d", model)

cc = CrosscheckStrainDriver(PythonIntegrator(model), fc_int)
cr = cc.run(load)
assert cr.passed, f"max stress rel err: {cr.max_stress_rel_err:.2e}"
```

ビルド手順の詳細は `fortran/README.md`、通しの実例は `examples/example_fortran_crosscheck.py`。

---

## 6. サービスに組み込む

WebAPI やジョブワーカーへの組み込み用に、`manforge.service` が **Web 依存ゼロの契約層**を提供する。入出力は plain な dict / ndarray / dataclass のみで、pydantic / HTTP 変換はデプロイ側の仕事。

```python
import numpy as np
from manforge.service import ModelRegistry, default_registry, run_simulation

# レジストリ: 名前 → モデルクラスの明示対応 (シングルトン無し —
# デプロイごとに作り、公開するモデルだけを登録して監査可能にする)
reg = default_registry()               # 参照 1D モデル入り
reg.names()                            # ['af_kinematic', 'j2_isotropic', 'ow_kinematic']

# describe() = GET /models 相当。param_meta (単位・説明・デフォルト・範囲)
# から WebUI の入力フォームを自動生成できる
d = reg.describe("j2_isotropic")
d.param_meta["E"]                      # ParamMeta(doc="Young's modulus", unit="MPa", default=210000.0, ...)

# 実行: 生のリストを受け取り plain な配列を返す
out = run_simulation(
    reg.get("j2_isotropic"),
    params={"E": 206e3, "nu": 0.3, "sigma_y0": 250.0, "H": 1000.0},
    load=np.linspace(0, 5e-3, 1000),   # ひずみ履歴 (control="stress" で応力制御)
    include_fields=["ep"],             # 状態量は選択制 (デフォルト応力のみ)
    progress=lambda done, total: ...,  # ジョブ進捗の報告に
)
out.stress, out.strain, out.fields["ep"]
```

**非収束は値で返る** — パラメータ探索中のユーザーは解の無い値を普通に入力するため、`run_simulation` は途中で解けなくなっても raise せず、そこまでの部分履歴 + `converged=False` / `failed_step` を返す (API 層は 200 で返し、UI が「ステップ k で非収束」と部分曲線を描ける)。一方、パラメータ名の過不足や未知の `include_fields` は**呼び出し側のバグ**なので `ValueError` (API 層で 400)。

**非公開モデルの登録規約** — 非公開モデルパッケージは `MODELS: dict[str, type[MaterialModel]]` を公開し、デプロイ側がそれを import して自分のレジストリに登録する:

```python
from manforge_internal import MODELS      # 非公開パッケージ側
for name, cls in MODELS.items():
    reg.register(name, cls)
```

モデルの `param_meta` はモデルクラス自身に宣言する (`ParamMeta`) — メタ情報がリポジトリ間で乖離しないため。実行例: `examples/example_service.py`

---

## 7. Development

```bash
make test        # 高速テスト (unit + integration、slow/fortran 除く)
make test-all    # 全スイート
```

その他のターゲット (e2e / slow / benchmarks / docker) は `Makefile` を参照。テストレイアウトは `tests/{unit,integration}` が `src/manforge` を 1:1 ミラー、`tests/benchmarks/<model>/` が数値同値ハーネス。

**Examples**

| スクリプト | 内容 |
|-----------|------|
| `examples/example_j2_uniaxial.py` | 繰返し単軸 + tangent 検証 + ヒステリシスのプロット |
| `examples/example_fit_j2.py` | パラメータフィッティング (基本 / ParameterSpace / 複数ターゲット / PCM) |
| `examples/example_custom_model_workflow.py` | 自作モデルの定義 → 検証の通しワークフロー |
| `examples/example_fortran_crosscheck.py` | Fortran UMAT のビルド → 実行 → クロス検証 (要 gfortran) |
| `examples/example_service.py` | service 層: registry → describe → run_simulation (非収束込み) |

---

## License

[MIT License](LICENSE)
