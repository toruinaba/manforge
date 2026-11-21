# モデルを作る

このページは、モデルに書けるフックを 1 つずつ説明する。モデルを置く場所、検証の宣言、UMAT までの手順は [新しいモデルを足す](adding-a-model.md) にある。

## Explicit 型 (J2 型) — closed-form 更新

1 ステップの入力は `Increment` にまとめて全フックへ渡される (`inc`):

| | 中身 |
|---|---|
| `inc.state_n` | ステップ開始時の状態。`inc.state_n["stress"]` は σ_n |
| `inc.strain_inc` | ひずみ増分 Δε (engineering shear) |
| `inc.trial` | 弾性試行状態。`inc.trial["stress"]` は σ_trial |

UMAT の構造と同じ並びになっている — 冒頭で試行状態を計算し、そこから分岐と補正を行う。フックの引数リストが違うモデルは、クラス定義時に `TypeError` になる (位置引数で呼ばれるため、黙って引数がずれるのを防ぐ)。

状態変数を `Explicit` として宣言し、`update_state` に閉形式更新を書く。explicit 状態は NR の未知数に入らず、NR は `[σ, Δλ]` を解く:

```python
from returnmap.core import MaterialModel, Explicit, SCALAR

class MyModel(MaterialModel):
    param_names = ["E", "nu", "sigma_y0", "H"]
    ep = Explicit(shape=SCALAR, doc="equivalent plastic strain")

    def yield_function(self, state):
        return self.vonmises(state["stress"]) - (self.sigma_y0 + self.H * state["ep"])

    def update_state(self, new, dlambda, inc):
        return [self.ep(inc.state_n["ep"] + dlambda)]

model = MyModel(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)   # 3D
```

**`__init__` は書かない。** `param_names` がパラメータの宣言で、基底クラスがその名前のキーワード引数を受け取り、同名の属性 (`self.E` など) に入れる。並び順は UMAT の `PROPS` の順。全パラメータが必須で、既定値は持てない。

- 定数はクラス属性 (`ae = 0.5`)、パラメータから導く値は `@property` に書く。
- パラメータではない設定 (数値的な正則化の係数など) が要るときだけ `__init__` を override し、残りを `super().__init__(**params)` に渡す — 例: `J2IsotropicDamageUnilateral` の `c`。

**次元は 2 通りで決まる。** クラス本体に `dimension` を書かなければ、構築時の `dimension=` で選ぶ (省略時は 3D)。クラス本体に書くと、そのクラスと子クラスはその次元に固定され、別の次元を渡すと `TypeError` になる。閉形式のリターンマッピングや手導出ブロックのように、代数が特定の次元でしか成り立たないクラスは固定する:

```python
from returnmap.core import PLANE_STRAIN, PLANE_STRESS

class MyModelPS(MyModel):
    dimension = PLANE_STRESS          # 固定

MyModel(dimension=PLANE_STRAIN, E=...)       # 可変: 任意の次元
MyModelPS(E=...)                             # 固定
MyModelPS(dimension=PLANE_STRAIN, E=...)     # TypeError
```

`elastic_stiffness(state)` の既定は `self.E`/`self.nu` から作る等方剛性 `isotropic_stiffness()`。剛性が状態で変わるモデルだけ override し、等方剛性を掛けて書く:

```python
def elastic_stiffness(self, state):                    # 損傷
    return (1.0 - state["D"]) * self.isotropic_stiffness()
```

物理ロジックは次元非依存に書けるので、親クラスに 1 度実装し、次元特化クラスは `dimension` の固定のみを担う — 参照実装: `src/returnmap/models/j2_isotropic/model.py`。

## Implicit 型 (Ohno-Wang 型) — 後方 Euler 残差

閉形式で解けない発展則は状態変数を `Implicit` として宣言し、`state_residual` に残差を書く。implicit 状態は σ、Δλ と並んで NR の未知数になる (`x = [σ, Δλ, implicit 状態]`)。σ は常に未知数で、`stress = Implicit(shape=NTENS)` を宣言すると σ 行の残差も自分で書く (宣言しなければ、関連流れの既定の残差が使われる):

```python
from returnmap.core import Implicit, NTENS, SCALAR

class MyImplicitModel(MaterialModel):
    param_names = ["E", "nu", "sigma_y0", "C_k", "gamma"]
    stress = Implicit(shape=NTENS, doc="Cauchy stress (NR unknown)")
    alpha  = Implicit(shape=NTENS, doc="backstress tensor")
    ep     = Implicit(shape=SCALAR, doc="equivalent plastic strain")

    def yield_function(self, state):
        return self.vonmises(state["stress"] - state["alpha"]) - self.sigma_y0

    def state_residual(self, new, dlambda, inc):
        R_stress = self.default_stress_residual(new, dlambda, inc.trial["stress"])
        R_alpha = ...   # 後方 Euler 残差, shape (ntens,)
        R_ep = new["ep"] - inc.state_n["ep"] - dlambda
        return [self.stress(R_stress), self.alpha(R_alpha), self.ep(R_ep)]
```

参照実装: `src/returnmap/models/ow_kinematic/model.py`。Explicit と Implicit の混在 (partial-implicit) も可。粘塑性などで Δλ 行の残差自体を差し替えたい場合は `self.dlambda(R_dl)` を返す (省略時は `yield_function = 0`)。

`inc.trial` / `inc.strain_inc` は**どの残差行にどの係数で入れてもよい** — consistent tangent は dR/dΔε を両者を通して autodiff するので、`R_σ = σ − σ_trial + …` の形を仮定しない。同じ根を持つ残差なら行のスケールを変えても接線は同一 (例: 損傷の `(1−D_n)/(1−D_{n+1})·σ − σ_trial + …`)。

## flow のオーバーライド — 非関連流れ・手導出

デフォルトの流れ方向は関連流れ (`∂f/∂σ` を autodiff)。塑性ポテンシャル g ≠ f や手導出式を使う場合は `flow` を override する (**`default_stress_residual` は触らない** — 残差の代数は不変)。返り値は計算した規約を `strain_flow` / `stress_flow` で名指しする:

```python
def flow(self, state):                       # 教科書形 (physical shear) で導出した場合
    s_xi = self.dev(state["stress"]) - state["alpha"]
    return self.stress_flow(1.5 * s_xi / self.vonmises_norm(s_xi))

def flow(self, state):                       # engineering shear で導出済みの場合
    return self.strain_flow(my_n_voigt)
```

生の ndarray を返すと境界で `TypeError`。モデル内で ×2/÷2 の変換算術を書かないことが設計意図 ([概念](../concepts.md) の「Voigt 規約と FlowVector」)。

## 折れと切り替えを滑らかに書く — smooth 関数

モデルの式は自動微分され、NR で解かれる。絶対値、ノルム、`max`、条件分岐のような折れや段差がそのまま入っていると、その点で導関数が定まらず (`sqrt(0)` の微分は無限大になる)、NR がそこで止まる。`returnmap.utils` に、折れの無い近似がある。

| 関数 | 置き換えるもの | 式 |
|---|---|---|
| `smooth_sqrt(x)` | `sqrt(x)` | √(x + ε²) |
| `smooth_abs(x)` | `abs(x)` | √(x² + ε²) |
| `smooth_norm(v)` | ベクトルのノルム | √(v·v + ε²) |
| `smooth_direction(v)` | `v / ‖v‖` | v / `smooth_norm(v)` |
| `smooth_macaulay(x)` | `max(x, 0)` | (x + `smooth_abs(x)`) / 2 |
| `smooth_max(a, b)` / `smooth_min(a, b)` | `max` / `min` | — |
| `smooth_heaviside(x, beta=500)` | `x > 0` の切り替え | ½ (1 + tanh(βx/2)) |

```python
from returnmap.utils import smooth_heaviside, smooth_macaulay

dp_D = smooth_macaulay(ep - self.p_D) - smooth_macaulay(ep_n - self.p_D)   # しきい値を超えた分
C_k = self.C_1 - (self.C_1 - self.C_2) * smooth_heaviside(theta_max - (self.B - self.Y))
```

- **`eps` は 0 での割り算を防ぐだけの値** (既定 1e-12) で、応答には現れない。
- **`smooth_heaviside` の幅は、引数の単位で決まる。** 既定の `beta=500` では、0 から 1 への遷移 (1 % → 99 %) が引数の値で ±0.01 の幅になる。引数が応力 (MPa) なら ±0.01 MPa で、実質的に段差である。ひずみのように小さな数で測る量を渡すときは、幅がその量の変化に対して意味のある大きさになるよう、`beta` を決めるか引数に倍率を掛ける。
- **手書きの経路と UMAT には、同じ関数を同じ係数で書く。** 係数が違うと、遷移の帯の中で 2 つの実装が別の方程式を解くことになる。

### 切り替えで有限の量を出し入れしない

切り替えに掛ける量が、切り替わる点で 0 にならない形は避ける。

```python
# 切り替わる点で段差になる: gate が 0 → 1 に動くと、R が dR だけ跳ぶ
R_new = R_n + smooth_heaviside(g) * dR

# 切り替わる点でつながる: 掛ける量が g とともに 0 になる
r_new = r_n + self.h * smooth_macaulay(g)
```

上の形では、解が切り替えの縁に来るステップで残差がほぼ不連続になる。Newton 反復が切り替えの両側を往復して収束しないことがある。収束しても、そのステップでは増分をわずかに変えるだけで解けたり解けなかったりして、接線の差分検査が成り立たないことがある。平滑化の幅を広げても段差の大きさは変わらないので、それだけでは直らない。

式の定義としてこの形が避けられないモデルもある (Yoshida-Uemori の限界曲面の膨張は、停留の判定でこの形になる)。その場合は次の 2 つを知っておく。

- 切り替わるステップでは、解き直しや細かい刻みが要ることがある。
- 切り替えを定数として微分した手導出は、そのステップでだけ近似になる。許容値を緩めず、`approx_when` で宣言する ([検証ワークフロー](verification.md))。

主値の符号で分ける量 (引張と圧縮の分解) は、ここの関数ではなく主値分解の関数を使う (下の「非線形な弾性」)。

## 弾塑性分岐のカスタム — elastic_check / elastic_update_state

既定の弾性判定は `yield_function(state_trial) <= 0`。**弾性域が降伏関数の内部と一致しないモデル** (subloading-surface 系が典型: 分岐が 2 条件の論理和になり、1 つの滑らかな関数のゼロ等値面で書けない) のみ `elastic_check(inc) -> bool` を override する。弾性ステップでも動く状態量 (応力に追従する正規降伏比など) は `elastic_update_state(inc)` に書く。古典的塑性では両方ともデフォルトで正しい。詳細な設計理由は `core/material/base.py` の docstring。

```python
def elastic_check(self, inc):
    R_hat = self.normal_yield_ratio(inc.state_n, inc.trial["stress"])
    return bool(R_hat <= self.R_e) or bool(R_hat <= inc.state_n["R"])

def elastic_update_state(self, inc):                   # 弾性と判定された後に適用
    return [self.R(self.normal_yield_ratio(inc.state_n, inc.trial["stress"]))]
```

追従変数を予測子 (後述の「非線形な弾性」) ではなくここに書くのは、試行状態に載せると `yield_function(試行状態)` が恒等的に 0 になり、既定の分岐判定が載荷と除荷を区別できなくなるため。弾性ステップの更新が応力に戻る発展則 (弾性エネルギー駆動の損傷、クリープ) は扱えない。

## 非線形な弾性 — elastic_predictor と主値分解

既定の弾性予測子は線形 `σ_trial = σ_n + C(state_n)·Δε`。片側損傷のように**弾性則が区分線形** (主値の符号で剛性が変わる) なモデルや、σ 以外の状態量 (例: Implicit の有効応力) が弾性的に動くモデルは `elastic_predictor` を override して試行状態をモデル側で定義する:

```python
def elastic_predictor(self, state_n, *, strain_inc):
    s_eff = state_n["stress_eff"] + self.undamaged_stiffness() @ strain_inc  # 有効空間
    sigma = self.nominal_from_effective(s_eff, state_n["D"])      # 非線形写像
    return [self.stress.update(sigma), self.stress_eff.update(s_eff)]
```

- stress は必須。値は `self.<field>.update(value)` で包む (Implicit 場を `self.<field>(value)` と呼ぶと残差扱いになるため)。
- 返した試行状態が `inc.trial` になる。フレームワークはこれを弾性ステップの解・弾性判定 (既定は試行状態全体で `yield_function`)・塑性 NR の初期値と接線のすべてに使う。微分可能 (autograd で追跡可能) に書くこと。
- `elastic_update_state` (前述の「弾塑性分岐のカスタム」) とは別物: 予測子は毎ステップの定数で分岐判定に使われる。`elastic_update_state` は弾性と判定された後の更新。
- override しなければ線形予測子のままで、弾性接線は `elastic_stiffness(state_n)`。

**弾性接線の手書きスロット — `elastic_tangent(inc)`** — 予測子を override すると、弾性接線は予測子の自動微分で得られる。動かすだけならそれで足りるが、UMAT には式そのものが要る。`elastic_tangent(inc)` に `dσ_trial/dΔε` を書くと、`PythonAnalyticalIntegrator` はそれを使い (未実装なら `NotImplementedError`)、`PythonNumericalIntegrator` は常に予測子を自動微分する。両者を突き合わせれば手導出を検証できる。

主値に依存する量は `returnmap.utils` の主値分解の関数を使う (`anp.linalg.eigh` の微分は主値が重なる単軸・平面応力・純せん断で NaN になるため):

```python
from returnmap.utils import positive_part, negative_part, positive_part_sq
sig_pos = positive_part(sigma, self.dimension)        # σ₊ = Σ⟨σ_i⟩ n_i⊗n_i, 接線 P₊ も厳密
energy  = positive_part_sq(sigma, self.dimension)     # Σ⟨σ_i⟩₊² (Hessian = 2P₊)
```

主値 0 では折れ点を平滑化せず、規約 H(0) = c (`DEFAULT_HEAVISIDE_AT_ZERO = 1/2`、**暫定** — UMAT 側の規約に合わせて確定) で扱う。この規約を使う関数 (`positive_part` / `negative_part` とその 2 乗版、`macaulay_sq` / `neg_macaulay_sq`) は `c=` で上書きできる。設計理由は `utils/spectral.py` の docstring、参照実装は `src/returnmap/models/j2_damage_unilateral.py`。

## user_defined フック — 閉形式ソルバ

閉形式の return mapping が存在する場合は `user_defined_return_mapping(self, inc)` を実装すると `PythonIntegrator` が自動使用する (`None` を返すと汎用 NR にフォールバック)。consistent tangent の閉形式は `user_defined_tangent(self, new, dlambda, inc)`。`inc.trial` には予測子が動かした全フィールドが入っているので、予測子を override したモデルでも手書きソルバを書ける。`State` は読み取り専用なので、作業用の dict は `dict(inc.state_n)` で作る。戻り値の `ReturnMappingResult` は全フィールドが必須で、`converged` も必ず書く (閉形式解は `n_iterations=1`、`residual_history=[f_trial, 0.0]`、`converged=True`。手書きの NR は、許容値で止まったのか反復上限で止まったのかを `converged` で伝える)。`J2Isotropic3D` が両フックの参照実装。
