# Yoshida-Uemori モデル

降伏曲面と限界曲面の 2 つの曲面を持つ移動硬化モデルである。大ひずみの繰返し塑性で現れる 3 つの現象を表す: 反転直後の早い再降伏 (Bauschinger 効果)、反転後に加工硬化が一時的に止まる現象 (加工硬化の停留)、塑性ひずみによる除荷剛性の低下。板材成形のスプリングバック解析で使われる。

| クラス | 次元 | リターンマッピング | UMAT |
|---|---|---|---|
| `YUKinematic` | 構築時に `dimension=` で選ぶ | 自動微分 | — |
| `YUKinematic3D` | 3D ソリッド | 手書きの NR と手導出ヤコビアン | `fortran/yu_kinematic_3d.f90` |
| `YUKinematicPS` | 平面応力 | 手書きの NR と手導出ヤコビアン | `fortran/yu_kinematic_ps.f90` |
| `YUKinematic1D` | 単軸応力 | 自動微分 | — |

## 式

### 2 つの曲面

降伏曲面は半径 $Y$ で大きさを変えず、限界曲面の内側を移動する。限界曲面は中心 $\boldsymbol\beta$、半径 $B + R$ を持ち、移動しながら膨らむ。降伏曲面の中心 (背応力) を $\boldsymbol\alpha$ として、限界曲面の中心から測った相対的な背応力 $\boldsymbol\theta = \boldsymbol\alpha - \boldsymbol\beta$ を状態変数にする。

$$
f = \sqrt{\tfrac{3}{2}\,\boldsymbol\xi:\boldsymbol\xi} - Y,
\qquad
\boldsymbol\xi = \boldsymbol{s} - \boldsymbol\theta - \boldsymbol\beta
$$

流れ則は関連流れで、$\mathrm{d}p = \mathrm{d}\lambda$ である。

### 降伏曲面の移動

$\boldsymbol\theta$ は、2 つの曲面の半径の差 $a = B + R - Y$ まで進める。$\bar\theta = \sqrt{\tfrac{3}{2}\,\boldsymbol\theta:\boldsymbol\theta}$ として、

$$
\mathrm{d}\boldsymbol\theta = C\left[\frac{a}{Y}\,\boldsymbol\xi - \sqrt{\frac{a}{\bar\theta}}\;\boldsymbol\theta\right]\mathrm{d}p
$$

$\bar\theta = a$ で降伏曲面が限界曲面に接し、それ以上は近づかない。係数 $C$ は、$\bar\theta$ の履歴上の最大値 $\theta_\mathrm{max}$ が $B - Y$ に達するまで $C_1$、達した後は $C_2$ である。初期降伏の直後だけ硬化を速くするための切り替えである。

### 限界曲面の移動と膨張

$$
\mathrm{d}\boldsymbol\beta = k\left[\frac{b}{Y}\,\boldsymbol\xi - \boldsymbol\beta\right]\mathrm{d}p,
\qquad
\mathrm{d}R = k\,(R_\mathrm{sat} - R)\,\mathrm{d}p
$$

中心 $\boldsymbol\beta$ の大きさは $b$ に、半径の増分 $R$ は $R_\mathrm{sat}$ に飽和する。$R$ が増えるのは、次の停留曲面の条件が成り立つ間だけである。

### 加工硬化の停留

応力空間に、中心 $\boldsymbol{q}$、半径 $r$ の停留曲面を置く。

$$
g = \sqrt{\tfrac{3}{2}\,(\boldsymbol\beta - \boldsymbol{q}):(\boldsymbol\beta - \boldsymbol{q})} - r
$$

- 限界曲面の中心 $\boldsymbol\beta$ が停留曲面の内側にある間 ($g < 0$) は、$R$ が増えない。反転の直後はこの状態になり、限界曲面が膨らまないので硬化が止まって見える。
- $\boldsymbol\beta$ が停留曲面の上にあって外へ向かうときは、$R$ が増え、停留曲面も $\boldsymbol\beta$ に押されて移動し膨らむ。

$$
\mathrm{d}\boldsymbol{q} = \mu\,(\boldsymbol\beta - \boldsymbol{q}),
\qquad
\mathrm{d}r = h\,\frac{3\,(\boldsymbol\beta - \boldsymbol{q}):\mathrm{d}\boldsymbol\beta}{2r}
$$

$\mu$ は、$\boldsymbol\beta$ が曲面の上にとどまる条件から決まる。$h$ ($0$〜$1$) は、押された分のうち膨張に回る割合で、大きいほど停留が短い。

### 除荷剛性の低下

ヤング率は相当塑性ひずみとともに下がる。

$$
E(p) = E_0 - (E_0 - E_a)\left(1 - e^{-\xi p}\right)
$$

試行応力はステップ開始時の $p$ での剛性で作り、塑性補正の項はステップ終了時の $p$ での剛性を使う。$\xi = 0$ にすると一定になる。

## 離散化

1 ステップは後退 Euler で積分する。

- **未知数**: $[\boldsymbol\sigma, \Delta\lambda, \boldsymbol\theta, \boldsymbol\beta]$。$\boldsymbol\theta$ と $\boldsymbol\beta$ の発展則は残差として書く。
- **$R$、$\boldsymbol{q}$、$r$、$p$、$\theta_\mathrm{max}$**: 閉じた形で更新する (Explicit)。$R$ は $R_{n+1} = (R_n + k R_\mathrm{sat}\Delta\lambda)/(1 + k\,\Delta\lambda)$ である。
- **停留曲面**: 後退 Euler の式から $r_{n+1} = \tfrac{1}{2}\left(r_n + \sqrt{r_n^2 + 6hF/(1 + \mu)}\right)$、$\boldsymbol{q}_{n+1} = \boldsymbol{q}_n + \mu\,(\boldsymbol\beta_{n+1} - \boldsymbol{q}_n)/(1 + \mu)$ となる ($F = (\boldsymbol\beta_{n+1} - \boldsymbol{q}_n):\Delta\boldsymbol\beta$)。$\mu$ は、$\boldsymbol\beta_{n+1}$ が更新後の曲面の上にある条件を内側の Newton 反復で解いて求める。
- **切り替えの平滑化**: 停留の判定と $C_1 \to C_2$ の切り替えは、滑らかにした Heaviside 関数 $\tfrac{1}{2}(1 + \tanh(250\,x))$ で行う ($x$ は応力の単位。遷移の幅は ±0.01 MPa 程度)。停留の判定には、ステップ開始時の曲面に対する $g = \bar{(\boldsymbol\beta_{n+1} - \boldsymbol{q}_n)} - r_n$ を使う。$C$ の切り替えには、ステップ開始時の $\theta_\mathrm{max}$ を使う。

## 性質

- **飽和応力**: 単調載荷の単軸応力は $B + R_\mathrm{sat} + b$ に漸近する (目安の値では 435 + 255 + 66 = 756 MPa)。
- **除荷の傾き**: 除荷直後の単軸の傾きは $E(p)$ である。

どちらも 1D のモデルで確かめた (60 % までの単調載荷、5 % からの除荷)。

## 状態変数

| 名前 | 種別 | 意味 |
|---|---|---|
| `stress` | Implicit | Cauchy 応力 |
| `theta` | Implicit | 降伏曲面の相対背応力 $\boldsymbol\theta$ (偏差) |
| `beta` | Implicit | 限界曲面の中心 $\boldsymbol\beta$ (偏差) |
| `R` | Explicit | 限界曲面の半径の増分 |
| `q` | Explicit | 停留曲面の中心 |
| `r` | Explicit | 停留曲面の半径 |
| `eps_eq` | Explicit | 相当塑性ひずみ $p$ |
| `theta_max` | Explicit | $\bar\theta$ の履歴上の最大値 |

## パラメータ

| 名前 | 意味 | 単位 | 目安の値 |
|---|---|---|---|
| `E` | ヤング率 (初期) $E_0$ | MPa | 206000 |
| `nu` | ポアソン比 | — | 0.3 |
| `Y` | 降伏応力 (降伏曲面の半径) | MPa | 360 |
| `B` | 限界曲面の初期の大きさ ($B > Y$) | MPa | 435 |
| `C_1` | 降伏曲面の移動硬化の速さ ($\theta_\mathrm{max}$ が $B - Y$ に達するまで) | — | 2000 |
| `C_2` | 同上 (達した後) | — | 200 |
| `Rsat` | 限界曲面の等方硬化の飽和値 | MPa | 255 |
| `k` | 限界曲面の硬化の速さ (等方・移動の両方) | — | 26 |
| `b` | 限界曲面の背応力の飽和値 | MPa | 66 |
| `h` | 停留曲面の膨張の割合 ($0$〜$1$) | — | 0.4 |
| `Ea` | 大きな塑性ひずみでのヤング率 $E_a$ | MPa | 159000 |
| `xi` | ヤング率の低下の速さ ($0$ で一定) | — | 61 |

## 次元ごとの違い

- **3D**: 流れ方向を手導出の式 $\tfrac{3}{2}\,\boldsymbol\xi/\bar\xi$ で与える (手導出ヤコビアンと同じ式にするため)。
- **平面応力**: 応力系の量を面内 3 成分で持ち、偏差との縮約は平面応力の計量 $\mathbf{P}$ で行う (`PLANE_STRESS_P`)。降伏関数は 2 次形式 $f = \tfrac{1}{2}\,\boldsymbol\xi^\mathsf{T}\mathbf{P}\,\boldsymbol\xi - \tfrac{1}{3}Y^2$ で書く。このときの $\Delta\lambda$ は上の式の $\Delta\lambda$ の $1/(\tfrac{2}{3}Y)$ 倍になるので、発展則の係数にその分が入っている。解いている式は同じである。
- **単軸**: $\boldsymbol\theta$、$\boldsymbol\beta$ は偏差の 11 成分を持つ。応力の単位で見た値はその $\tfrac{3}{2}$ 倍である。

## 手導出の近似

3D と平面応力の手導出ヤコビアンと手書きの接線は、停留の判定のゲートを定数として微分している。ゲートが 0 か 1 のステップでは厳密で、切り替わるステップ ($R$ が動き始める、または止まるステップ) でだけ自動微分と合わない。応力と状態量は、そのステップでも一致する。

この近似はモデルに宣言してある (`approx_when="stagnation_gate_in_transition"`)。共通の検査は、宣言されたステップを数えたうえで、ほかのステップを厳密に照合する。

## 制約

- **$\mu$ の反復**: 粗い増分で $\mu$ の内側の反復が収束しないことがあり、そのときモデルは `ValueError` を出す (非収束として扱われない)。
- **平滑化の幅は応力の単位に依る**: ゲートの幅は MPa を前提にした値である。

## 検証

- 3D と平面応力は、手書きの解・手導出のブロック (平面応力の 16 個)・手書きの接線を自動微分と照合している。
- UMAT は 2 つとも、標準の載荷履歴 (多軸と、2 % の反転繰返しを含む) の全ステップで Python と一致する (応力・状態・接線とも 1e-15 程度)。
- 検査には、標準の履歴に加えて 2 % の反転繰返しを使う。標準の履歴は降伏ひずみの 5 倍までで、$C_1 \to C_2$ の切り替えと停留が現れないためである。

## 文献

F. Yoshida, T. Uemori (2002), *A model of large-strain cyclic plasticity describing the Bauschinger effect and workhardening stagnation*, International Journal of Plasticity.
