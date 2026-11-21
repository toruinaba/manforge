# Chaboche モデル (背応力 3 つ)

背応力を 3 つ重ね合わせた Chaboche モデルである。1 つの背応力は 1 つの速さでしか飽和しないので、実測のヒステリシスループには足りないことが多い。速く飽和する背応力で降伏直後の丸みを、遅い背応力でその先の硬化を表す。等方硬化も 3 項を重ねる。

| クラス | 次元 | リターンマッピング |
|---|---|---|
| `ChabocheKinematicMulti` | 構築時に `dimension=` で選ぶ | 自動微分 |
| `ChabocheKinematicMulti3D` | 3D ソリッド | 自動微分 |
| `ChabocheKinematicMultiPS` | 平面応力 | 自動微分 |
| `ChabocheKinematicMulti1D` | 単軸応力 | 自動微分 |

手書きの解法と UMAT は持たない。

## 式

降伏関数は、3 つの背応力の和だけ中心を移し、3 つの等方硬化の和だけ半径を変えた Mises 応力である。

$$
f = \sqrt{\tfrac{3}{2}\,\boldsymbol\xi:\boldsymbol\xi} - \Bigl(\sigma_{y0} + \sum_{i=1}^{3} R_i\Bigr),
\qquad
\boldsymbol\xi = \boldsymbol{s} - \sum_{i=1}^{3}\boldsymbol\alpha_i
$$

流れ則は関連流れで、$\boldsymbol{n} = \tfrac{3}{2}\,\boldsymbol\xi/\sqrt{\tfrac{3}{2}\,\boldsymbol\xi:\boldsymbol\xi}$、$\mathrm{d}p = \mathrm{d}\lambda$ である。各項は、背応力 1 つのモデルと同じ発展則に、それぞれの係数で従う。

$$
\mathrm{d}\boldsymbol\alpha_i = \tfrac{2}{3}\,C_i\,\boldsymbol{n}\,\mathrm{d}p - \gamma_i\,\boldsymbol\alpha_i\,\mathrm{d}p,
\qquad
\mathrm{d}R_i = b_i\,(Q_{\infty i} - R_i)\,\mathrm{d}p
\qquad (i = 1, 2, 3)
$$

後退 Euler の更新は、項ごとに閉じた形になる。

$$
\boldsymbol\alpha_{i,n+1} = \frac{\boldsymbol\alpha_{i,n} + \tfrac{2}{3}\,C_i\,\Delta\lambda\,\boldsymbol{n}_{n+1}}{1 + \gamma_i\,\Delta\lambda},
\qquad
R_{i,n+1} = \frac{R_{i,n} + b_i\,Q_{\infty i}\,\Delta\lambda}{1 + b_i\,\Delta\lambda}
$$

状態変数はすべて Explicit で、NR の未知数は $[\boldsymbol\sigma, \Delta\lambda]$ である。

## 性質

- **飽和**: 単調載荷の単軸応力は $\sigma_{y0} + \sum C_i/\gamma_i + \sum Q_{\infty i}$ に漸近する (目安の値では 250 + 300 + 70 = 620 MPa)。
- **項ごとの速さ**: 背応力 $i$ の飽和の速さは $1/\gamma_i$ が目安で、塑性ひずみがその 3 倍進むと飽和値の 95 % に達する。目安の値は $\gamma_i$ を 800、100、10 と 1 桁ずつ変えてあり、どれも 100 MPa に飽和する。
- **項を切る**: $C_i = 0$ でその背応力が、$Q_{\infty i} = 0$ でその等方硬化が働かなくなる。
- **1 項だけ残す**: 背応力と等方硬化を 1 つずつ残すと、背応力 1 つのモデル (`ChabocheKinematic`) とビット単位で一致する。
- **$\gamma_i = 0$**: その背応力は線形移動硬化になり、飽和しない。

## 状態変数

| 名前 | 種別 | 意味 |
|---|---|---|
| `stress` | — | Cauchy 応力 |
| `alpha1` | Explicit | 背応力 $\boldsymbol\alpha_1$ (偏差) |
| `alpha2` | Explicit | 背応力 $\boldsymbol\alpha_2$ |
| `alpha3` | Explicit | 背応力 $\boldsymbol\alpha_3$ |
| `alpha` | Explicit | 背応力の和 |
| `R1` | Explicit | 等方硬化 $R_1$ |
| `R2` | Explicit | 等方硬化 $R_2$ |
| `R3` | Explicit | 等方硬化 $R_3$ |
| `R` | Explicit | 等方硬化の和 |
| `ep` | Explicit | 相当塑性ひずみ $p$ |

`alpha` と `R` は和をそのまま持つ状態変数で、結果を読むときに足し合わせなくて済むように置いてある。降伏関数は各項から和を作るので、この 2 つは計算には使われない。背応力の持ち方は Armstrong-Frederick と同じである (単軸は偏差成分 $\alpha_{11}$ で、応力の単位で見た値はその $\tfrac{3}{2}$ 倍)。

## パラメータ

| 名前 | 意味 | 単位 | 目安の値 |
|---|---|---|---|
| `E` | ヤング率 | MPa | 210000 |
| `nu` | ポアソン比 | — | 0.3 |
| `sigma_y0` | 初期降伏応力 | MPa | 250 |
| `C1` | 背応力 1 の移動硬化係数 | MPa | 80000 |
| `gamma1` | 背応力 1 の動的回復の係数 | — | 800 |
| `C2` | 背応力 2 の移動硬化係数 | MPa | 10000 |
| `gamma2` | 背応力 2 の動的回復の係数 | — | 100 |
| `C3` | 背応力 3 の移動硬化係数 | MPa | 1000 |
| `gamma3` | 背応力 3 の動的回復の係数 | — | 10 |
| `Qinf1` | 等方硬化 1 の飽和値 | MPa | 40 |
| `b1` | 等方硬化 1 が飽和する速さ | — | 40 |
| `Qinf2` | 等方硬化 2 の飽和値 | MPa | 30 |
| `b2` | 等方硬化 2 が飽和する速さ | — | 5 |
| `Qinf3` | 等方硬化 3 の飽和値 | MPa | 0 |
| `b3` | 等方硬化 3 が飽和する速さ | — | 1 |

項の数は 3 つに固定している。少なくてよいときは、使わない項の $C_i$ と $Q_{\infty i}$ を 0 にする。

## 検証

- 共通の検査では、自動微分の経路が収束することと、整合接線が中心差分と合うことを確かめている。
- ファミリのテストは、飽和応力、背応力がそれぞれの速さで飽和すること、1 項だけ残すと背応力 1 つのモデルとビット単位で一致することを確かめている。
- 引張とせん断が同じ曲線になること、単軸応力の応答が 1D・3D・平面応力で一致することは、全モデルに共通のテストで確かめている。

## 文献

J.-L. Chaboche (1986), *Time-independent constitutive theories for cyclic plasticity*, International Journal of Plasticity.
