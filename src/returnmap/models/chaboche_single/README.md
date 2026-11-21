# Chaboche モデル (背応力 1 つ)

Armstrong-Frederick の移動硬化に、飽和する等方硬化を足したモデルである。降伏曲面は移動しながら大きさも変える。繰返し載荷で、Bauschinger 効果と繰返し硬化 (または軟化) を同時に表す。

| クラス | 次元 | リターンマッピング |
|---|---|---|
| `ChabocheKinematic` | 構築時に `dimension=` で選ぶ | 自動微分 |
| `ChabocheKinematic3D` | 3D ソリッド | 自動微分 |
| `ChabocheKinematicPS` | 平面応力 | 自動微分 |
| `ChabocheKinematic1D` | 単軸応力 | 自動微分 |

手書きの解法と UMAT は持たない。背応力を複数重ねる版は `ChabocheKinematicMulti` である。

## 式

降伏関数は、背応力 $\boldsymbol\alpha$ (偏差) だけ中心を移し、等方硬化 $R$ だけ半径を変えた Mises 応力である。

$$
f = \sqrt{\tfrac{3}{2}\,\boldsymbol\xi:\boldsymbol\xi} - (\sigma_{y0} + R),
\qquad
\boldsymbol\xi = \boldsymbol{s} - \boldsymbol\alpha
$$

流れ則は関連流れである。

$$
\mathrm{d}\boldsymbol\varepsilon^p = \mathrm{d}\lambda\,\boldsymbol{n},
\qquad
\boldsymbol{n} = \frac{3}{2}\frac{\boldsymbol\xi}{\sqrt{\tfrac{3}{2}\,\boldsymbol\xi:\boldsymbol\xi}},
\qquad
\mathrm{d}p = \mathrm{d}\lambda
$$

背応力は Armstrong-Frederick の発展則に従い、等方硬化は $Q_\infty$ に向かって飽和する。

$$
\mathrm{d}\boldsymbol\alpha = \tfrac{2}{3}\,C\,\boldsymbol{n}\,\mathrm{d}p - \gamma\,\boldsymbol\alpha\,\mathrm{d}p,
\qquad
\mathrm{d}R = b\,(Q_\infty - R)\,\mathrm{d}p
$$

後退 Euler で離散化すると、どちらも閉じた形で解ける。

$$
\boldsymbol\alpha_{n+1} = \frac{\boldsymbol\alpha_n + \tfrac{2}{3}\,C\,\Delta\lambda\,\boldsymbol{n}_{n+1}}{1 + \gamma\,\Delta\lambda},
\qquad
R_{n+1} = \frac{R_n + b\,Q_\infty\,\Delta\lambda}{1 + b\,\Delta\lambda}
$$

このため状態変数はすべて Explicit で、NR の未知数は $[\boldsymbol\sigma, \Delta\lambda]$ である。

## 性質

- **飽和**: 単調載荷の単軸応力は $\sigma_{y0} + C/\gamma + Q_\infty$ に漸近する (目安の値では 250 + 200 + 100 = 550 MPa)。背応力の大きさは $C/\gamma$ に、等方硬化は $Q_\infty$ に飽和する。
- **初期の硬化係数**: 降伏直後の単軸の塑性係数は $C + b\,Q_\infty$ である。
- **$Q_\infty = 0$**: Armstrong-Frederick (`AFKinematic`) と同じモデルになる。計算結果もビット単位で一致する。
- **$Q_\infty < 0$**: 繰返し軟化を表す。降伏曲面が縮む。
- **$\gamma = 0$**: 背応力が Prager の線形移動硬化になり、飽和しない。
- **引張とせん断**: 相当応力を累積塑性ひずみに対して描くと、単軸引張と単純せん断 ($\sqrt{3}\,\tau$) が同じ曲線になる。

## 状態変数

| 名前 | 種別 | 意味 |
|---|---|---|
| `stress` | — | Cauchy 応力 |
| `alpha` | Explicit | 背応力 $\boldsymbol\alpha$ (偏差) |
| `R` | Explicit | 等方硬化 $R$ (降伏曲面の半径の増分) |
| `ep` | Explicit | 相当塑性ひずみ $p$ |

`alpha` の持ち方は Armstrong-Frederick と同じである。3D は 6 成分の偏差テンソル、平面応力は $[\alpha_{11}, \alpha_{22}, \alpha_{12}]$ ($\alpha_{33} = -(\alpha_{11} + \alpha_{22})$ として扱う)、単軸は偏差成分 $\alpha_{11}$ で、降伏曲面の中心の移動量はその $\tfrac{3}{2}$ 倍である。

## パラメータ

| 名前 | 意味 | 単位 | 目安の値 |
|---|---|---|---|
| `E` | ヤング率 | MPa | 210000 |
| `nu` | ポアソン比 | — | 0.3 |
| `sigma_y0` | 初期降伏応力 | MPa | 250 |
| `C` | 移動硬化係数 | MPa | 20000 |
| `gamma` | 動的回復の係数 | — | 100 |
| `Qinf` | 等方硬化の飽和値 $Q_\infty$ (負なら軟化) | MPa | 100 |
| `b` | 等方硬化が飽和する速さ | — | 10 |

## 検証

- 共通の検査では、自動微分の経路が収束することと、整合接線が中心差分と合うことを確かめている。
- ファミリのテストは、飽和応力、降伏直後の塑性係数、$Q_\infty = 0$ で Armstrong-Frederick とビット単位で一致することを確かめている。
- 引張とせん断が同じ曲線になること、単軸応力の応答が 1D・3D・平面応力で一致することは、全モデルに共通のテストで確かめている。

## 文献

J.-L. Chaboche (1986), *Time-independent constitutive theories for cyclic plasticity*, International Journal of Plasticity.
