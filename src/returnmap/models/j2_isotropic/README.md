# J2 等方硬化

von Mises の降伏条件に線形の等方硬化を組み合わせた、最も単純な参照モデルである。3D と単軸は閉じた形のリターンマッピングと接線を持ち、3D には UMAT がある。

| クラス | 次元 | リターンマッピング | UMAT |
|---|---|---|---|
| `J2Isotropic` | 構築時に `dimension=` で選ぶ | 自動微分 | — |
| `J2Isotropic3D` | 3D ソリッド | 閉じた形 (半径方向のリターン) | `fortran/j2_isotropic_3d.f90` |
| `J2IsotropicPS` | 平面応力 | 自動微分 | — |
| `J2Isotropic1D` | 単軸応力 | 閉じた形 | — |

## 式

降伏関数は、Mises 応力と降伏応力の差である。

$$
f = \sigma_\mathrm{vm}(\boldsymbol\sigma) - (\sigma_{y0} + H\,p),
\qquad
\sigma_\mathrm{vm} = \sqrt{\tfrac{3}{2}\,\boldsymbol{s}:\boldsymbol{s}}
$$

$\boldsymbol{s}$ は偏差応力、$p$ は相当塑性ひずみである。流れ則は関連流れで、塑性乗数の増分がそのまま $p$ の増分になる。

$$
\mathrm{d}\boldsymbol\varepsilon^p = \mathrm{d}\lambda\,\boldsymbol{n},
\qquad
\boldsymbol{n} = \frac{\partial f}{\partial\boldsymbol\sigma} = \frac{3}{2}\frac{\boldsymbol{s}}{\sigma_\mathrm{vm}},
\qquad
\mathrm{d}p = \mathrm{d}\lambda
$$

1 ステップは後退 Euler で積分する。試行応力を $\boldsymbol\sigma^\mathrm{tr} = \boldsymbol\sigma_n + \mathbf{C}\,\Delta\boldsymbol\varepsilon$ として、

$$
\boldsymbol\sigma_{n+1} = \boldsymbol\sigma^\mathrm{tr} - \Delta\lambda\,\mathbf{C}\,\boldsymbol{n}_{n+1},
\qquad
p_{n+1} = p_n + \Delta\lambda,
\qquad
f_{n+1} = 0
$$

## 閉じた形 (3D と単軸)

3D では $\mathbf{C}\,\boldsymbol{n} = 3\mu\,\boldsymbol{s}^\mathrm{tr}/q$ ($q = \sigma_\mathrm{vm}(\boldsymbol\sigma^\mathrm{tr})$、$\mu$ はせん断弾性係数) が成り立ち、リターンは試行偏差応力の向きに沿う。未知数は $\Delta\lambda$ だけになり、硬化が線形なので 1 回で解ける。

$$
\Delta\lambda = \frac{q - (\sigma_{y0} + H p_n)}{3\mu + H},
\qquad
\boldsymbol\sigma_{n+1} = \boldsymbol\sigma^\mathrm{tr} - \frac{3\mu\,\Delta\lambda}{q}\,\boldsymbol{s}^\mathrm{tr}
$$

整合接線も閉じた形になる。

$$
\mathbf{D} = \mathbf{I}_\mathrm{vol}\mathbf{C} + \theta\,\mathbf{I}_\mathrm{dev}\mathbf{C} - \beta\,\boldsymbol{s}^\mathrm{tr}\otimes\boldsymbol{s}^\mathrm{tr},
\qquad
\theta = 1 - \frac{3\mu\,\Delta\lambda}{q},
\qquad
\beta = \frac{9\mu^2\,(\sigma_{y0} + H p_n)}{(3\mu + H)\,q^3}
$$

単軸応力では、同じ式の $3\mu$ が $E$ に替わる: $\Delta\lambda = (q - \sigma_y)/(E + H)$、接線は $E H/(E + H)$ である。

平面応力は面外の応力を 0 に保つ拘束があり、$\mathbf{C}\,\boldsymbol{n}$ が試行偏差応力の向きにならない。そのため閉じた形を持たず、自動微分の NR で解く。

## 状態変数

| 名前 | 種別 | 意味 |
|---|---|---|
| `stress` | — | Cauchy 応力 |
| `ep` | Explicit | 相当塑性ひずみ $p$ |

NR の未知数は $[\boldsymbol\sigma, \Delta\lambda]$ である。

## パラメータ

| 名前 | 意味 | 単位 | 目安の値 |
|---|---|---|---|
| `E` | ヤング率 | MPa | 210000 |
| `nu` | ポアソン比 | — | 0.3 |
| `sigma_y0` | 初期降伏応力 | MPa | 250 |
| `H` | 等方硬化係数 (線形) | MPa | 1000 |

## 検証

- 3D と単軸は、閉じた形の解と接線を自動微分の経路と照合している。
- 3D の UMAT は、標準の載荷履歴の全ステップで Python と一致する (応力・状態・接線とも 1e-15 程度)。
- 単軸の塑性域の傾きが $E H/(E + H)$ になることを確かめている。
