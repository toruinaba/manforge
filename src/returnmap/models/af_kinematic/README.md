# Armstrong-Frederick 移動硬化

von Mises の降伏曲面が、大きさを変えずに応力空間を移動するモデルである。背応力の発展則に動的回復の項があり、背応力は有限の値に飽和する。繰返し載荷での Bauschinger 効果を表す最小のモデルになる。

| クラス | 次元 | リターンマッピング |
|---|---|---|
| `AFKinematic` | 構築時に `dimension=` で選ぶ | 自動微分 |
| `AFKinematic3D` | 3D ソリッド | 自動微分 |
| `AFKinematicPS` | 平面応力 | 自動微分 |
| `AFKinematic1D` | 単軸応力 | 自動微分 |

手書きの解法と UMAT は持たない。

## 式

降伏関数は、背応力 $\boldsymbol\alpha$ (偏差) だけ中心を移した Mises 応力である。降伏曲面の大きさは変わらない。

$$
f = \sqrt{\tfrac{3}{2}\,\boldsymbol\xi:\boldsymbol\xi} - \sigma_{y0},
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

背応力は、塑性ひずみに比例して進む項と、背応力自身に比例して戻る項 (動的回復) を持つ。

$$
\mathrm{d}\boldsymbol\alpha = \tfrac{2}{3}\,C_k\,\boldsymbol{n}\,\mathrm{d}p - \gamma\,\boldsymbol\alpha\,\mathrm{d}p
$$

後退 Euler で離散化すると、$\boldsymbol\alpha_{n+1}$ について閉じた形で解ける。

$$
\boldsymbol\alpha_{n+1} = \frac{\boldsymbol\alpha_n + \tfrac{2}{3}\,C_k\,\Delta\lambda\,\boldsymbol{n}_{n+1}}{1 + \gamma\,\Delta\lambda}
$$

このため背応力は Explicit の状態変数で、NR の未知数は $[\boldsymbol\sigma, \Delta\lambda]$ のままである。

## 性質

- **飽和**: 単調載荷では背応力の大きさが $C_k/\gamma$ に飽和する。単軸の応力は $\sigma_{y0} + C_k/\gamma$ に漸近する (目安の値では 250 + 200 = 450 MPa)。
- **初期の硬化係数**: 降伏直後の単軸の塑性係数は $C_k$ である ($\tfrac{2}{3}$ と $\boldsymbol{n}$ の中の $\tfrac{3}{2}$ が打ち消す)。
- **$\gamma = 0$**: Prager の線形移動硬化になる。

## 状態変数

| 名前 | 種別 | 意味 |
|---|---|---|
| `stress` | — | Cauchy 応力 |
| `alpha` | Explicit | 背応力 $\boldsymbol\alpha$ (偏差) |
| `ep` | Explicit | 相当塑性ひずみ $p$ |

次元ごとの `alpha` の持ち方は次のとおりである。

- **3D**: 6 成分の偏差テンソル。
- **平面応力**: $[\alpha_{11}, \alpha_{22}, \alpha_{12}]$ を持ち、$\alpha_{33} = -(\alpha_{11} + \alpha_{22})$ として扱う。
- **単軸**: 偏差成分 $\alpha_{11}$ を持つ。降伏曲面の中心の移動量 (応力の単位で見た背応力) はその $\tfrac{3}{2}$ 倍である。

## パラメータ

| 名前 | 意味 | 単位 | 目安の値 |
|---|---|---|---|
| `E` | ヤング率 | MPa | 210000 |
| `nu` | ポアソン比 | — | 0.3 |
| `sigma_y0` | 降伏応力 (一定) | MPa | 250 |
| `C_k` | 移動硬化係数 | MPa | 20000 |
| `gamma` | 動的回復の係数 | — | 100 |

## 検証

共通の検査では、自動微分の経路が収束することと、整合接線が中心差分と合うことを確かめている。単軸の飽和応力が $\sigma_{y0} + C_k/\gamma$ になることは、20 % までの単調載荷で確かめた。

## 文献

P. J. Armstrong, C. O. Frederick (1966), *A mathematical representation of the multiaxial Bauschinger effect*, CEGB Report RD/B/N731.
