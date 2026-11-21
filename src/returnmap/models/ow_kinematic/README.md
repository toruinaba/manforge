# Ohno-Wang 型の移動硬化

Armstrong-Frederick の動的回復を、背応力の大きさで重み付けしたモデルである。離散化した発展則が閉じた形で解けないので、背応力を NR の未知数に入れて解く。returnmap では、implicit な状態変数を持つモデルの参照実装になっている。

| クラス | 次元 | リターンマッピング |
|---|---|---|
| `OWKinematic` | 構築時に `dimension=` で選ぶ | 自動微分 |
| `OWKinematic3D` | 3D ソリッド | 自動微分 |
| `OWKinematicPS` | 平面応力 | 自動微分 |
| `OWKinematic1D` | 単軸応力 | 自動微分 |

手書きの解法と UMAT は持たない。

## 式

降伏関数と流れ則は Armstrong-Frederick と同じである。

$$
f = \bar\xi - \sigma_{y0},
\qquad
\boldsymbol\xi = \boldsymbol{s} - \boldsymbol\alpha,
\qquad
\bar\xi = \sqrt{\tfrac{3}{2}\,\boldsymbol\xi:\boldsymbol\xi},
\qquad
\boldsymbol{n} = \frac{3}{2}\frac{\boldsymbol\xi}{\bar\xi}
$$

違うのは動的回復の項で、背応力の大きさ $\bar\alpha = \sqrt{\tfrac{3}{2}\,\boldsymbol\alpha:\boldsymbol\alpha}$ が掛かる。背応力が小さいうちは回復がほとんど働かず、大きくなると急に効く。

$$
\mathrm{d}\boldsymbol\alpha = \tfrac{2}{3}\,C_k\,\boldsymbol{n}\,\mathrm{d}p - \gamma\,\bar\alpha\,\boldsymbol\alpha\,\mathrm{d}p
$$

後退 Euler で離散化すると $\boldsymbol\alpha_{n+1}$ について非線形になるので、残差として書く。

$$
\boldsymbol{R}_\alpha = \boldsymbol\alpha_{n+1} - \boldsymbol\alpha_n - \tfrac{2}{3}\,C_k\,\Delta\lambda\,\boldsymbol{n}_{n+1} + \gamma\,\Delta\lambda\,\bar\alpha_{n+1}\,\boldsymbol\alpha_{n+1} = \boldsymbol{0}
$$

NR の未知数は $[\boldsymbol\sigma, \Delta\lambda, \boldsymbol\alpha, p]$ で、残差は応力の行 (関連流れの既定の形)、降伏条件 $f = 0$、$\boldsymbol{R}_\alpha$、$R_p = p_{n+1} - p_n - \Delta\lambda$ である。

### 文献のモデルとの関係

実装しているのは上の 1 本の発展則で、Ohno と Wang が提案した「動的回復を背応力の大きさで制御する」考え方を、背応力 1 つ・べき指数 1 の形に簡略化したものである。複数の背応力の重ね合わせ、臨界状態 $\bar\alpha_i = r_i$ に対するべき乗、塑性ひずみ速度の背応力方向への射影は入っていない。文献のモデルそのものが要るときは、このモデルを出発点にして発展則を書き換える。

## 性質

- **飽和**: 単調載荷では背応力の大きさが $\sqrt{C_k/\gamma}$ に飽和する。単軸の応力は $\sigma_{y0} + \sqrt{C_k/\gamma}$ に漸近する (目安の値では 250 + 14.1 = 264.1 MPa)。
- **初期の硬化係数**: 降伏直後の単軸の塑性係数は $C_k$ である。
- **$\gamma = 0$**: Prager の線形移動硬化になる (Armstrong-Frederick と同じ極限)。
- **$\gamma$ の単位**: $\gamma\,\bar\alpha$ が無次元になるので、$\gamma$ は応力の逆数の単位を持つ。Armstrong-Frederick の $\gamma$ (無次元) と同じ値を入れても同じ飽和値にはならない。

## 状態変数

| 名前 | 種別 | 意味 |
|---|---|---|
| `stress` | Implicit | Cauchy 応力 |
| `alpha` | Implicit | 背応力 $\boldsymbol\alpha$ (偏差) |
| `ep` | Implicit | 相当塑性ひずみ $p$ |

次元ごとの `alpha` の持ち方は Armstrong-Frederick と同じである (単軸は偏差成分 $\alpha_{11}$ を持ち、応力の単位で見た背応力はその $\tfrac{3}{2}$ 倍)。

## パラメータ

| 名前 | 意味 | 単位 | 目安の値 |
|---|---|---|---|
| `E` | ヤング率 | MPa | 210000 |
| `nu` | ポアソン比 | — | 0.3 |
| `sigma_y0` | 降伏応力 (一定) | MPa | 250 |
| `C_k` | 移動硬化係数 | MPa | 20000 |
| `gamma` | 動的回復の係数 | 1/MPa | 100 |

## 検証

共通の検査では、自動微分の経路が収束することと、整合接線が中心差分と合うことを確かめている。単軸の飽和応力が $\sigma_{y0} + \sqrt{C_k/\gamma}$ になることは、20 % までの単調載荷で確かめた。

## 文献

N. Ohno, J.-D. Wang (1993), *Kinematic hardening rules with critical state of dynamic recovery*, International Journal of Plasticity.
