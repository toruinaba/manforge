# J2 等方硬化 + Lemaitre 損傷

J2 等方硬化に、Lemaitre の延性損傷 (有効断面の減少) を弱連成で組み合わせたモデルである。引張と圧縮の扱いが違う 3 つの段階がある。

| クラス (`*` は `3D` / `PS` / `1D`) | 引張と圧縮の扱い |
|---|---|
| `J2IsotropicDamage*` | 分けない |
| `J2IsotropicDamageCrackClosure*` | 損傷を進める力 $Y$ だけを分ける |
| `J2IsotropicDamageUnilateral*` | 応力も分ける。圧縮で剛性が回復する |

どれも自動微分で解く。手書きの解法と UMAT は持たない。

## 共通の考え方: 弱連成

損傷 $D$ ($0$ = 健全) は、荷重を受け持つ断面が減ることを表す。断面が減った分だけ高くなった応力を有効応力 $\tilde{\boldsymbol\sigma}$ と呼ぶ。

塑性は有効応力の空間で進み、$D$ を見ない。

$$
f = \sigma_\mathrm{vm}(\tilde{\boldsymbol\sigma}) - (\sigma_{y0} + H\,p),
\qquad
\mathrm{d}\boldsymbol\varepsilon^p = \mathrm{d}\lambda\,\frac{\partial f}{\partial\tilde{\boldsymbol\sigma}},
\qquad
\mathrm{d}p = \mathrm{d}\lambda
$$

したがって $\tilde{\boldsymbol\sigma}$ と $p$ の履歴は、損傷の無い J2 等方硬化と一致する。$D$ が効くのは、有効応力と公称応力 $\boldsymbol\sigma$ の対応と、弾性剛性だけである。

損傷は塑性ひずみとともに進む。$p$ がしきい値 $p_D$ を超えた分だけが効く。

$$
\mathrm{d}D = \left(\frac{Y}{S}\right)^{s}\mathrm{d}p \quad (p > p_D)
$$

$Y$ はエネルギー解放率で、3 つの段階で定義が違う。後退 Euler の残差は次の形である。

$$
R_D = E\left[D_{n+1} - D_n - \left(\frac{Y_{n+1}}{S}\right)^{s}\Delta p_D\right],
\qquad
\Delta p_D = \langle p_{n+1} - p_D\rangle - \langle p_n - p_D\rangle
$$

$E$ を掛けているのは、ほかの行と同じ応力の単位にそろえるためである (NR の許容値は応力の大きさで決まる)。$\langle\cdot\rangle$ は滑らかにした Macaulay 括弧で、$p = p_D$ での折れを接線に残さない。

## 段階 1: 引張と圧縮を分けない

$$
\boldsymbol\sigma = (1 - D)\,\tilde{\boldsymbol\sigma},
\qquad
Y = \tfrac{1}{2}\,\tilde{\boldsymbol\sigma}:\mathbf{C}_0^{-1}:\tilde{\boldsymbol\sigma}
= \frac{1}{2E}\left[\tfrac{2}{3}(1 + \nu)\,\tilde\sigma_\mathrm{vm}^2 + 3(1 - 2\nu)\,\tilde\sigma_H^2\right]
$$

$\mathbf{C}_0$ は健全な材料の弾性剛性、$\tilde\sigma_H$ は有効応力の静水圧成分である。弾性剛性は $(1 - D)\,\mathbf{C}_0$ になる。

応力の行は、有効応力の空間でのリターン $\tilde{\boldsymbol\sigma} = \tilde{\boldsymbol\sigma}^\mathrm{tr} - \Delta\lambda\,\mathbf{C}_0\,\tilde{\boldsymbol{n}}$ に $(1 - D_n)$ を掛けた形で書く。

$$
\boldsymbol{R}_\sigma = \frac{1 - D_n}{1 - D_{n+1}}\,\boldsymbol\sigma_{n+1} - \boldsymbol\sigma^\mathrm{tr} + \Delta\lambda\,(1 - D_n)\,\mathbf{C}_0\,\tilde{\boldsymbol{n}}_{n+1}
$$

既定の残差 $\boldsymbol\sigma - \boldsymbol\sigma^\mathrm{tr} + \Delta\lambda\,\mathbf{C}(D)\,\partial f/\partial\boldsymbol\sigma$ は使えない。弾性ひずみが同じでも損傷が進むと応力が下がる、その分が抜けるためである。

## 段階 2: 損傷を進める力を分ける (crack closure)

$Y$ を、有効応力の主値 $\tilde\sigma_i$ の符号で引張と圧縮に分ける。圧縮の寄与には $h$ ($0 \le h \le 1$) を掛ける。

$$
Y = \frac{1 + \nu}{2E}\left[\sum_i\langle\tilde\sigma_i\rangle_+^2 + h\sum_i\langle\tilde\sigma_i\rangle_-^2\right]
- \frac{\nu}{2E}\left[\langle\operatorname{tr}\tilde{\boldsymbol\sigma}\rangle_+^2 + h\,\langle\operatorname{tr}\tilde{\boldsymbol\sigma}\rangle_-^2\right]
$$

- $h = 1$ で段階 1 の $Y$ に一致する。$h = 0$ では、圧縮だけの状態で損傷が進まない。
- 分けるのは $Y$ だけで、応力は $\boldsymbol\sigma = (1 - D)\,\tilde{\boldsymbol\sigma}$ のままである。圧縮に転じても剛性は回復しない。

## 段階 3: 応力も分ける (準片側)

公称応力の主値で分けた補エネルギーを、弾性の唯一の定義にする。

$$
\rho\psi^*(\boldsymbol\sigma, D) = \frac{1 + \nu}{2E}\left[\frac{\boldsymbol\sigma_+:\boldsymbol\sigma_+}{1 - D} + \frac{\boldsymbol\sigma_-:\boldsymbol\sigma_-}{1 - hD}\right]
- \frac{\nu}{2E}\left[\frac{\langle\operatorname{tr}\boldsymbol\sigma\rangle_+^2}{1 - D} + \frac{\langle\operatorname{tr}\boldsymbol\sigma\rangle_-^2}{1 - hD}\right]
$$

$$
\boldsymbol\varepsilon^e = \frac{\partial\psi^*}{\partial\boldsymbol\sigma},
\qquad
Y = \frac{\partial\psi^*}{\partial D},
\qquad
\tilde{\boldsymbol\sigma} = \mathbf{C}_0:\boldsymbol\varepsilon^e
$$

- 単軸では、剛性が引張で $(1 - D)E$、圧縮で $(1 - hD)E$ になる。圧縮に転じると剛性が回復する。
- 弾性がポテンシャルから導かれるので、弾性接線は対称で、閉じた弾性サイクルは仕事をしない。単純な分解 $\boldsymbol\sigma = (1 - D)\,\tilde{\boldsymbol\sigma}_+ + (1 - hD)\,\tilde{\boldsymbol\sigma}_-$ は、$\nu \ne 0$ でこれを満たさない。
- $\boldsymbol\varepsilon^e$ と $Y$ は、どちらも $\psi^*$ の自動微分で求める。

有効応力から公称応力への対応 $\tilde{\boldsymbol\sigma} = \mathbf{C}_0:\partial\psi^*/\partial\boldsymbol\sigma(\boldsymbol\sigma, D)$ は閉じた形で逆に解けない。そこで有効応力も未知数に入れる。NR の未知数は $[\boldsymbol\sigma, \Delta\lambda, \tilde{\boldsymbol\sigma}, D]$ で、残差は次の 4 つである。

$$
\begin{aligned}
\boldsymbol{R}_\sigma &= \tilde{\boldsymbol\sigma} - \mathbf{C}_0:\frac{\partial\psi^*}{\partial\boldsymbol\sigma}(\boldsymbol\sigma, D_{n+1}) \\
R_{\Delta\lambda} &= \sigma_\mathrm{vm}(\tilde{\boldsymbol\sigma}) - (\sigma_{y0} + H\,p) \\
\boldsymbol{R}_{\tilde\sigma} &= \tilde{\boldsymbol\sigma} - \tilde{\boldsymbol\sigma}^\mathrm{tr} + \Delta\lambda\,\mathbf{C}_0\,\tilde{\boldsymbol{n}} \\
R_D &= E\left[D - D_n - \left(Y(\boldsymbol\sigma, D)/S\right)^{s}\Delta p_D\right]
\end{aligned}
$$

弾性予測子も線形ではない。$\tilde{\boldsymbol\sigma}^\mathrm{tr} = \tilde{\boldsymbol\sigma}_n + \mathbf{C}_0\,\Delta\boldsymbol\varepsilon$ を作り、$D$ を固定して上の対応を解いて $\boldsymbol\sigma^\mathrm{tr}$ を求める。この逆写像は、陰関数定理による導関数を持つ自動微分の基本演算として実装してあり、弾性接線と塑性の整合接線はそれを通して厳密に微分される。

## 状態変数

| 名前 | 種別 | 意味 | 持つクラス |
|---|---|---|---|
| `stress` | Implicit | 公称応力 $\boldsymbol\sigma$ | 全部 |
| `stress_eff` | Implicit | 有効応力 $\tilde{\boldsymbol\sigma}$ | `Unilateral` |
| `ep` | Explicit | 相当塑性ひずみ $p$ | 全部 |
| `D` | Implicit | 損傷 | 全部 |

## パラメータ

| 名前 | 意味 | 単位 | 目安の値 | 持つクラス |
|---|---|---|---|---|
| `E` | 健全な材料のヤング率 | MPa | 210000 | 全部 |
| `nu` | ポアソン比 | — | 0.3 | 全部 |
| `sigma_y0` | 初期降伏応力 (有効応力) | MPa | 250 | 全部 |
| `H` | 等方硬化係数 (有効応力、線形) | MPa | 1000 | 全部 |
| `S` | 損傷の強さ | MPa | 1 | 全部 |
| `s` | 損傷の指数 | — | 1 | 全部 |
| `p_D` | 損傷が始まる相当塑性ひずみ | — | 0 | 全部 |
| `h` | 圧縮での効き方 ($0$〜$1$) | — | 0.2 | `CrackClosure`、`Unilateral` |

`Unilateral` はこのほかに、主値が 0 のときの Heaviside 関数の値 `c` を構築時に受け取る (既定 $1/2$)。数値上の設定で、材料パラメータではないので `PROPS` には入らない。

## 制約

- **損傷は塑性ステップでしか進まない。** 弾性ステップ (除荷を含む) では $D$ が変わらない。
- **破壊の判定は無い。** 臨界損傷 $D_c$ も要素の削除も無く、$D \to 1$ で応力の行が特異になる。
- **主値が 0 のときの規約 `c` は暫定値である。** UMAT 側の規約に合わせて決める。

## 検証

共通の検査では、9 つのクラスについて、自動微分の経路が収束することと、整合接線が中心差分と合うことを確かめている。ファミリのテストは、有効応力と $p$ の履歴が損傷の無い J2 と一致すること (弱連成) を確かめている。

## 文献

- J. Lemaitre (1985), *A continuous damage mechanics model for ductile fracture*, Journal of Engineering Materials and Technology.
- J. Lemaitre, R. Desmorat (2005), *Engineering Damage Mechanics*, Springer. 準片側の補エネルギーはこちらによる。
