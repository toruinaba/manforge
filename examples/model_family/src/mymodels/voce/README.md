# J2 + Voce 硬化

von Mises の降伏条件に、飽和する等方硬化 (Voce 則) を組み合わせたモデルである。

| クラス | 次元 | リターンマッピング | UMAT |
|---|---|---|---|
| `J2Voce` | 構築時に `dimension=` で選ぶ | 自動微分 | — |
| `J2Voce3D` | 3D ソリッド | 手書き (スカラーの Newton 反復) | `fortran/voce_3d.f90` |
| `J2Voce1D` | 単軸応力 | 自動微分 | — |

## 式

$$
f = \sigma_\mathrm{vm}(\boldsymbol\sigma) - \sigma_y(p),
\qquad
\sigma_y(p) = \sigma_{y0} + Q\left(1 - e^{-b p}\right)
$$

流れ則は関連流れで、$\mathrm{d}p = \mathrm{d}\lambda$ である。降伏応力は $\sigma_{y0} + Q$ に飽和し、降伏直後の硬化係数は $Q b$ である。

## 手書きの解法 (3D)

リターンは試行偏差応力の向きに沿うので、1 ステップはスカラーの方程式になる ($q$ は試行応力の Mises 応力、$\mu$ はせん断弾性係数)。

$$
g(\Delta\lambda) = q - 3\mu\,\Delta\lambda - \sigma_y(p_n + \Delta\lambda) = 0
$$

$g$ は下に凸で単調に減るので、$\Delta\lambda = 0$ からの Newton 反復は根を超えずに収束する。整合接線は、ステップ終了時の $\sigma_y$ と $H = \mathrm{d}\sigma_y/\mathrm{d}p$ を使って

$$
\mathbf{D} = \mathbf{I}_\mathrm{vol}\mathbf{C} + \theta\,\mathbf{I}_\mathrm{dev}\mathbf{C} - \beta\,\boldsymbol{s}^\mathrm{tr}\otimes\boldsymbol{s}^\mathrm{tr},
\qquad
\theta = 1 - \frac{3\mu\,\Delta\lambda}{q},
\qquad
\beta = \frac{9\mu^2\,(\sigma_y - H\,\Delta\lambda)}{(3\mu + H)\,q^3}
$$

## 状態変数とパラメータ

| 状態変数 | 種別 | 意味 |
|---|---|---|
| `ep` | Explicit | 相当塑性ひずみ $p$ |

| パラメータ | 意味 | 単位 | 目安の値 |
|---|---|---|---|
| `E` | ヤング率 | MPa | 210000 |
| `nu` | ポアソン比 | — | 0.3 |
| `sigma_y0` | 初期降伏応力 | MPa | 250 |
| `Q` | 硬化の飽和値 | MPa | 150 |
| `b` | 飽和の速さ | — | 20 |
