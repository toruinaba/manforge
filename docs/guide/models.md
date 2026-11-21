# 組み込みモデルと次元選択

モデルファミリごとに親クラス (物理) + 次元特化の具象クラスがある:

| ファミリ | クラス (`*` は `3D` / `PS` / `1D`) | 内容 | 定式化 |
|---|---|---|---|
| J2 等方硬化 | `J2Isotropic*` | 3D と 1D は閉形式のリターンマッピングを持つ | [式](models/j2_isotropic.md) |
| J2 + Lemaitre 損傷 | `J2IsotropicDamage*` | 弱連成 | [式](models/j2_damage.md) |
| 同上 + crack closure | `J2IsotropicDamageCrackClosure*` | 損傷の駆動力 Y を主値で引張/圧縮に分ける | 同上 |
| 同上 + 準片側 | `J2IsotropicDamageUnilateral*` | 応力も分解し、圧縮で剛性が回復する | 同上 |
| Armstrong-Frederick | `AFKinematic*` | 移動硬化 (閉形式で更新) | [式](models/af_kinematic.md) |
| Ohno-Wang 型 | `OWKinematic*` | 移動硬化 (implicit な発展則) | [式](models/ow_kinematic.md) |
| Yoshida-Uemori | `YUKinematic*` | 二曲面 + 停留曲面。3D と PS は手書きの NR と手導出ヤコビアンを持つ | [式](models/yu_kinematic.md) |

各ファミリには、次元を固定しない親クラス (`J2Isotropic`、`AFKinematic` など) もある。

定式化のページは、式、状態変数、パラメータ、飽和値などの性質、検証の範囲をファミリごとにまとめている。本文は各ファミリのディレクトリにある `README.md` で、モデルの隣に置いてある。

```{toctree}
:hidden:

models/j2_isotropic
models/j2_damage
models/af_kinematic
models/ow_kinematic
models/yu_kinematic
```

```python
from returnmap.models import AFKinematic3D

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
from returnmap.models import J2Isotropic
from returnmap.core import PLANE_STRAIN
model_pe = J2Isotropic(dimension=PLANE_STRAIN, E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)
```

`*3D` / `*PS` / `*1D` は次元を固定したクラスで、別の `dimension=` を渡すと `TypeError` になる (`J2Isotropic3D` の閉形式解のように、その次元でしか成り立たない代数を持つため)。

**パラメータはすべて必須で、既定値は無い。** コンストラクタの引数は `param_names` そのもので、キーワードで渡す。不足や綴り違いは名前を挙げて `TypeError` になる。UMAT の `PROPS` に既定値が無いのと同じ規則で、`p_D=0.0` のような「機能を切る値」も明示する。

## 損傷モデル (有効断面低下)

3 段階の J2 + Lemaitre 延性損傷。いずれも**弱連成** — 塑性は有効応力空間に閉じて D を見ないので、有効応力 σ̃ と ep は損傷なし J2 と一致し、D は公称応力 σ との写像と剛性にだけ効く。D は塑性増分でのみ進む (ΔD = (Y/S)^s·Δp、p > p_D)。

| クラス | 引張/圧縮の扱い | 状態変数 |
|---|---|---|
| `J2IsotropicDamage*` | 分けない: σ = (1−D)σ̃ | `stress`, `ep`, `D` |
| `J2IsotropicDamageCrackClosure*` | 駆動力 Y のみ主値で分解 (圧縮は h 倍) | 同上 |
| `J2IsotropicDamageUnilateral*` | 応力も分解: 引張 (1−D)E / 圧縮 (1−hD)E で剛性回復 | `stress`, `stress_eff` (σ̃), `ep`, `D` |

```python
from returnmap.models import J2IsotropicDamageUnilateral3D
model = J2IsotropicDamageUnilateral3D(
    E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_500.0,
    S=0.5, s=1.0, p_D=0.0,   # Lemaitre: 損傷強度 [MPa], 指数, 閾値 (相当塑性ひずみ)
    h=0.2,                   # crack-closure 係数 (CrackClosure / Unilateral のみ)
)
```

`Unilateral` は Lemaitre の準片側補エネルギー ψ*(σ, D) を唯一の定義とし、ε_e = ∂ψ*/∂σ・Y = ∂ψ*/∂D を autodiff で導く (弾性接線は対称)。注意: 臨界損傷 D_c・要素削除は無く D → 1 で特異になる。主値 0 での規約 ([モデルを作る](writing-models.md) の「非線形な弾性」) は暫定値。挙動の図は `examples/example_j2_damage*.py`。
