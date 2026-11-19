# manforge

<!-- docs:start -->
Fortran UMAT (Abaqus) の検証・フィッティング用 Python フレームワーク。降伏関数と硬化則を実装するだけで、return mapping / consistent tangent (autograd 自動微分) / パラメータフィッティング / Fortran クロス検証がすべて使える。

- autograd による **consistent tangent 自動計算** — 手動でアルゴリズム接線を導出する必要なし
- **汎用 NR ソルバ** — スカラー Δλ / ベクトル `[σ, Δλ, q]` の 2 経路を自動選択。Broyden 更新 (`nr_jacobian="broyden"`) で数倍高速化
- **多次元対応** — 3D ソリッド / 平面ひずみ / 平面応力 / 1D を `StressDimension` で統一管理
- **手導出の照合** — 手で導出したヤコビアンのブロックをモデル上で宣言し、自動微分とブロックごとに照合
- **組み込みモデル** — J2 等方硬化 / J2 + Lemaitre 損傷 / Armstrong-Frederick / Ohno-Wang / YU を同梱
- **サービス契約層** — WebAPI などへの組み込み用に plain な入出力の `ModelRegistry` / `run_simulation` を提供

## インストール

現状は PyPI 未公開のため、ソースからの導入が正:

```bash
git clone <this-repo> && cd manforge
uv sync --extra dev          # テスト依存含む
uv sync --all-extras         # matplotlib / Fortran ビルド依存含む

# Fortran モジュールのビルド (gfortran 必須、無ければ Docker: make docker-build)
uv run manforge build fortran/abaqus_stubs.f90 fortran/j2_isotropic_3d.f90 --name j2_isotropic_3d
```

PyPI 公開後は `pip install manforge` (+ extras: `[dev]` / `[examples]` / `[fortran]`)。

## クイックスタート

```python
import numpy as np
from manforge.models import J2Isotropic3D
from manforge.simulation import StrainDriver, PythonIntegrator, FieldHistory, FieldType

model = J2Isotropic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)

strain_history = np.zeros((100, 6))
strain_history[:, 0] = np.linspace(0.0, 5e-3, 100)  # ε11 を増加
load = FieldHistory(FieldType.STRAIN, "Strain", strain_history)

result = StrainDriver(PythonIntegrator(model)).run(load)
print(f"Final σ11: {result.stress[-1, 0]:.1f} MPa")
```

繰返し載荷は `FieldHistory` のコンストラクタ群で生成できる:

```python
load = FieldHistory.cyclic_strain(peaks=[5e-3, -5e-3, 5e-3], n_per_segment=25, ntens=6)
# 他: triangular_strain / sine_strain / decaying_cyclic_strain (+ 各 stress 版)
```

実行できる使用例は `examples/` を参照。
<!-- docs:end -->

## ドキュメント

ガイドは [`docs/`](docs/) にある (日本語)。API リファレンスは docstring (英語) から生成する。

| ページ | 内容 |
|---|---|
| [概念](docs/concepts.md) | 三層構造、import 元の 7 パッケージ、integrator の種類、Voigt 規約と `FlowVector`、非収束の扱い |
| [組み込みモデルと次元選択](docs/guide/models.md) | 参照モデル、次元の選び方、損傷モデル |
| [シミュレーション実行](docs/guide/simulation.md) | ひずみ制御 / 応力制御 / 混合制御、状態量の履歴 |
| [パラメータフィッティング](docs/guide/fitting.md) | `fit_params`、`ParameterSpace`、複数カーブ、PCM |
| [モデルを作る](docs/guide/writing-models.md) | Explicit / Implicit、`flow`、弾性ステップのフック、閉形式ソルバ |
| [検証ワークフロー](docs/guide/verification.md) | ヤコビアンの照合、手導出ブロック、FD tangent、クロスチェック |
| [Fortran UMAT クロス検証](docs/guide/fortran.md) | ビルド → 読み込み → 比較 |
| [サービスに組み込む](docs/guide/service.md) | `ModelRegistry`、`run_simulation` |
| [開発](docs/development.md) | テスト、例、ドキュメントのビルド |
| [設計メモ](docs/design/api-redesign.md) | API 再設計の方針と決定の記録 |

HTML 版 (API リファレンスを含む) は手元でビルドする:

```bash
uv sync --extra docs
make docs            # → docs/_build/html/index.html
```

## License

[MIT License](LICENSE)
