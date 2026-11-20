# returnmap

<!-- docs:start -->
Fortran UMAT (Abaqus) の検証・フィッティング用 Python フレームワーク。降伏関数と硬化則を実装するだけで、return mapping / consistent tangent (autograd 自動微分) / パラメータフィッティング / Fortran クロス検証がすべて使える。

- autograd による **consistent tangent 自動計算** — 手動でアルゴリズム接線を導出する必要なし
- **汎用 NR ソルバ** — 未知数は `[σ, Δλ, implicit 状態]`。閉形式で更新できる状態 (explicit) は未知数に入らない。Broyden 更新 (`nr_jacobian="broyden"`) で数倍高速化
- **多次元対応** — 3D ソリッド / 平面ひずみ / 平面応力 / 1D を `StressDimension` で統一管理
- **手導出の照合** — 手で導出したヤコビアンのブロックをモデル上で宣言し、自動微分とブロックごとに照合
- **組み込みモデル** — J2 等方硬化 / J2 + Lemaitre 損傷 / Armstrong-Frederick / Ohno-Wang / YU を同梱
- **サービス契約層** — WebAPI などへの組み込み用に plain な入出力の `ModelRegistry` / `run_simulation` を提供

## インストール

**使う** — PyPI には未公開なので、GitHub のタグから入れる (依存は numpy / scipy / autograd):

```bash
pip install "git+https://github.com/toruinaba/returnmap@v0.3.2"
# 図も描くなら matplotlib を足す:
pip install "returnmap[examples] @ git+https://github.com/toruinaba/returnmap@v0.3.2"
```

**開発する** — リポジトリを取ってきて、テストや Fortran のビルドまで含めた環境を作る:

```bash
git clone https://github.com/toruinaba/returnmap && cd returnmap
uv sync --extra dev          # テスト依存含む
uv sync --all-extras         # matplotlib / ドキュメント / Fortran ビルド依存含む

# Fortran モジュールのビルド (gfortran 必須、無ければ Docker: make docker-build)
uv run returnmap build --abaqus-stubs src/returnmap/models/j2_isotropic/fortran/j2_isotropic_3d.f90 --name j2_isotropic_3d
```

## クイックスタート

単軸の引張・圧縮試験を、既存のモデルで計算する (`*1D` のモデルは単軸応力):

```python
from returnmap.models import AFKinematic1D
from returnmap.simulation import FieldHistory, PythonIntegrator, StrainDriver

model = AFKinematic1D(E=206_000.0, nu=0.3, sigma_y0=250.0, C_k=20_000.0, gamma=100.0)
load = FieldHistory.cyclic_strain([0.015, -0.015, 0.015], n_per_segment=50)   # ±1.5 % の繰返し

result = StrainDriver(PythonIntegrator(model)).run(load)
result.strain[:, 0], result.stress[:, 0]      # 応力-ひずみ曲線
result.history("ep")                          # 状態量の履歴
```

- 各パラメータの意味・単位・目安の値は `AFKinematic1D.param_meta` にある。
- 試験データに合わせる (手で調整 → フィット) 手順は、[チュートリアル](https://github.com/toruinaba/returnmap/blob/main/docs/guide/tutorial.md) を参照。
- 載荷履歴は `FieldHistory(FieldType.STRAIN, 配列)` で直接与えるか、`cyclic_strain` / `triangular_strain` / `sine_strain` / `decaying_cyclic_strain` (+ 各 stress 版) で作る。

実行できる使用例は `examples/` を参照。
<!-- docs:end -->

## ドキュメント

ガイドは [`docs/`](docs/) にある (日本語)。API リファレンスは docstring (英語) から生成する。

| ページ | 内容 |
|---|---|
| [チュートリアル](docs/guide/tutorial.md) | 試験データにモデルを合わせる: 手で調整、フィット、パラメータを振る |
| [概念](docs/concepts.md) | 三層構造、import 元の 7 パッケージ、integrator の種類、Voigt 規約と `FlowVector`、非収束の扱い |
| [組み込みモデルと次元選択](docs/guide/models.md) | 参照モデル、次元の選び方、損傷モデル |
| [シミュレーション実行](docs/guide/simulation.md) | ひずみ制御 / 応力制御 / 混合制御、状態量の履歴 |
| [パラメータフィッティング](docs/guide/fitting.md) | `fit_params`、`ParameterSpace`、複数カーブ、PCM |
| [モデルを比べる](docs/guide/comparing.md) | `compare_models`: 応答、計算コスト、フィットの良さ |
| [モデルを作る](docs/guide/writing-models.md) | Explicit / Implicit、`flow`、弾性ステップのフック、閉形式ソルバ |
| [検証ワークフロー](docs/guide/verification.md) | ヤコビアンの照合、手導出ブロック、FD tangent、クロスチェック |
| [Fortran UMAT クロス検証](docs/guide/fortran.md) | ビルド → 読み込み → 比較 |
| [サービスに組み込む](docs/guide/service.md) | `ModelRegistry`、`run_simulation` |
| [開発](docs/development.md) | テスト、例、ドキュメントのビルド |
| [変更履歴](CHANGELOG.md) | 版ごとの変更と、0.1.0 からの移行の対応表 |
| [設計メモ](docs/design/api-redesign.md) | API 再設計の方針と決定の記録 |

HTML 版 (API リファレンスを含む) は手元でビルドする:

```bash
uv sync --extra docs
make docs            # → docs/_build/html/index.html
```

## License

[MIT License](LICENSE)
