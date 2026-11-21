# returnmap

<!-- docs:start -->
弾塑性の構成則を Python で書いて計算し、Abaqus の UMAT (Fortran) まで確かめながら進めるためのフレームワーク。降伏関数と硬化則を書くだけで、リターンマッピングと整合接線 (consistent tangent) は自動微分で解かれる。使い道は 2 つある。

- **既存のモデルを使う** — 組み込みのモデルで応力-ひずみ応答を計算し、試験データに合わせ、モデルどうしを比べる。
- **自分のモデルを UMAT にする** — Python のモデルを基準にして、手で導いたリターンマッピングと Fortran の UMAT を段階ごとに照合する。

主な機能:

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
- 試験データに合わせる (手で調整 → フィット) 手順は、[チュートリアル](https://toruinaba.github.io/returnmap/guide/tutorial.html) を参照。
- 載荷履歴は `FieldHistory(FieldType.STRAIN, 配列)` で直接与えるか、`cyclic_strain` / `triangular_strain` / `sine_strain` / `decaying_cyclic_strain` (+ 各 stress 版) で作る。

実行できる使用例は `examples/` を参照。
<!-- docs:end -->

## ドキュメント

**<https://toruinaba.github.io/returnmap/>** で公開している。ガイド (日本語) と API リファレンス (英語、docstring から生成) がある。

**既存のモデルを使う**

| ページ | 内容 |
|---|---|
| [チュートリアル](https://toruinaba.github.io/returnmap/guide/tutorial.html) | 試験データにモデルを合わせる: 手で調整、フィット、パラメータを振る |
| [組み込みモデルと次元選択](https://toruinaba.github.io/returnmap/guide/models.html) | 参照モデルの一覧と定式化 (式、パラメータ、飽和値)、次元の選び方 |
| [シミュレーション実行](https://toruinaba.github.io/returnmap/guide/simulation.html) | ひずみ制御 / 応力制御 / 混合制御、載荷履歴の作り方、解けなかったとき |
| [パラメータフィッティング](https://toruinaba.github.io/returnmap/guide/fitting.html) | `fit_params`、`ParameterSpace`、複数カーブ、PCM |
| [モデルを比べる](https://toruinaba.github.io/returnmap/guide/comparing.html) | `compare_models`: 応答、計算コスト、フィットの良さ |

**自分のモデルを UMAT にする**

| ページ | 内容 |
|---|---|
| [新しいモデルを足す](https://toruinaba.github.io/returnmap/guide/adding-a-model.html) | Python のモデルから検証済みの UMAT までの手順: 置き場所、検証の宣言、手書きの解法、UMAT |
| [モデルを作る](https://toruinaba.github.io/returnmap/guide/writing-models.html) | Explicit / Implicit、`flow`、滑らかな切り替え、弾性ステップのフック、閉形式ソルバ |
| [検証ワークフロー](https://toruinaba.github.io/returnmap/guide/verification.html) | `check_model`、手導出ブロックの照合、接線の差分検査、2 つの実装の比較 |
| [Fortran UMAT クロス検証](https://toruinaba.github.io/returnmap/guide/fortran.html) | UMAT の宣言 → ビルド → Python との比較 |

**仕組みと、そのほか**

| ページ | 内容 |
|---|---|
| [概念](https://toruinaba.github.io/returnmap/concepts.html) | 三層構造、import 元の 7 パッケージ、integrator の種類、Voigt 規約と `FlowVector`、非収束の扱い |
| [サービスに組み込む](https://toruinaba.github.io/returnmap/guide/service.html) | `ModelRegistry`、`run_simulation` |
| [開発](https://toruinaba.github.io/returnmap/development.html) | テスト、例、ドキュメントのビルド |
| [変更履歴](https://toruinaba.github.io/returnmap/changelog.html) | 版ごとの変更と、0.1.0 からの移行の対応表 |
| [設計メモ](https://toruinaba.github.io/returnmap/design/api-redesign.html) | API 再設計の方針と決定の記録 |

文書のソースは [`docs/`](docs/) と、各モデルの `README.md` (定式化) にある。`main` に入るたびに自動でビルドされ、公開される。手元でビルドするには:

```bash
uv sync --extra docs
make docs            # → docs/_build/html/index.html
```

## License

[MIT License](LICENSE)
