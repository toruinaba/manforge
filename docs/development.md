# 開発

## テスト

```bash
make test          # 高速テスト (フレームワークの unit + integration と全モデル、slow/fortran 除く)
make test-models   # モデルのテストだけ (slow を含む)
make test-all      # 全スイート
uv run pytest src/returnmap/models/yu_kinematic -m "not fortran"    # 1 ファミリだけ
```

その他のターゲット (e2e / slow / docker) は `Makefile` を参照。テストは 2 箇所にある。

- **フレームワークのテスト** — `tests/{unit,integration}` が `src/returnmap` のモジュール構成に対応する (モジュールごとに unit か integration のどちらか)。複数のモデルにまたがるテストもここ (`tests/integration/models/`)。
- **モデルのテスト** — そのモデルのファミリの中、`src/returnmap/models/<family>/tests/` にある。pip で入れた先でも `pytest --pyargs returnmap.models.yu_kinematic` で実行できる。

### モデルのテスト

ファミリのディレクトリは、そのモデルに属するものを全部持つ: モデル本体、UMAT のソース (`fortran/`)、テスト (`tests/`)。テストは公開の import パス (`returnmap.core` など) とそのファミリの中のものだけを import する — ディレクトリごと動かせるようにするためである。

2 つの独立した実装を比べるテスト (手書きの経路と自動微分、Python と UMAT) は、次の名前と基準値に揃える。外れるものは、そのモジュールの docstring に理由を書く。

| ファイル | 比べるもの |
|---|---|
| `test_analytical_vs_numerical.py` | 手書きのリターンマッピング vs フレームワークの NR |
| `test_numerical_vs_fortran.py` | Python vs コンパイルした UMAT (`@pytest.mark.fortran`。モジュールが無ければ skip) |

| 量 | 許容値 |
|---|---|
| 応力 | atol=1e-6 (絶対値、MPa) |
| Δλ、ep | atol=1e-10 (絶対値) |
| 接線 | 最大相対誤差 < 1e-5 |

## 例

| スクリプト | 内容 |
|-----------|------|
| `examples/example_j2_uniaxial.py` | 繰返し単軸 + tangent 検証 + ヒステリシスのプロット |
| `examples/example_j2_damage.py` | J2 + Lemaitre 損傷: 損傷の進展と剛性低下 |
| `examples/example_j2_damage_crack_closure.py` | crack closure 係数 h による引張/圧縮の違い |
| `examples/example_j2_damage_unilateral.py` | 準片側モデル: 圧縮での剛性回復 |
| `examples/example_fit_j2.py` | パラメータフィッティング (基本 / ParameterSpace / 複数ターゲット / PCM) |
| `examples/example_identification.py` | 試験データへの合わせ込み: 手で調整 → フィット → 全点で仕上げ → パラメータを振る (約 50 秒) |
| `examples/notebook_tuning.py` | 同じ調整をスライダーで行う marimo ノートブック (`uv sync --extra notebook`) |
| `examples/example_compare_models.py` | 複数モデルの比較: 応答 / 計算コスト / フィットの良さ / 非収束 |
| `examples/example_custom_model_workflow.py` | 自作モデルの定義 → 検証の通しワークフロー |
| `examples/example_fortran_crosscheck.py` | Fortran UMAT のビルド → 実行 → クロス検証 (要 gfortran) |
| `examples/example_service.py` | service 層: registry → describe → run_simulation (非収束込み) |

## ドキュメント

ガイドは `docs/` の Markdown (日本語)、API リファレンスは docstring (英語) から Sphinx + MyST で生成する。

```bash
uv sync --extra docs
make docs            # docs/_build/html/index.html を生成
```

`make docs` は警告をエラーとして扱う。次のどれかがあるとビルドが失敗する:

- 解決できない相互参照 (改名・削除された名前へのリンク)
- `docs/` 内のページへのリンク切れ
- 公開パッケージの `__all__` に載っているのに、リファレンスに項目が無い名前 (docstring の無いクラスや関数、`docs/reference/` に説明の無い定数)

docstring の中で他の名前を参照するときは、公開パッケージのパスで書く (`` :class:`~returnmap.core.MaterialModel` ``)。
