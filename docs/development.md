# 開発

## テスト

```bash
make test        # 高速テスト (unit + integration、slow/fortran 除く)
make test-all    # 全スイート
```

その他のターゲット (e2e / slow / benchmarks / docker) は `Makefile` を参照。テストレイアウトは `tests/{unit,integration}` が `src/returnmap` のモジュール構成に対応し (モジュールごとに unit か integration のどちらか)、`tests/benchmarks/<model>/` が数値同値ハーネス。

## 例

| スクリプト | 内容 |
|-----------|------|
| `examples/example_j2_uniaxial.py` | 繰返し単軸 + tangent 検証 + ヒステリシスのプロット |
| `examples/example_j2_damage.py` | J2 + Lemaitre 損傷: 損傷の進展と剛性低下 |
| `examples/example_j2_damage_crack_closure.py` | crack closure 係数 h による引張/圧縮の違い |
| `examples/example_j2_damage_unilateral.py` | 準片側モデル: 圧縮での剛性回復 |
| `examples/example_fit_j2.py` | パラメータフィッティング (基本 / ParameterSpace / 複数ターゲット / PCM) |
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
