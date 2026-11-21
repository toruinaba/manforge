# 新しいモデルを足す

新しい構成則を、Python のモデルから検証済みの UMAT まで進める手順の概要を示す。各段階の手順は、次の 3 ページにある。

| 段階 | ページ | 書くもの | 実行される検査 |
|---|---|---|---|
| A | [モデルを書く](model-python.md) | `yield_function` と、状態の発展則 (`update_state` / `state_residual`) | 収束すること、接線が差分と一致すること |
| B | [手書きの解法を照合する](model-hand-written.md) | `user_defined_return_mapping`、`user_defined_tangent` (必要なら `@hand_block`) | 手書きの解が残差を満たすこと、自動微分と同じ履歴をたどること、手書きの接線が差分と一致すること |
| C | [UMAT を確かめる](model-umat.md) | Fortran のソースと `Umat` の宣言 | UMAT が Python のモデルと全ステップで一致すること |

例は J2 + Voce 硬化 (飽和型の等方硬化) であり、完成形は [`examples/model_family/`](https://github.com/toruinaba/returnmap/tree/main/examples/model_family) にある。このディレクトリは、取り出せば独立したリポジトリの雛形になる。

## 段階の位置づけ

モデルは、どの段階で止めてもよい。実行される検査は、モデルが宣言しているものから決まる。

- **段階 A で完結する。** リターンマッピングと整合接線は自動微分で解かれる。応答の計算、モデルの比較、パラメータの同定は、この段階で行える。
- **段階 B は UMAT の下書きである。** 手で導いた代数を、自動微分を基準として検査する。副次的に計算が速くなる (この例では 1 ステップ 2.6 ms → 0.03 ms)。
- **段階 C は段階 B の移植である。** UMAT の比較相手は手書きの経路であり、同じ代数を 2 つの言語で書いたものの一致を検査する。

## リポジトリの構成

1 つのモデルファミリを 1 つのディレクトリにまとめる。モデル本体、検証の宣言、UMAT のソース、テストを同じ場所に置く。

```text
mymodels/                          ← リポジトリ
  pyproject.toml                   returnmap をリリースのタグに固定する
  report.py                        検査の報告を表示する
  umat_crosscheck.py               UMAT との比較を個別に実行する (段階 C)
  src/mymodels/
    voce/                          ← 1 ファミリ
      __init__.py                  公開するクラス
      README.md                    定式化 (式、状態変数、パラメータ)
      model.py                     モデル本体
      verification.py              検証の宣言 (MODELS、UMATS)
      fortran/voce_3d.f90          UMAT のソース (段階 C まで進むモデルのみ)
      tests/test_verification.py   共通の検査を実行する
  fortran/                         コンパイルした UMAT の置き場 (ビルド時に作られる)
```

returnmap は PyPI に無いため、`pyproject.toml` でタグに固定する。

```toml
[project]
dependencies = ["returnmap>=0.3.2"]

[project.optional-dependencies]
dev = ["pytest"]
fortran = ["meson", "ninja"]       # UMAT をビルドする場合

[tool.uv.sources]
returnmap = { git = "https://github.com/toruinaba/returnmap", tag = "v0.3.2" }

[tool.pytest.ini_options]
testpaths = ["src/mymodels"]
addopts = "--strict-markers --import-mode=importlib"
```

| 項目 | 内容 |
|---|---|
| pytest の設定 | `fortran` / `slow` マーカーの登録と、ビルド先 (`fortran/`) を import 可能にする設定は、returnmap が配布する pytest プラグインが行う。`conftest.py` への記述は不要である |
| 例の `conftest.py` | returnmap をインストールせず、checkout から使う場合のためのもの |
| テストの import | 公開の import パス (`returnmap.core` など) と、そのファミリの中のものに限る。ディレクトリごと別のリポジトリへ移せるようにするためである |

## ファミリの範囲

ファミリは、1 つのモデルと、それを継承した変種である。

| 対象 | 置き方 |
|---|---|
| 1 つのモデル | 親クラス 1 つ (物理) と、その次元違いのクラス (`3D` / `PS` / `1D`) |
| 継承した変種 | 同じディレクトリに入れる。元のモデルに項を足したもの、段階を進めたものが該当する (本体の `j2_damage/` は、損傷・crack closure・準片側の 3 段階を持つ) |
| 互いに独立なモデル | 系統が同じでも別のディレクトリにし、名前の先頭をそろえる (`yu_proj/`、`yu_continuous/` など)。同じディレクトリに入れると、テストと UMAT のソースが混在し、片方だけを移せなくなる |
| 別のファミリの継承 | モデル本体は、継承元のファミリを import してよい。テストは、自分のファミリと公開の import パスのみを用いる |

目安は `verification.py` である。そこに並ぶモデルが、1 つの定式化のページ (`README.md`) で説明できる範囲に収まっていればよい。

## 完了の条件

| 段階 | 条件 |
|---|---|
| A | `check_model` の `converges` と `tangent` が `ok` |
| B | `hand-written solution`、`hand-written vs autodiff`、`hand-written tangent` が `ok`。近似は宣言に書いてある |
| C | ビルドした環境で `UMAT` の行が `ok`。UMAT の収束フラグを `parse_umat_failure` で読ませている |

## 固有のテスト

共通の検査で不足するもの (特定のパラメータでの回帰、文献値との比較など) は、ファミリの `tests/` に追加する。ファイル名の規約は次のとおり。

| 名前 | 内容 |
|---|---|
| `test_verification.py` | 共通の検査のみ。固有のテストは書かない |
| `test_model.py` | モデルの関数ごとの確認 (integrator を用いない) |
| `test_<確かめること>.py` | そのモデルに固有の確認 |

共通の検査に合格しないモデルは、既知の理由を付けて `pytest.mark.xfail(strict=True, reason=...)` とする。修正されたときに検出できる。

## returnmap 本体への移行

ファミリのディレクトリを `src/returnmap/models/<family>/` に移す。テストが公開の import パスのみを用いていれば、変更は次の箇所に限られる。

1. ファミリの中の import を `returnmap.models.<family>` に変更する。
2. `returnmap/models/__init__.py` の `__all__` にクラスを追加する。
3. 全パラメータに `param_meta` を書く (組み込みモデルは、目安の値で計算が通ることをテストで確認している)。
4. ファミリの `README.md` に定式化を書き、`docs/guide/models/<family>.md` から取り込む。
5. [組み込みモデルと次元選択](models.md) の表と `CHANGELOG.md` に追加する。

## 関連ページ

- モデルに書けるフックの一覧: [モデルを作る](writing-models.md)
- 各検査の測定内容: [検証ワークフロー](verification.md)
