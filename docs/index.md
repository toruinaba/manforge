# returnmap

```{include} ../README.md
:start-after: <!-- docs:start -->
:end-before: <!-- docs:end -->
```

## どこから読むか

| したいこと | 読むページ |
|---|---|
| 試験データに既存のモデルを合わせる | [チュートリアル](guide/tutorial.md) |
| どんなモデルがあるか、式を知る | [組み込みモデルと次元選択](guide/models.md) |
| 載荷履歴を与えて応答を計算する | [シミュレーション実行](guide/simulation.md) |
| モデルやパラメータを比べる | [モデルを比べる](guide/comparing.md) |
| 自分のモデルを書いて、UMAT まで進める | [新しいモデルを足す](guide/adding-a-model.md) |
| 仕組みを知る (規約、非収束の扱い) | [概念](concepts.md) |

```{toctree}
:caption: 既存のモデルを使う
:maxdepth: 1

guide/tutorial
guide/models
guide/simulation
guide/fitting
guide/comparing
```

```{toctree}
:caption: 自分のモデルを UMAT にする
:maxdepth: 1

guide/adding-a-model
guide/writing-models
guide/verification
guide/fortran
```

```{toctree}
:caption: 仕組みと組み込み
:maxdepth: 1

concepts
guide/service
```

```{toctree}
:caption: リファレンス
:maxdepth: 1

reference/index
```

```{toctree}
:caption: 開発
:maxdepth: 1

development
changelog
design/api-redesign
```
