# API リファレンス

公開の import 元は次の 7 つのパッケージです。どの名前も 1 つのパッケージにだけ属します。
リファレンスの本文は、ソースコードの docstring (英語) から生成しています。

| パッケージ | 中身 |
|---|---|
| [`manforge.core`](core.md) | モデルを書く: `MaterialModel`、状態変数の宣言、次元、`hand_block`、`verified_against_fortran`、結果型 |
| [`manforge.models`](models.md) | 参照モデル (J2 / 損傷 / AF / OW / YU) |
| [`manforge.simulation`](simulation.md) | integrator、driver、載荷履歴、`FortranModule` |
| [`manforge.fitting`](fitting.md) | `fit_params`、`FitTarget`、`ParameterSpace` |
| [`manforge.verification`](verification.md) | 検査: `JacobianChecker`、`TangentChecker`、`check_bindings`、Crosscheck |
| [`manforge.service`](service.md) | WebAPI 向けの契約層 |
| [`manforge.utils`](utils.md) | モデル用の補助関数 (smooth 近似、主値分解) |

その下のサブモジュール (`manforge.core.state`、`manforge.simulation.driver` など) はコードの置き場所で、
文書化されたインターフェースではありません。

```{toctree}
:hidden:
:maxdepth: 1

core
models
simulation
fitting
verification
service
utils
```
