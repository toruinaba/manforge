# サービスに組み込む

WebAPI やジョブワーカーへの組み込み用に、`returnmap.service` が **Web 依存ゼロの契約層**を提供する。入出力は plain な dict / ndarray / dataclass のみで、pydantic / HTTP 変換はデプロイ側の仕事。

```python
import numpy as np
from returnmap.service import ModelRegistry, default_registry, run_simulation

# レジストリ: 名前 → モデルクラスの明示対応 (シングルトン無し —
# デプロイごとに作り、公開するモデルだけを登録して監査可能にする)
reg = default_registry()               # 参照 1D モデル入り
reg.names()                            # ['af_kinematic', 'j2_isotropic', 'ow_kinematic']

# describe() = GET /models 相当。param_meta (単位・説明・デフォルト・範囲)
# から WebUI の入力フォームを自動生成できる
d = reg.describe("j2_isotropic")
d.param_meta["E"]                      # ParamMeta(doc="Young's modulus", unit="MPa", default=210000.0, ...)

# 実行: 生のリストを受け取り plain な配列を返す
out = run_simulation(
    reg.get("j2_isotropic"),
    params={"E": 206e3, "nu": 0.3, "sigma_y0": 250.0, "H": 1000.0},
    load=np.linspace(0, 5e-3, 1000),   # ひずみ履歴 (control="stress" で応力制御)
    include_fields=["ep"],             # 状態量は選択制 (デフォルト応力のみ)
    nr_jacobian="broyden",             # 既定は全層 "exact"。速度を優先するなら明示する
    progress=lambda done, total: ...,  # ジョブ進捗の報告に
)
out.stress, out.strain, out.fields["ep"]
```

**非収束は値で返る** — パラメータ探索中のユーザーは解の無い値を普通に入力するため、`run_simulation` は途中で解けなくなっても raise せず、そこまでの部分履歴 + `converged=False` / `failed_step` を返す (API 層は 200 で返し、UI が「ステップ k で非収束」と部分曲線を描ける)。一方、パラメータ名の過不足や未知の `include_fields` は**呼び出し側のバグ**なので `ValueError` (API 層で 400)。

**非公開モデルの登録規約** — 非公開モデルパッケージは `MODELS: dict[str, type[MaterialModel]]` を公開し、デプロイ側がそれを import して自分のレジストリに登録する:

```python
from returnmap_internal import MODELS      # 非公開パッケージ側
for name, cls in MODELS.items():
    reg.register(name, cls)
```

モデルの `param_meta` はモデルクラス自身に宣言する (`ParamMeta`) — メタ情報がリポジトリ間で乖離しないため。実行例: `examples/example_service.py`
