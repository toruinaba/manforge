"""試験データに合わせる — 試験の記録に対して、既存のモデルのパラメータを同定する。

文書の「試験データに合わせる」(docs/guide/tutorial.md) のコード。文書のページは、
このファイルの各節 (``# %%`` の区切り) をそのまま載せている。

    uv run python examples/example_identification.py

実行時間は約 50 秒 (Armstrong-Frederick モデルは自動微分で解くため、フィッティングの
評価 1 回が 0.3〜1 秒かかる)。作図には matplotlib が要る
(``pip install "returnmap[examples]"``)。無い場合は作図を省く。

記録 ``data/cyclic_tension_compression.csv`` は合成したもので (``data/generate.py``)、
Armstrong-Frederick モデルの応答に測定ノイズを加えてある。
"""

try:
    import matplotlib
    matplotlib.use("Agg")                 # 画面に表示せず、ファイルに保存する
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# %% 1. 試験データの読み込み
from pathlib import Path

import numpy as np

record = np.loadtxt(Path(__file__).parent / "data" / "cyclic_tension_compression.csv",
                    delimiter=",", skiprows=1)
strain_all, stress_all = record[:, 0], record[:, 1]     # ひずみ [-]、応力 [MPa]

strain, stress = strain_all[::3], stress_all[::3]       # 調整中は 3 点おきに用いる
print(f"記録 {len(strain_all)} 点、調整に用いる点 {len(strain)} 点")

# %% 2. モデルの選択
from returnmap.models import AFKinematic1D

ELASTIC = dict(E=206_000.0, nu=0.3)                     # ヤング率は記録の弾性勾配から決める
start = AFKinematic1D(**ELASTIC, sigma_y0=250.0, C_k=20_000.0, gamma=100.0)

# %% 3. 候補の比較
from returnmap.fitting import FitTarget
from returnmap.models import J2Isotropic1D
from returnmap.simulation import FieldHistory, FieldType, compare_models

load = FieldHistory(FieldType.STRAIN, strain)
target = FitTarget.stress_strain(strain, stress)        # このひずみ履歴に対する応力の記録

candidates = {
    "J2, H=5000": J2Isotropic1D(**ELASTIC, sigma_y0=300.0, H=5_000.0),
    "AF start": start,
    "AF yield up": start.with_params(sigma_y0=290.0),
    "AF yield up, C_k up": start.with_params(sigma_y0=290.0, C_k=26_000.0),
    "AF by hand": start.with_params(sigma_y0=290.0, C_k=26_000.0, gamma=135.0),
}
tuned = compare_models(candidates, load, targets=[target])
for name, run in tuned.runs.items():
    print(f"{name:22s} RMS {np.sqrt(run.fit_residual):6.1f} MPa")

# %% 4. フィッティング
from returnmap.fitting import ParameterSpace, fit_params


def space_around(model):
    """AF の 3 パラメータを、model の値を初期値として探索する空間。"""
    return ParameterSpace(
        free={                                          # 名前: (初期値, (下限, 上限))
            "sigma_y0": (model.sigma_y0, (150.0, 450.0)),
            "C_k": (model.C_k, (5_000.0, 80_000.0)),
            "gamma": (model.gamma, (20.0, 500.0)),
        },
        fixed=ELASTIC,
    )


by_hand = candidates["AF by hand"]
fit = fit_params(
    by_hand, [target], space_around(by_hand),
    nr_jacobian="broyden",                              # 自動微分で解くモデルの評価を速くする
    tol=1e-6,                                           # 最適化の収束判定
)
print(f"収束 {fit.success}、評価 {fit.n_evaluations} 回、RMS {np.sqrt(fit.residual):.1f} MPa")
print({name: round(value, 1) for name, value in fit.model_params.items()})

# %% 5. 全点での確認
load_all = FieldHistory(FieldType.STRAIN, strain_all)
target_all = FitTarget.stress_strain(strain_all, stress_all)

finalists = {"AF by hand": by_hand, "AF fitted (thinned)": fit.model}
thinned = compare_models(finalists, load, targets=[target])
full = compare_models(finalists, load_all, targets=[target_all])
for name in finalists:
    print(f"{name:20s} {len(strain)} 点: {np.sqrt(thinned.runs[name].fit_residual):5.1f} MPa"
          f"   {len(strain_all)} 点: {np.sqrt(full.runs[name].fit_residual):5.1f} MPa")

# %% 5b. 全点でのフィッティング
final = fit_params(
    fit.model, [target_all], space_around(fit.model),   # 範囲は同じ、初期値のみ変更
    nr_jacobian="broyden", tol=1e-6,
)
print(f"収束 {final.success}、評価 {final.n_evaluations} 回、RMS {np.sqrt(final.residual):.1f} MPa")
print({name: round(value, 1) for name, value in final.model_params.items()})

# %% 6. 結果の保存
fitted = compare_models({"AF fitted": final.model}, load_all).runs["AF fitted"]
np.savetxt(
    "output_af_fitted.csv",
    np.column_stack([fitted.strain[:, 0], fitted.stress[:, 0], fitted.history("ep")]),
    delimiter=",", header="strain,stress,ep", comments="",
)

# %% 7. 作図
if plt is not None:                                     # matplotlib がある場合のみ
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(strain_all * 100, stress_all, s=8, color="0.6", label="test record")
    ax.plot(tuned.runs["AF start"].strain[:, 0] * 100, tuned.runs["AF start"].stress[:, 0],
            label="AF start")
    ax.plot(fitted.strain[:, 0] * 100, fitted.stress[:, 0], label="AF fitted")
    ax.set_xlabel("strain [%]")
    ax.set_ylabel("stress [MPa]")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("output_identification.png", dpi=150)
