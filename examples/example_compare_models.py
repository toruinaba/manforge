"""モデルを比べる — 複数のモデル、パラメータ、解法を同じ載荷履歴で計算して並べる。

文書の「モデルを比べる」(docs/guide/comparing.md) のコード。文書のページは、
このファイルの各節 (``# %%`` の区切り) をそのまま載せている。

    uv run python examples/example_compare_models.py

実行時間は約 15 秒。作図には matplotlib が要る (``pip install "returnmap[examples]"``)。
無い場合は作図を省く。
"""

try:
    import matplotlib
    matplotlib.use("Agg")                 # 画面に表示せず、ファイルに保存する
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# %% 1. 応答と計算コスト
from returnmap.models import AFKinematic1D, J2Isotropic1D, OWKinematic1D
from returnmap.simulation import FieldHistory, compare_models

ELASTIC = dict(E=206_000.0, nu=0.3)
j2 = J2Isotropic1D(**ELASTIC, sigma_y0=250.0, H=2_000.0)
af = AFKinematic1D(**ELASTIC, sigma_y0=250.0, C_k=20_000.0, gamma=100.0)
ow = OWKinematic1D(**ELASTIC, sigma_y0=250.0, C_k=20_000.0, gamma=100.0)

load = FieldHistory.cyclic_strain([0.015, -0.015, 0.015], n_per_segment=40)

comparison = compare_models({"J2": j2, "AF": af, "OW": ow}, load)        # 名前 → モデル
print(comparison.summary())

# %% 2. 応答の参照
for name, run in comparison.runs.items():
    print(f"{name}: 最大応力 {run.stress[:, 0].max():.1f} MPa")

print(comparison.runs["AF"].history("alpha").shape)       # 状態量の履歴
print(type(comparison.runs["AF"].result).__name__)        # driver の結果そのもの

# %% 3. 非収束のモデル
absurd = OWKinematic1D(**ELASTIC, sigma_y0=250.0, C_k=1e9, gamma=1e12)

partial = compare_models({"AF": af, "OW (absurd recovery)": absurd}, load)
print(partial.summary())

stopped = partial.runs["OW (absurd recovery)"]
print(partial.converged, stopped.converged, stopped.failed_step, len(stopped.stress))

# %% 4. フィットの良さ
import numpy as np

from returnmap.fitting import FitTarget, ParameterSpace, fit_params

# 記録の代わり: AF の応答に標準偏差 3 MPa のノイズを加えたもの
rng = np.random.default_rng(0)
stress_exp = comparison.runs["AF"].stress[:, 0] + rng.normal(0.0, 3.0, len(load.data))
target = FitTarget.stress_strain(load.data, stress_exp)

fit = fit_params(
    j2, [target],
    ParameterSpace(
        free={"sigma_y0": (250.0, (100.0, 500.0)), "H": (2_000.0, (0.0, 50_000.0))},
        fixed=ELASTIC,
    ),
)
scored = compare_models(
    {"J2": j2, "J2 (fitted)": fit.model, "AF": af, "OW": ow}, load, targets=[target],
)
print(scored.summary())
print(np.isclose(scored.runs["J2 (fitted)"].fit_residual, fit.residual))

# %% 5. パラメータスタディ
study = compare_models(
    {f"gamma={g:g}": af.with_params(gamma=g) for g in (50.0, 100.0, 200.0, 400.0)},
    load, targets=[target],
)
for name, run in study.runs.items():
    print(f"{name:10s} 最大応力 {run.stress[:, 0].max():6.1f} MPa   RMS {np.sqrt(run.fit_residual):5.1f} MPa")

# %% 6. 解法の比較
from returnmap.models import AFKinematic3D
from returnmap.simulation import MixedDriver, PythonIntegrator, StrainDriver

af_3d = AFKinematic3D(**af.params)


def uniaxial_stress(integrator):
    """次元によらず単軸応力にする driver。"""
    if integrator.ntens == 1:
        return StrainDriver(integrator)
    return MixedDriver(integrator, prescribed_strain_idx=[0])


solvers = compare_models(
    {
        "AF 1D": af,
        "AF 3D, exact Jacobian": PythonIntegrator(af_3d),
        "AF 3D, Broyden": PythonIntegrator(af_3d, nr_jacobian="broyden"),
    },
    load,
    driver_factory=uniaxial_stress,
)
print(solvers.summary())

# %% 6b. 解き直しの有無
from returnmap.models import YUKinematic1D

yu = YUKinematic1D.at_defaults()
coarse = FieldHistory.cyclic_strain([0.02, -0.02, 0.02], n_per_segment=3)   # 1 ステップ 0.67 % 以上

restart = compare_models(
    {
        "no restart": PythonIntegrator(yu, nr_restart=False, raise_on_nonconverged=False),
        "restart (default)": yu,
    },
    coarse,
)
print(restart.summary())

# %% 7. 作図
if plt is not None:                                       # matplotlib がある場合のみ
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(load.data[:, 0] * 100, stress_exp, s=8, color="0.6", label="record (AF + noise)")
    for name in ("J2 (fitted)", "AF", "OW"):
        run = scored.runs[name]
        ax.plot(run.strain[:, 0] * 100, run.stress[:, 0], label=name)
    ax.set_xlabel("strain [%]")
    ax.set_ylabel("stress [MPa]")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("output_compare_models.png", dpi=150)
