"""応答を計算する — 既存のモデルに載荷履歴を与えて、応力とひずみの履歴を求める。

文書の「応答を計算する」(docs/guide/simulation.md) の手順を、上から順に実行する。
文書のページは、このファイルの各節 (``# %%`` の区切り) をそのまま載せている。

    uv run python examples/example_response.py

図を描くには matplotlib が要る (``pip install "returnmap[examples]"``)。無ければ図だけ省く。
"""

try:
    import matplotlib
    matplotlib.use("Agg")                 # 画面に出さず、ファイルに保存する
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# %% 1. 最小の計算
from returnmap.models import AFKinematic1D
from returnmap.simulation import FieldHistory, PythonIntegrator, StrainDriver

model = AFKinematic1D(E=206_000.0, nu=0.3, sigma_y0=250.0, C_k=20_000.0, gamma=100.0)
load = FieldHistory.cyclic_strain([0.015, -0.015, 0.015], n_per_segment=50)   # ±1.5 % の繰返し

result = StrainDriver(PythonIntegrator(model)).run(load)

print(f"ステップ数           : {len(result.stress)}")
print(f"最初の折返しでの応力 : {result.stress[49, 0]:.1f} MPa")
print(f"最後の累積塑性ひずみ : {result.history('ep')[-1]:.4f}")

# %% 2. モデルを選ぶ
import returnmap.models as models

print([name for name in models.__all__ if name.endswith("1D")])

for name, meta in AFKinematic1D.param_meta.items():
    print(f"{name:9s} {meta.doc:33s} {meta.unit:4s} {meta.default:>9}  {meta.bounds}")

trial = AFKinematic1D.at_defaults()             # 目安の値で作ったモデル
harder = model.with_params(C_k=40_000.0)        # 一部の値だけを変えたモデル
print(harder.params)

# %% 3. 載荷履歴を作る
import numpy as np

from returnmap.simulation import FieldType

ramp = FieldHistory(FieldType.STRAIN, np.linspace(0.0, 0.02, 101)[1:])          # 配列をそのまま渡す
cycles = FieldHistory.triangular_strain(0.01, n_cycles=3, n_per_segment=25)     # 作る関数を使う
print(ramp.data.shape, cycles.data.shape)

# %% 3b. 刻みを変えて確かめる
for n_steps in (5, 50, 1000):
    pull = FieldHistory.cyclic_strain([0.015], n_per_segment=n_steps)           # 0 → 1.5 %
    stress = StrainDriver(PythonIntegrator(model)).run(pull).stress[-1, 0]
    print(f"{n_steps:5d} ステップ: {stress:.1f} MPa")

# %% 4. 結果を読む
print(result.stress.shape, result.strain.shape)                  # (ステップ数, 成分数)
print(model.state_names)                                         # 履歴を取り出せる名前
print(result.history("ep").shape, result.history("alpha").shape)

detail = result.step_results[60]                                 # 61 番目のステップの詳細
print(detail.is_plastic, detail.n_iterations, f"{detail.dlambda:.2e}")

# %% 4b. 1 ステップずつ受け取る
for step in StrainDriver(PythonIntegrator(model)).iter_run(load):
    if step.result.is_plastic:
        print(f"降伏したステップ: {step.index} (ひずみ {load.data[step.index, 0]:.4f})")
        break

# %% 4c. 図にする
if plt is not None:                                   # matplotlib が入っているとき
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(result.strain[:, 0] * 100, result.stress[:, 0])
    ax.set_xlabel("strain [%]")
    ax.set_ylabel("stress [MPa]")
    fig.tight_layout()
    fig.savefig("output_response.png", dpi=150)

# %% 5. 応力で載荷する
from returnmap.simulation import StressDriver

stress_load = FieldHistory.cyclic_stress([400.0, -400.0, 400.0], n_per_segment=40)
by_stress = StressDriver(PythonIntegrator(model)).run(stress_load)
print(f"400 MPa でのひずみ: {by_stress.strain[39, 0] * 100:.2f} %")

# %% 5b. 3D のモデルで単軸応力にする
from returnmap.models import AFKinematic3D
from returnmap.simulation import MixedDriver

model_3d = AFKinematic3D(**model.params)
pull = FieldHistory.cyclic_strain([0.015], n_per_segment=50)      # ε11 を 1.5 % まで

# ε11 だけを与え、ほかのひずみは 0 のまま: 横を拘束した単軸ひずみになる
pull_6 = FieldHistory.cyclic_strain([0.015], n_per_segment=50, ntens=model_3d.ntens)
confined = StrainDriver(PythonIntegrator(model_3d)).run(pull_6)

# ε11 を与え、ほかの成分は応力 0 にする: 引張試験と同じ単軸応力になる
uniaxial = MixedDriver(PythonIntegrator(model_3d), prescribed_strain_idx=[0]).run(pull)

print(f"単軸ひずみ: σ11 = {confined.stress[-1, 0]:7.1f} MPa, σ22 = {confined.stress[-1, 1]:7.1f} MPa")
print(f"単軸応力  : σ11 = {uniaxial.stress[-1, 0]:7.1f} MPa, σ22 = {abs(uniaxial.stress[-1, 1]):7.1f} MPa")

# %% 6. 解けなかったとき
from returnmap.models import YUKinematic1D

yu = YUKinematic1D.at_defaults()
coarse = FieldHistory.cyclic_strain([0.02, -0.02, 0.02], n_per_segment=5)      # 折返しの後は 1 ステップ 0.8 %

try:
    StrainDriver(PythonIntegrator(yu)).run(coarse)
except RuntimeError:
    print("既定では例外になる")

partial = StrainDriver(PythonIntegrator(yu, raise_on_nonconverged=False)).run(coarse)
print(partial.converged, partial.failed_step, len(partial.stress))

# %% 6b. 刻みを細かくする、増分を分割する
from returnmap.simulation import SubSteppingIntegrator

fine = FieldHistory.cyclic_strain([0.02, -0.02, 0.02], n_per_segment=20)
print(StrainDriver(PythonIntegrator(yu)).run(fine).converged)

split = StrainDriver(SubSteppingIntegrator(PythonIntegrator(yu))).run(coarse)
print(split.converged, [s.n_substeps for s in split.step_results])
