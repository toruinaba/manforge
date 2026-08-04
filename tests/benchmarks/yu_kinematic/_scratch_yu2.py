from datetime import datetime

from manforge.models.yu_kinematic import YUKinematic1D, YUKinematic3D
from manforge.simulation.integrator import PythonNumericalIntegrator, PythonAnalyticalIntegrator
from manforge.simulation.driver import MixedDriver, StrainDriver
from manforge.simulation.types import FieldHistory, FieldType

from matplotlib import pyplot as plt

model3d = YUKinematic3D(
    E=206_000, nu=0.3, Y=360.0, C_1=2000.0, C_2=200.0,
    B=435.0, Rsat=255.0, k=26.0, b=66.0,
    h=0.4, Ea=159_000, xi=61.0)
model1d = YUKinematic1D(
    E=206_000, nu=0.3, Y=360.0, C_1=2000.0, C_2=200.0,
    B=435.0, Rsat=255.0, k=26.0, b=66.0,
    h=0.4, Ea=159_000, xi=61.0)

idx = 0
int_n = PythonNumericalIntegrator(model3d)
int_a = PythonAnalyticalIntegrator(model3d)
d_n = MixedDriver(int_n, prescribed_strain_idx=[idx])
d_a = MixedDriver(int_a, prescribed_strain_idx=[idx])
int_1dn = PythonNumericalIntegrator(model1d)
d_1dn = StrainDriver(int_1dn)

hist = FieldHistory.cyclic_strain([0.05, -0.05, 0.05, -0.05], n_per_segment=50)
print(80*"=")
print("Numerical calculation")
start = datetime.now()
res_n = d_n.run(hist)
end = datetime.now()
interval = end - start
print(f"Total time: {interval}")
print(80*"=")
print("Numerical calculation(1d)")
start = datetime.now()
res_1dn = d_1dn.run(hist)
end = datetime.now()
interval = end - start
print(f"Total time: {interval}")
print(80*"=")
print("Analytical calculation")
start = datetime.now()
res_a = d_a.run(hist)
end = datetime.now()
interval = end - start
print(f"Total time: {interval}")

hist_n = [r.n_iterations for r in res_n.step_results]
hist_a = [r.n_iterations for r in res_a.step_results]
hist_1dn = [r.n_iterations for r in res_1dn.step_results]

fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.plot(res_n.strain[:, idx], res_n.stress[:, idx], "-r", label="Numerical3d")
ax1.plot(res_a.strain[:, idx], res_n.stress[:, idx], "--b", label="Analytical3d")
ax1.plot(res_1dn.strain, res_1dn.stress, ":g", label="Numerical1d")
ax2.plot(range(len(hist_n)), hist_n, "-r", label="Numerical3d")
ax2.plot(range(len(hist_a)), hist_a, "--b", label="Analytical3d")
ax2.plot(range(len(hist_1dn)), hist_1dn, ":g", label="Numerical1d")
plt.legend()
plt.show()

