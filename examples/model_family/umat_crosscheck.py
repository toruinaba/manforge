"""Compare the compiled UMAT with the Python model, one step at a time.

``check_model`` does this from the declaration in ``verification.py`` and
reports one row.  This script takes the same comparison apart — the way in
when that row fails and the step that disagrees has to be looked at.

    uv run returnmap build --family mymodels      # needs gfortran
    uv run python umat_crosscheck.py
"""

# %% The UMAT as an integrator
from returnmap.simulation import PythonIntegrator
from returnmap.verification import crosscheck, standard_histories

from mymodels.voce.model import J2Voce3D
from mymodels.voce.verification import UMATS

model = J2Voce3D.at_defaults()
umat = UMATS[0].integrator(model)                   # FortranIntegrator, built from the declaration
load = standard_histories(model)["multiaxial"]

report = crosscheck(PythonIntegrator(model), umat, load)
print(report.summary())

# %% A disagreement, and where it comes from
from returnmap.verification import compare_jacobians

# A UMAT with a wrong constant, stood in for by a Python model whose
# hardening rate differs from the one the UMAT is given.
other = model.with_params(b=25.0)
wrong = crosscheck(PythonIntegrator(other), umat, load)
print(wrong.summary())

step = wrong.first_failure                          # the first step that disagrees
python, fortran = step.result_a, step.result_b      # the two raw results of that step
print(f"step {step.index}, plastic: {python.is_plastic}")
print(f"ep        {float(python.state['ep']):.4e} (Python)   {float(fortran.state['ep']):.4e} (UMAT)")
print(f"stress 11 {python.stress[0]:10.3f} (Python)   {fortran.stress[0]:10.3f} (UMAT)")
print(compare_jacobians(model, python, fortran).summary())

# %% The UMAT under a driver
from returnmap.simulation import FieldHistory, MixedDriver

pull = FieldHistory.cyclic_strain([0.01, -0.01], n_per_segment=20)
by_python = MixedDriver(PythonIntegrator(model), prescribed_strain_idx=[0]).run(pull)
by_umat = MixedDriver(umat, prescribed_strain_idx=[0]).run(pull)
print(f"{abs(by_umat.stress[:, 0] - by_python.stress[:, 0]).max():.1e}")
