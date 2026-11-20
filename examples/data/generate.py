"""Generate the stand-in test record used by the tutorial and the tuning notebook.

``cyclic_tension_compression.csv`` is **synthetic**: the response of an
Armstrong-Frederick model to one and a half cycles at ±1.5 % strain, with
measurement noise added (3 MPa on the stress, 2e-6 on the strain).  It stands
in for the CSV a testing machine exports, so the examples run without a real
test.  The parameters it was generated with are below; a fit of the AF model
should come back to them.

Usage
-----
    uv run python examples/data/generate.py
"""

from pathlib import Path

import numpy as np

from returnmap.models import AFKinematic1D
from returnmap.simulation import FieldHistory, PythonIntegrator, StrainDriver

TRUE = dict(E=206_000.0, nu=0.3, sigma_y0=285.0, C_k=26_000.0, gamma=140.0)

load = FieldHistory.cyclic_strain([0.015, -0.015, 0.015], n_per_segment=100)
result = StrainDriver(PythonIntegrator(AFKinematic1D(**TRUE))).run(load)

rng = np.random.default_rng(1)
strain = result.strain[:, 0] + rng.normal(0.0, 2e-6, len(result.strain))
stress = result.stress[:, 0] + rng.normal(0.0, 3.0, len(result.strain))

out = Path(__file__).with_name("cyclic_tension_compression.csv")
np.savetxt(out, np.column_stack([strain, stress]), delimiter=",", fmt="%.6e",
           header="strain,stress_MPa", comments="")
print(f"{len(strain)} rows → {out}")
