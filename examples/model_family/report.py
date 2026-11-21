"""Print the report of the common checks for every model of the family.

The tests assert that the report passes; this prints it, row by row — what
was checked, how far apart the two sides were, and what was not declared.

    uv run python report.py
"""

# %% The report of every model
from returnmap.verification import check_model

from mymodels.voce.verification import MODELS

reports = {type(model).__name__: check_model(model) for model in MODELS}
for report in reports.values():
    print(report.summary())

# %% One row in detail
row = reports["J2Voce3D"]["hand-written tangent"]
print(row.status, "—", row.detail)
print(sorted(row.reports))                          # one report per step that was checked
print(row.reports["multiaxial[1]"].summary())
