# returnmap.fitting

```{eval-rst}
.. automodule:: returnmap.fitting

.. rubric:: Constants

.. py:data:: FAILURE_PENALTY
   :value: 1e30

   Objective value assigned to an evaluation the model could not complete.
   Large enough that no real fit reaches it, but deliberately finite:
   returning ``inf`` breaks L-BFGS-B, whose finite-difference gradient next to
   an ``inf`` point evaluates to ``nan`` and poisons the line search.
```
