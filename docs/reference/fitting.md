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

.. py:data:: SLOW_EVALUATION_SECONDS
   :value: 0.5

   Time for one objective evaluation above which
   :func:`~returnmap.fitting.fit_params` issues a
   :class:`~returnmap.fitting.SlowFitWarning`.  A fit evaluates the objective
   a few hundred times, so half a second each is where the wait turns from
   seconds into minutes.
```
