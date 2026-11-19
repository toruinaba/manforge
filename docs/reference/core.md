# returnmap.core

```{eval-rst}
.. automodule:: returnmap.core

.. rubric:: Constants

.. py:data:: NTENS

   Shape of a tensor-valued state field: ``(model.ntens,)``, resolved from the
   model's dimension.  Use as ``Implicit(shape=NTENS)`` / ``Explicit(shape=NTENS)``.

.. py:data:: SCALAR

   Shape of a scalar state field: ``()``.

.. py:data:: SOLID_3D

   3D solid, 6 components ``[11, 22, 33, 12, 13, 23]`` (ABAQUS UMAT order).
   The default dimension of a model that does not fix one.

.. py:data:: PLANE_STRAIN

   Plane strain, 4 components ``[11, 22, 33, 12]``.

.. py:data:: PLANE_STRESS

   Plane stress, 3 components ``[11, 22, 12]``.  A stored stress-like vector
   is read as part of a 3D deviator: its 33 component is reconstructed as
   ``-(s11 + s22)``.

.. py:data:: PLANE_STRESS_P

   Plane stress, 3 components ``[11, 22, 12]``, storing the in-plane tensor
   itself with the 33 component identically zero.  Deviatoric operations go
   through the metric :data:`~returnmap.core.P_PLANE_STRESS`.  The two
   plane-stress conventions disagree on yield functions that carry a
   backstress; this is the one the YU models use.

.. py:data:: UNIAXIAL_1D

   Uniaxial stress, 1 component ``[11]``.

.. py:data:: P_PLANE_STRESS

   3×3 plane-stress deviatoric metric: ``sᵀ P t = dev₃D(s) : dev₃D(t)`` for
   σ33 ≡ 0.  Positive-definite — a metric, not a projector.  It absorbs both
   the deviatoric projection and the Mandel shear weighting, and ``P @ ξ``
   comes out in the engineering-shear convention.
```
