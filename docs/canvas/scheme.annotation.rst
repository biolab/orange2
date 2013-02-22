.. schemeannotation:

====================================
Scheme Annotations (``annotations``)
====================================

.. automodule:: Orange.OrangeCanvas.scheme.annotations


.. autoclass:: BaseSchemeAnnotation
   :members:
   :member-order: bysource
   :show-inheritance:

   .. autoattribute:: geometry_changed()

      Signal emitted when the geometry of the annotation changes


.. autoclass:: SchemeArrowAnnotation
   :members:
   :member-order: bysource
   :show-inheritance:


.. autoclass:: SchemeTextAnnotation
   :members:
   :member-order: bysource
   :show-inheritance:

   .. autoattribute:: text_changed(str)

      Signal emitted when the annotation text changes.
