.. canvas-node-item:

========================
Node Item (``nodeitem``)
========================

.. automodule:: Orange.OrangeCanvas.canvas.items.nodeitem


.. autoclass:: NodeItem
   :members:
   :exclude-members:
      from_node,
      from_node_meta,
      setupGraphics,
      setProgressMessage,
      positionChanged,
      anchorGeometryChanged,
      activated,
      hovered
   :member-order: bysource
   :show-inheritance:

   .. autoattribute:: positionChanged()

   .. autoattribute:: anchorGeometryChanged()

   .. autoattribute:: activated()


.. autoclass:: AnchorPoint
   :members:
   :exclude-members:
      scenePositionChanged,
      anchorDirectionChanged
   :member-order: bysource
   :show-inheritance:

   .. autoattribute:: scenePositionChanged(QPointF)

   .. autoattribute:: anchorDirectionChanged(QPointF)
