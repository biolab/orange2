.. canvas-node-item:

========================
Node Item (``nodeitem``)
========================

.. automodule:: Orange.OrangeCanvas.canvas.items.nodeitem


.. autoclass:: NodeItem
   :members:
   :exclude-members: from_node, from_node_meta, setupGraphics,  setProgressMessage
   :member-order: bysource
   :show-inheritance:

   .. autoattribute:: positionChanged()

      Signal emitted when the scene position of the node has changes.

   .. autoattribute:: anchorGeometryChanged()

      Signal emitted when the geometry of the channel anchors changes.

   .. autoattribute:: activated()

      Signal emitted when the item has been activated (by a mouse double
      click or a keyboard)
