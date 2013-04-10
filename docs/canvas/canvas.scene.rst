.. canvas-scene:

========================
Canvas Scene (``scene``)
========================

.. automodule:: Orange.OrangeCanvas.canvas.scene


.. autoclass:: CanvasScene
   :members:
   :exclude-members:
      node_item_added,
      node_item_removed,
      link_item_added,
      link_item_removed,
      annotation_added,
      annotation_removed,
      node_item_position_changed,
      node_item_double_clicked,
      node_item_activated,
      node_item_hovered,
      link_item_hovered
   :member-order: bysource
   :show-inheritance:

   .. autoattribute:: node_item_added(NodeItem)

   .. autoattribute:: node_item_removed(NodeItem)

   .. autoattribute:: link_item_added(LinkItem)

   .. autoattribute:: link_item_removed(LinkItem)

   .. autoattribute:: annotation_added(Annotation)

   .. autoattribute:: annotation_removed(Annotation)

   .. autoattribute:: node_item_position_changed(NodeItem, QPointF)

   .. autoattribute:: node_item_double_clicked(NodeItem)

   .. autoattribute:: node_item_activated(NodeItem)

   .. autoattribute:: node_item_hovered(NodeItem)

   .. autoattribute:: link_item_hovered(LinkItem)


.. autofunction:: grab_svg