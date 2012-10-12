"""
Canvas Graphics View
"""

from PyQt4.QtGui import QGraphicsView
from PyQt4.QtCore import Qt, QPointF, QSizeF, QRectF


class CanvasView(QGraphicsView):
    """Canvas View handles the zooming and panning.
    """

    def __init__(self, *args):
        QGraphicsView.__init__(self, *args)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)

    def push_zoom_rect(self, rect):
        """Zoom into the rect.
        """
        raise NotImplementedError

    def pop_zoom(self):
        """Pop a zoom level.
        """
        raise NotImplementedError

    def setScene(self, scene):
        QGraphicsView.setScene(self, scene)
        self._ensureSceneRect(scene)

    def _ensureSceneRect(self, scene):
        r = scene.addRect(QRectF(0, 0, 400, 400))
        scene.sceneRect()
        scene.removeItem(r)
