

from PyQt4.QtGui import QGraphicsItem, QGraphicsTextItem, QGraphicsRectItem, QGraphicsObject, QColor, QPen
from PyQt4.QtCore import QPointF, QRectF, Qt, QPropertyAnimation

from owpoint import *
from owcurve import *

PointColor = 1
PointSize = 2
PointSymbol = 4

class OWLegendItem(QGraphicsObject):
    def __init__(self, curve, parent):
        QGraphicsObject.__init__(self, parent)
        self.text_item = QGraphicsTextItem(curve.name, self)
        s = curve.pointSize()
        height = max(2*s, self.text_item.boundingRect().height())
        p = 0.5 * height
        self.text_item.setPos(height, 0)
        self.point_item = curve.pointItem(p, p, s, self)
        self._rect = QRectF(0, 0, height + self.text_item.boundingRect().width(), height )
        self.rect_item = QGraphicsRectItem(self._rect, self)
        self.rect_item.setPen(QPen(Qt.NoPen))
        
    def boundingRect(self):
        return self._rect
        
    def paint(self, painter, option, widget):
        pass
        
class OWLegend(QGraphicsItem):
    def __init__(self, graph, scene):
        QGraphicsItem.__init__(self, None, scene)
        self.graph = graph
        self.curves = []
        self.items = []
        self.attributes = []
        self.point_attrs = {}
        self.point_vals = {}
        self.default_values = {
                               PointColor : Qt.black, 
                               PointSize : 8, 
                               PointSymbol : OWPoint.Ellipse
                               }
        self.box_rect = QRectF()
        self.setFiltersChildEvents(True)
        self.setFlag(self.ItemHasNoContents, True)
        self.mouse_down = False
        self._orientation = Qt.Vertical
        self.animated = True
        
    def clear(self):
        self.curves = []
        self.point_attrs = {}
        self.point_vals = {}
        self.update()
        

    def add_curve(self, curve, attributes = []):
        self.items.append(OWLegendItem(curve, self))
        self.update()
        
    def update(self):
        self._animations = []
        self.box_rect = QRectF()
        if self._orientation == Qt.Vertical:
            y = 0
            for item in self.items:
                self.move_item(item, 0, y)
                y = y + item.boundingRect().height()
                self.box_rect = self.box_rect | item.mapRectToParent(item.boundingRect())
        elif self._orientation == Qt.Horizontal:
            x = 0
            for item in self.items:
                self.move_item(item, x, 0)
                x = x + item.boundingRect().width()
                self.box_rect = self.box_rect | item.mapRectToParent(item.boundingRect())
                
    def mouseMoveEvent(self, event):
        self.setPos(self.pos() + event.scenePos() - event.lastScenePos())
        self.graph.notify_legend_moved()
        event.accept()
            
    def mousePressEvent(self, event):
        self.setCursor(Qt.DragMoveCursor)
        self.mouse_down = True
        event.accept()
        
    def mouseReleaseEvent(self, event):
        self.unsetCursor()
        self.mouse_down = False
        event.accept()

    def boundingRect(self):
        return self.box_rect
        
    def paint(self, painter, option, widget=None):
        pass
    
    def set_orientation(self, orientation):
        self._orientation = orientation
        self.update()
        
    def move_item(self, item, x, y):
        if self.animated:
            a = QPropertyAnimation(item, 'pos')
            a.setStartValue(item.pos())
            a.setEndValue(QPointF(x,y))
            a.start(QPropertyAnimation.DeleteWhenStopped)
            self._animations.append(a)
        else:
            item.setPos(x, y)
