

from PyQt4.QtGui import QGraphicsTextItem, QGraphicsRectItem, QGraphicsObject, QColor, QPen
from PyQt4.QtCore import QPointF, QRectF, Qt, QPropertyAnimation, QSizeF, qDebug

from owpoint import *
from owcurve import OWCurve
from owtools import move_item, move_item_xy

PointColor = 1
PointSize = 2
PointSymbol = 4

class OWLegendItem(QGraphicsObject):
    def __init__(self, name, point, parent):
        QGraphicsObject.__init__(self, parent)
        self.text_item = QGraphicsTextItem(name, self)
        if point:
            s = point.size()
            height = max(2*s, self.text_item.boundingRect().height())
        else:
            height = self.text_item.boundingRect().height()
        p = 0.5 * height
        self.text_item.setPos(height, 0)
        self.point_item = point
        if point:
            self.point_item.setParentItem(self)
            self.point_item.setPos(p, p)
        self._rect = QRectF(0, 0, height + self.text_item.boundingRect().width(), height )
        self.rect_item = QGraphicsRectItem(self._rect, self)
        self.rect_item.setPen(QPen(Qt.NoPen))
        self.rect_item.setBrush(Qt.white)
        self.rect_item.stackBefore(self.text_item)
        if self.point_item:
            self.rect_item.stackBefore(self.point_item)
        
    def boundingRect(self):
        return self._rect
        
    def paint(self, painter, option, widget):
        pass
    
class OWLegendTitle(QGraphicsObject):
    def __init__(self, text, parent):
        QGraphicsObject.__init__(self, parent)
        self.text_item = QGraphicsTextItem(text + ':', self)
        f = self.text_item.font()
        f.setBold(True)
        self.text_item.setFont(f)
        
    def boundingRect(self):
        return self.text_item.boundingRect()
        
    def paint(self, painter, option, widget):
        pass
        
class OWLegend(QGraphicsObject):
    def __init__(self, graph, scene):
        QGraphicsObject.__init__(self)
        if scene:
            scene.addItem(self)
        self.graph = graph
        self.curves = []
        self.items = {}
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
        self.max_size = QSizeF()
        self._floating = True
        self._floating_animation = None

    def clear(self):
        for lst in self.items.itervalues():
            for i in lst:
                i.setParentItem(None)
                self.scene().removeItem(i)
        self.items = {}
        self.update()
        

    def add_curve(self, curve, attributes = []):
        i = curve.name.find('=')
        if i == -1:
            cat = ''
            name = curve.name
        else:
            cat = curve.name[:i]
            name = curve.name[i+1:]
        self.add_item('', curve.name, curve.point_item(0, 0, 0))
        
    def add_item(self, category, value, point):
        if category not in self.items:
            self.items[category] = [OWLegendTitle(category, self)]
        self.items[category].append(OWLegendItem(str(value), point, self))
        self.update()
        
    def remove_category(self, category):
        if category not in self.items:
            return
        for item in self.items[category]:
            self.scene().removeItem(item)
        del self.items[category]
        
    def update(self):
        self.box_rect = QRectF()
        x, y = 0, 0
        
        if self._orientation == Qt.Vertical:
            for lst in self.items.itervalues():
                for item in lst:
                    if self.max_size.height() and y and y + item.boundingRect().height() > self.max_size.height():
                        y = 0
                        x = x + item.boundingRect().width()
                    self.box_rect = self.box_rect | item.boundingRect().translated(0, y)
                    move_item_xy(item, x, y)
                    y = y + item.boundingRect().height()
        elif self._orientation == Qt.Horizontal:
            for lst in self.items.itervalues():
                for item in lst:
                    if self.max_size.width() and x and x + item.boundingRect().width() > self.max_size.width():
                        x = 0
                        y = y + item.boundingRect().height()
                    self.box_rect = self.box_rect | item.boundingRect().translated(x, y)
                    move_item_xy(item, x, y)
                    x = x + item.boundingRect().width()
        else:
            qDebug('A bad orientation of the legend')
    
    def mouseMoveEvent(self, event):
        self.graph.notify_legend_moved(event.scenePos())
        if self._floating:
            p = event.scenePos() - self._mouse_down_pos
            if self._floating_animation and self._floating_animation.state() == QPropertyAnimation.Running:
                self.set_pos_animated(p)
            else:
                self.setPos(p)
        event.accept()
            
    def mousePressEvent(self, event):
        self.setCursor(Qt.DragMoveCursor)
        self.mouse_down = True
        self._mouse_down_pos = event.scenePos() - self.pos()
        event.accept()
        
    def mouseReleaseEvent(self, event):
        self.unsetCursor()
        self.mouse_down = False
        self._mouse_down_pos = QPointF()
        event.accept()

    def boundingRect(self):
        return self.box_rect
        
    def paint(self, painter, option, widget=None):
        pass
    
    def set_orientation(self, orientation):
        if self._orientation != orientation:
            self._orientation = orientation
            self.update()
            
    def set_pos_animated(self, pos):
        if (self.pos() - pos).manhattanLength() < 6 or not self.graph.use_animations:
            self.setPos(pos)
        else:
            t = 250
            if self._floating_animation and self._floating_animation.state() == QPropertyAnimation.Running:
                t = t - self._floating_animation.currentTime()
                self._floating_animation.stop()
            self._floating_animation = QPropertyAnimation(self, 'pos')
            self._floating_animation.setStartValue(self.pos())
            self._floating_animation.setEndValue(pos)
            self._floating_animation.setDuration(t)
            self._floating_animation.start(QPropertyAnimation.KeepWhenStopped)
        
    def set_floating(self, floating, pos=None):
        if floating == self._floating:
            return
        self._floating = floating
        if pos:
            if floating:
                self.set_pos_animated(pos - self._mouse_down_pos)
            else:
                self.set_pos_animated(pos)