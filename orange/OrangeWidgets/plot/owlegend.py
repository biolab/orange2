

from PyQt4.QtGui import QGraphicsItem, QGraphicsTextItem, QGraphicsRectItem, QColor
from PyQt4.QtCore import QPointF, QRectF, Qt

from owpoint import *
from owcurve import *

PointColor = 1
PointSize = 2
PointSymbol = 4

class OWLegend(QGraphicsItem):
    def __init__(self, scene):
        QGraphicsItemGroup.__init__(self, None, scene)
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
        
    def clear(self):
        self.curves = []
        self.point_attrs = {}
        self.point_vals = {}
        self.update()
        

    def add_curve(self, curve, attributes = []):
        for point_attribute, data_attribute, value in attributes:
            if point_attribute not in self.point_attrs:
                self.point_attrs[point_attribute] = data_attribute
                
            if point_attribute == PointColor:
                point_val = curve.color()
            elif point_attribute == PointSize:
                point_val = curve.pointSize()
            else:
                point_val = curve.symbol()

            if not point_attribute in self.point_vals:
                self.point_vals[point_attribute] = {}
            self.point_vals[point_attribute][point_val] = value
        self.curves.append(curve)
        self.update()
        
    def update(self):
        for i in self.items:
            self.scene().removeItem(i)
        del self.items[:]
        y = 10
        length = 0
        if self.point_attrs:
            ## Using the owplot API to specify paremeters
            ## NOTE: The API is neither finished nor used
            for p_a, d_a in self.point_attrs.iteritems():
                ## We construct a separate box for each attribute 
                title_item = QGraphicsTextItem( d_a, self )
                title_item.setPos(QPointF(10, y-10))
                self.items.append(title_item)
                y = y + 20
                for p_v, d_v in self.point_vals[p_a].iteritems():
                    color = p_v if p_a == PointColor else self.default_values[PointColor]
                    symbol = p_v if p_a == PointSymbol else self.default_values[PointSymbol]
                    size = p_v if p_a == PointSize else self.default_values[PointSize]
                    self.items.append( point_item(10, y,  color, symbol, size, self) )
                    text = QGraphicsTextItem(d_v, self)
                    text.setPos(QPointF(20, y-10))
                    self.items.append(text)
                    y = y + 20
                y = y + 10
        else:
            for curve in self.curves:
                self.items.append(curve.pointItem(10, y, curve.pointSize(), self))
                text = QGraphicsTextItem(curve.name, self)
                length = max(length, text.boundingRect().width())
                text.setPos(QPointF(20, y-10))
                self.items.append(text)
                y = y + 20
            if self.curves:
                self.box_rect = QRectF(0, 0, 20 + length, y-10)
                box_item = QGraphicsRectItem(self.box_rect, self)
                box_item.setBrush(Qt.white)
                box_item.setZValue(-1)
                self.items.append(box_item)
            else:
                box_rect = QRectF()
                
    def mouseMoveEvent(self, event):
        self.setPos(self.pos() + event.scenePos() - event.lastScenePos())
        event.accept()
            
    def mousePressEvent(self, event):
        event.accept()

    def boundingRect(self):
        return self.box_rect
        
    def paint(self, painter, option, widget=None):
        pass
    
