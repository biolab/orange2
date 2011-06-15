

from PyQt4.QtGui import QGraphicsItemGroup, QGraphicsTextItem, QGraphicsRectItem, QColor
from PyQt4.QtCore import QPointF, QRectF, Qt

from point import *

PointColor = 1
PointSize = 2
PointSymbol = 4

class Legend(QGraphicsItemGroup):
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
                               PointSymbol : Ellipse
                               }
        
    def clear(self):
        self.curves = []
        self.update()
        

    def add_curve(self, curve, attributes = []):
        for point_attribute, data_attribute, value in attributes:
            if point_attribute not in self.point_attrs:
                self.point_attrs[point_attribute] = data_attribute
                
            if point_attribute == PointColor:
                point_val = curve.color()
            elif point_attribute == PointSize:
                point_val = curve.size()
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
            ## Using the OWGraph API to specify paremeters
            for p_a, d_a in self.point_attrs:
                ## We construct a separate box for each attribute 
                title_item = QGraphicsTextItem( d_a, self )
                title_item.setPos(QPointF(10, y-10))
                self.items.append(title_item)
                y = y + 20
                for p_v, d_v in self.point_vals[p_a]:
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
                box_rect = QRectF(0, 0, 20 + length, y-10)
                box_item = QGraphicsRectItem(box_rect, self)
                self.items.append(box_item)
