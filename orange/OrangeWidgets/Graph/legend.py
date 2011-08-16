

from PyQt4.QtGui import QGraphicsItemGroup, QGraphicsTextItem, QGraphicsRectItem
from PyQt4.QtCore import QPointF, QRectF

class Legend(QGraphicsItemGroup):
    def __init__(self, scene):
        QGraphicsItemGroup.__init__(self, None, scene)
        self.curves = []
        self.items = []
        
    def clear(self):
        self.curves = []
        self.update()
        

    def add_curve(self, curve):
        self.curves.append(curve)
        self.update()
        
    def update(self):
        for i in self.items:
            self.scene().removeItem(i)
        del self.items[:]
        y = 10
        length = 0
        for curve in self.curves:
            self.items.append(curve.symbol(10, y, parent=self))
            text = QGraphicsTextItem(curve.name, self)
            length = max(length, text.boundingRect().width())
            text.setPos(QPointF(20, y-10))
            self.items.append(text)
            y = y + 20
        box_rect = QRectF(0, 0, 20 + length, y-10)
        box_item = QGraphicsRectItem(box_rect, self)
        self.items.append(box_item)
