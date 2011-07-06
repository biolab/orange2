
from PyQt4.QtGui import QGraphicsPathItem, QPen, QBrush
from PyQt4.QtCore import Qt, QPointF

from owcurve import *

def point_item(x, y, color = Qt.black, symbol = OWCurve.Ellipse, size = 5, parent = None):
    path = OWCurve.pathForSymbol(symbol, size)
    item = QGraphicsPathItem(path, parent)
    item.setPen(QPen(Qt.NoPen))
    item.setBrush(QBrush(color))
    item.setPos(QPointF(x, y))
    return item
    
