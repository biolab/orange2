
from PyQt4.QtGui import QGraphicsPathItem, QPen, QBrush
from PyQt4.QtCore import Qt, QPointF

import curve

NoSymbol = -1
Ellipse = 0
Rect = 1
Diamond = 2
Triangle = 3
DTriangle = 4
UTriangle = 5
LTriangle = 6
RTriangle = 7
Cross = 8
XCross = 9
HLine = 10
VLine = 11
Star1 = 12
Star2 = 13
Hexagon = 14
UserStyle = 1000

def point_item(x, y, color = Qt.black, symbol = Ellipse, size = 5, parent = None):
    path = curve.Curve.pathForSymbol(symbol, size)
    item = QGraphicsPathItem(path, parent)
    item.setPen(QPen(Qt.NoPen))
    item.setBrush(QBrush(color))
    item.setPos(QPointF(x, y))
    return item
    
