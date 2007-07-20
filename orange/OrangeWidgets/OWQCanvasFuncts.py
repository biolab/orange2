from PyQt4.QtCore import *
from PyQt4.QtGui import *

class OWCanvasText(QGraphicsTextItem):
    def __init__(self, canvas, text, x  = 0, y = 0, alignment = 0, bold = 0, font = None, z = 0, tooltip = None, show = 1):
        QGraphicsTextItem.__init__(self, text, None, canvas)

        if font:
            self.setFont(font)
        if bold:
            font = self.font(); font.setBold(bold); self.setFont(font)

        self.alignment = alignment

        self.setPos(x, y)
        self.setZValue(z)
        if tooltip: self.setToolTip(tooltip)

        if show: self.show()
        else: self.hide()

    def setPos(self, x, y):
        rect = self.boundingRect()
        if self.alignment & Qt.AlignRight:     x -= rect.width()
        elif self.alignment & Qt.AlignHCenter: x-= rect.width()/2.
        if self.alignment & Qt.AlignBottom:    y-= rect.height()
        elif self.alignment & Qt.AlignVCenter: y-= rect.height()/2.
        QGraphicsTextItem.setPos(self, x, y)


def OWCanvasRectangle(canvas, x, y, width, height, penColor = Qt.black, brushColor = None, penWidth = 1, z = 0, penStyle = Qt.SolidLine, pen = None, tooltip = None, show = 1):
    rect = QGraphicsRectItem(x, y, width, height, None, canvas)
    if brushColor: rect.setBrush(QBrush(brushColor))
    if pen: rect.setPen(pen)
    else:   rect.setPen(QPen(penColor, penWidth, penStyle))
    rect.setZValue(z)
    if tooltip: rect.setToolTip(tooltip)
    if show: rect.show()
    return rect


def OWCanvasLine(canvas, x1, y1, x2, y2, penWidth = 1, penColor = Qt.black, z = 0, tooltip = None, show = 1):
    r = QGraphicsLineItem(x1, y1, x2, y2, None, canvas)
    r.setPen(QPen(penColor, penWidth))
    r.setZValue(z)
    if tooltip: r.setToolTip(tooltip)
    if show: r.show()
    return r

def OWCanvasEllipse(canvas, x, y, width, height, penWidth = 1, startAngle = 0, angles = 360, penColor = Qt.black, brushColor = None, z = 0, penStyle = Qt.SolidLine, tooltip = None, show = 1):
    e = QGraphicsEllipseItem(x, y, width, height, None, canvas)
    e.setZValue(z)
    if brushColor != None:
        e.setBrush(QBrush(brushColor))
    e.setPen(QPen(penColor))
    e.setStartAngle(startAngle)
    e.setSpanAngle(angles*16)
    if tooltip: e.setToolTip(tooltip)
    if show: e.show()
    return e

#    if penColor != None and brushColor == None:
#        # if we dont want to fill the ellipse then we have to draw it with a series of lines - QCanvasEllipse always draws with NoPen
#        p = QPointArray()
#        p.makeArc(x, y, width, height, startAngle, angles*16)
#        lines = []
#        for i in range(0,p.size(),2):
#            l = OWCanvasLine(canvas, p.point(i)[0], p.point(i)[1], p.point((i+2)%p.size())[0], p.point((i+2)%p.size())[1], penWidth, penColor, z, show)
#            lines.append(l)
#        return lines
#    else:
#        e = QCanvasEllipse(width, height, canvas)
#        e.setX(x)
#        e.setY(y)
#        e.setZValue(z)
#        e.setBrush(QBrush(brushColor))
#        e.setAngles(startAngle, angles*16)
#        if show: e.show()
#        return e
