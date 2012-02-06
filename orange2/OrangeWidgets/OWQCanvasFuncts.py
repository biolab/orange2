from PyQt4.QtCore import *
from PyQt4.QtGui import *

class OWCanvasText(QGraphicsTextItem):
    def __init__(self, canvas, text = "", x  = 0, y = 0, alignment = Qt.AlignLeft | Qt.AlignTop, bold = 0, font = None, z = 0, htmlText=None, tooltip = None, show = 1):
        QGraphicsTextItem.__init__(self, text, None, canvas)

        if font:
            self.setFont(font)
        if bold:
            font = self.font(); font.setBold(bold); self.setFont(font)
        if htmlText:
            self.setHtml(htmlText)

        self.alignment = alignment

        self.setPos(x, y)
        self.x, self.y = x, y
        self.setZValue(z)
        if tooltip: self.setToolTip(tooltip)

        if show: self.show()
        else: self.hide()

    def setPos(self, x, y):
        self.x, self.y = x, y
        rect = self.boundingRect()
        if int(self.alignment & Qt.AlignRight):     x -= rect.width()
        elif int(self.alignment & Qt.AlignHCenter): x-= rect.width()/2.
        if int(self.alignment & Qt.AlignBottom):    y-= rect.height()
        elif int(self.alignment & Qt.AlignVCenter): y-= rect.height()/2.
        QGraphicsTextItem.setPos(self, x, y)


def OWCanvasRectangle(canvas, x = 0, y = 0, width = 0, height = 0, penColor = Qt.black, brushColor = None, penWidth = 1, z = 0, penStyle = Qt.SolidLine, pen = None, tooltip = None, show = 1):
    rect = QGraphicsRectItem(x, y, width, height, None, canvas)
    if brushColor: rect.setBrush(QBrush(brushColor))
    if pen: rect.setPen(pen)
    else:   rect.setPen(QPen(penColor, penWidth, penStyle))
    rect.setZValue(z)
    if tooltip: rect.setToolTip(tooltip)
    if show: rect.show()
    else: rect.hide()
    return rect


def OWCanvasLine(canvas, x1 = 0, y1 = 0, x2 = 0, y2 = 0, penWidth = 1, penColor = Qt.black, pen = None, z = 0, tooltip = None, show = 1):
    r = QGraphicsLineItem(x1, y1, x2, y2, None, canvas)
    if pen != None:
        r.setPen(pen)
    else:
        r.setPen(QPen(penColor, penWidth))
    r.setZValue(z)
    if tooltip: r.setToolTip(tooltip)
    
    if show: r.show()
    else: r.hide()
    
    return r

def OWCanvasEllipse(canvas, x = 0, y = 0, width = 0, height = 0, penWidth = 1, startAngle = 0, angles = 360, penColor = Qt.black, brushColor = None, z = 0, penStyle = Qt.SolidLine, pen = None, tooltip = None, show = 1):
    e = QGraphicsEllipseItem(x, y, width, height, None, canvas)
    e.setZValue(z)
    if brushColor != None:
        e.setBrush(QBrush(brushColor))
    if pen != None: e.setPen(pen)
    else:           e.setPen(QPen(penColor, penWidth))
    e.setStartAngle(startAngle)
    e.setSpanAngle(angles*16)
    if tooltip: e.setToolTip(tooltip)
    
    if show: e.show()
    else: e.hide()
    
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
