from qt import *
from qtcanvas import *

def OWCanvasText(canvas, text, x  = 0, y = 0, alignment = Qt.AlignLeft + Qt.AlignVCenter, bold = 0, font = None, z = 0, show = 1):
    text = QCanvasText(text, canvas)
    text.setTextFlags(alignment)

    if font:
        text.setFont(font)
    if bold:
        font = text.font(); font.setBold(bold); text.setFont(font)
    text.move(x, y)
    text.setZ(z)

    if show: text.show()
    else: text.hide()
    return text

def OWCanvasRectangle(canvas, x, y, width, height, penColor = Qt.black, brushColor = None, penWidth = 1, z = 0, penStyle = Qt.SolidLine, show = 1):
    rect = QCanvasRectangle(x, y, width, height, canvas)
    if brushColor: rect.setBrush(QBrush(brushColor))
    rect.setPen(QPen(penColor, penWidth, penStyle))
    rect.setZ(z)
    if show: rect.show()
    return rect


def OWCanvasLine(canvas, x1, y1, x2, y2, penWidth = 1, penColor = Qt.black, z = 0, show = 1):
    r = QCanvasLine(canvas)
    r.setPoints(x1, y1, x2, y2)
    r.setPen(QPen(penColor, penWidth))
    r.setZ(z)
    if show: r.show()
    return r

def OWCanvasEllipse(canvas, x, y, width, height, penWidth = 1, startAngle = 0, angles = 360, penColor = Qt.black, brushColor = None, z = 0, penStyle = Qt.SolidLine, show = 1):
    if penColor != None and brushColor == None:
        # if we dont want to fill the ellipse then we have to draw it with a series of lines - QCanvasEllipse always draws with NoPen
        p = QPointArray()
        p.makeArc(x, y, width, height, startAngle, angles*16)
        lines = []
        for i in range(0,p.size(),2):
            l = OWCanvasLine(canvas, p.point(i)[0], p.point(i)[1], p.point((i+2)%p.size())[0], p.point((i+2)%p.size())[1], penWidth, penColor, z, show)
            lines.append(l)
        return lines
    else:        
        e = QCanvasEllipse(width, height, canvas)
        e.setX(x)
        e.setY(y)
        e.setZ(z)
        e.setBrush(QBrush(brushColor))
        e.setAngles(startAngle, angles*16)
        if show: e.show()
        return e
    