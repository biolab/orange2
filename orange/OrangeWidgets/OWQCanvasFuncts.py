from qt import *
from qtcanvas import *

def OWCanvasText(canvas, text, x  = 0, y = 0, alignment = Qt.AlignLeft + Qt.AlignVCenter, bold = 0, font = None, show = 1):
    text = QCanvasText(text, canvas)
    text.setTextFlags(alignment)

    if font:
        text.setFont(font)
    if bold:
        font = text.font(); font.setBold(bold); text.setFont(font)
    text.move(x, y)
    if show: text.show()
    return text

def OWCanvasRectangle(canvas, x, y, width, height, penColor = Qt.black, brushColor = None, penWidth = 1, z = 0, show = 1):
    rect = QCanvasRectangle(x, y, width, height, canvas)
    if brushColor: rect.setBrush(QBrush(brushColor))
    if penColor:   rect.setPen(QPen(penColor))
    pen = rect.pen(); pen.setWidth(penWidth); rect.setPen(pen)
    rect.setZ(z)
    if show: rect.show()
    
    return rect


def OWCanvasLine(canvas, x1, y1, x2, y2, penWidth = 1, penColor = None, z = 0, show = 1):
    r = QCanvasLine(canvas)
    r.setPoints(x1, y1, x2, y2)
    if penColor: r.setPen(QPen(penColor))
    pen = r.pen(); pen.setWidth(penWidth); r.setPen(pen)
    r.setZ(z)
    if show: r.show()
    return r