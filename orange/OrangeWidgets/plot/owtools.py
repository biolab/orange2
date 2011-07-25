from PyQt4.QtGui import QGraphicsItem, QGraphicsRectItem, QPolygonF, QGraphicsPolygonItem, QPen, QBrush
from PyQt4.QtCore import Qt, QRectF, QPointF, qDebug

from owcurve import *

"""
    Efficiently resizes a list of QGraphicsItems (PlotItems, Curves, etc.)
    If the list is to be reduced, i.e. if len(lst) > size, then the extra items are first removed from the scene
"""

def resize_plot_item_list(lst, size, item_type, parent):
    n = len(lst)
    if n > size:
        for i in lst[n:]:
            i.setParentItem(None)
        return lst[:n]
    elif n < size:
        return lst + [item_type(parent) for i in range(size - n)]
    else:
        return lst
        
#A dynamic tool tip class
class TooltipManager:
    # Creates a new dynamic tool tip.
    # The second argument is a OWGraph instance
    def __init__(self, graph):
        self.graph = graph
        self.positions=[]
        self.texts=[]

    # Adds a tool tip. If a tooltip with the same name already exists, it updates it instead of adding a new one.
    def addToolTip(self, x, y, text, customX = 0, customY = 0):
        self.positions.append((x,y, customX, customY))
        self.texts.append(text)

    #Decides whether to pop up a tool tip and which text to pop up
    def maybeTip(self, x, y):
        if len(self.positions) == 0: return ("", -1, -1)
        dists = [max(abs(x-position[0])- position[2],0) + max(abs(y-position[1])-position[3], 0) for position in self.positions]
        nearestIndex = dists.index(min(dists))
        
        intX = abs(self.graph.transform(xBottom, x) - self.graph.transform(xBottom, self.positions[nearestIndex][0]))
        intY = abs(self.graph.transform(yLeft, y) - self.graph.transform(yLeft, self.positions[nearestIndex][1]))
        if self.positions[nearestIndex][2] == 0 and self.positions[nearestIndex][3] == 0:   # if we specified no custom range then assume 6 pixels
            if intX + intY < 6:  return (self.texts[nearestIndex], self.positions[nearestIndex][0], self.positions[nearestIndex][1])
            else:                return ("", None, None)
        else:
            if abs(self.positions[nearestIndex][0] - x) <= self.positions[nearestIndex][2] and abs(self.positions[nearestIndex][1] - y) <= self.positions[nearestIndex][3]:
                return (self.texts[nearestIndex], x, y)
            else:
                return ("", None, None)

    def removeAll(self):
        self.positions = []
        self.texts = []

# Convenience curve classes
class PolygonCurve(OWCurve):
    def __init__(self, pen = QPen(Qt.black), brush = QBrush(Qt.white), xData = [], yData = [], tooltip = None):
        OWCurve.__init__(self, xData, yData, tooltip=tooltip)
        self._data_polygon = self.polygon_from_data(xData, yData)
        self._polygon_item = QGraphicsPolygonItem(self)
        self.set_pen(pen)
        self.set_brush(brush)
        
    def update_properties(self):
        self._polygon_item.setPolygon(self.graphTransform().map(self._data_polygon))
        self._polygon_item.setPen(self.pen())
        self._polygon_item.setBrush(self.brush())
        
    @staticmethod
    def polygon_from_data(xData, yData):
        if xData and yData:
            n = min(len(xData), len(yData))
            p = QPolygonF(n+1)
            for i in range(n):
                p[i] = QPointF(xData[i], yData[i])
            p[n] = QPointF(xData[0], yData[0])
            return p
        else:
            return QPolygonF()
            
    def set_data(self, xData, yData):
        self._data_polygon = self.polygon_from_data(xData, yData)
        OWCurve.set_data(self, xData, yData)
           
class RectangleCurve(OWCurve):
    def __init__(self, pen = QPen(Qt.black), brush = QBrush(Qt.white), xData = None, yData = None, tooltip = None):
        OWCurve.__init__(self, xData, yData, tooltip=tooltip)
        self.set_pen(pen)
        self.set_brush(brush)
        self._item = QGraphicsRectItem(self)
        
    def update_properties(self):
        self._item.setRect(self.graph_transform().mapRect(self.data_rect()))
        self._item.setPen(self.pen())
        self._item.setBrush(self.brush())
        
class UnconnectedLinesCurve(orangeplot.UnconnectedLinesCurve):
    def __init__(self, name, pen = QPen(Qt.black), xData = None, yData = None):
        orangeplot.UnconnectedLinesCurve.__init__(self, xData, yData)
        if pen:
            self.set_pen(pen)
        self.name = name
        
class CircleCurve(OWCurve):
    def __init__(self, pen = QPen(Qt.black), brush = QBrush(Qt.NoBrush), xCenter = 0.0, yCenter = 0.0, radius = 1.0):
        OWCurve.__init__(self)
        self.set_pen(pen)
        self.set_brush(brush)
        self._item = QGraphicsEllipseItem(self)
        self.center = xCenter, yCenter
        self.radius = radius
        self._rect = QRectF(xCenter-radius, yCenter-radius, 2*radius, 2*radius)
        
    def update_properties(self):
        self._item.setRect(self.graph_transform().mapRect(self.data_rect()))
        self._item.setPen(self.pen())
        self._item.setBrush(self.brush())
        
    def data_rect(self):
        x, y = self.center
        r = self.radius
        return QRectF(x-r, y-r, 2*r, 2*r)
        
class Marker(orangeplot.PlotItem):
    def __init__(self, text, x, y, align, bold = 0, color = None, brushColor = None, size=None):
        orangeplot.PlotItem.__init__(self)
        self._item = QGraphicsTextItem(text, parent=self)
        self._data_point = QPointF(x,y)
        f = self._item.font()
        f.setBold(bold)
        if size:
            f.setPointSize(size)
        self._item.setFont(f)
        self._item.setPos(x, y)
        if color:
            self._item.setPen(QPen(color))
        if brushColor:
            self._item.setBrush(QBrush(brushColor))
            
    def update_properties(self):
        self._item.setPos(self.graph_transform().map(self._data_point))

