from PyQt4.QtGui import QGraphicsRectItem, QPolygonF, QGraphicsPolygonItem, QPen, QBrush
from PyQt4.QtCore import Qt, QRectF, QPointF, qDebug

from curve import *

"""
    Efficiently resizes a list of QGraphicsItems (PlotItems, Curves, etc.)
    If the list is to be reduced, i.e. if len(lst) > size, then the extra items are first removed from the scene
"""

def resize_plot_item_list(lst, size, item_type, parent):
    n = len(lst)
    if n > size:
        if parent and parent.scene():
            for i in lst[n:]:
                parent.scene().removeItem(i)
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

        intX = abs(self.graph.transform(self.graph.xBottom, x) - self.graph.transform(self.graph.xBottom, self.positions[nearestIndex][0]))
        intY = abs(self.graph.transform(self.graph.yLeft, y) - self.graph.transform(self.graph.yLeft, self.positions[nearestIndex][1]))
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
class PolygonCurve(Curve):
    def __init__(self, pen = QPen(Qt.black), brush = QBrush(Qt.white), xData = [], yData = [], tooltip = None):
        Curve.__init__(self, xData, yData, tooltip=tooltip)
        self._data_polygon = self.polygon_from_data(xData, yData)
        self._polygon_item = QGraphicsPolygonItem(self)
        self.pen = pen
        self.brush = brush
        
    def updateProperties(self):
        self._polygon_item.setPolygon(self.graphTransform().map(self._data_polygon))
        self._polygon_item.setPen(self.pen)
        self._polygon_item.setBrush(self.brush)
        
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
            

class RectangleCurve(Curve):
    def __init__(self, pen = QPen(Qt.black), brush = QBrush(Qt.white), xData = None, yData = None, tooltip = None):
        Curve.__init__(self, xData, yData, tooltip=tooltip)
        self.setPen(pen)
        self.setBrush(brush)
        self._item = QGraphicsRectItem()
        
    def updateProperties(self):
        d = self.data()
        if len(d) < 4:
            self._item.setRect(QRectF())
        else:
            self._item.setRect(self.graphTransform().mapRect(self.boundingRectFromData(d)))

class UnconnectedLinesCurve(orangegraph.UnconnectedLinesCurve):
    def __init__(self, name, pen = QPen(Qt.black), xData = None, yData = None):
        orangegraph.UnconnectedLinesCurve.__init__(self, xData, yData)
        if pen:
            self.setPen(pen)
        self.name = name
