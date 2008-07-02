from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.Qwt5 import *
import numpy

SelectionCurveRtti = QwtPlotCurve.Rtti_PlotUserItem + 123
LegendCurveRtti = QwtPlotCurve.Rtti_PlotUserItem + 124


# ####################################################################
# add val to sorted list list. if len > maxLen delete last element
def addToList(list, val, ind, maxLen):
    i = 0
    for i in range(len(list)):
        (val2, ind2) = list[i]
        if val < val2:
            list.insert(i, (val, ind))
            if len(list) > maxLen:
                list.remove(list[maxLen])
            return
    if len(list) < maxLen:
        list.insert(len(list), (val, ind))


#A dynamic tool tip class
class TooltipManager:
    # Creates a new dynamic tool tip.
    def __init__(self, qwtplot):
        self.qwtplot = qwtplot
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

        intX = abs(self.qwtplot.transform(self.qwtplot.xBottom, x) - self.qwtplot.transform(self.qwtplot.xBottom, self.positions[nearestIndex][0]))
        intY = abs(self.qwtplot.transform(self.qwtplot.yLeft, y) - self.qwtplot.transform(self.qwtplot.yLeft, self.positions[nearestIndex][1]))
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


# ####################################################################
# used in widgets that enable to draw a rectangle or a polygon to select a subset of data points
class SelectionCurve(QwtPlotCurve):
    def __init__(self, name = "", pen = Qt.SolidLine ):
        QwtPlotCurve.__init__(self, name)
        self.setStyle(QwtPlotCurve.Lines)
        self.setPen(QPen(QColor(128,128,128), 1, pen))
        self.setItemAttribute(QwtPlotItem.Legend, 0)

    def rtti(self):
        return SelectionCurveRtti

    def addPoint(self, xPoint, yPoint):
        self.setData([self.x(i) for i in range(self.dataSize())] + [xPoint], [self.y(i) for i in range(self.dataSize())] + [yPoint])

    def removeLastPoint(self):
        self.setData([self.x(i) for i in range(self.dataSize()-1)], [self.y(i) for i in range(self.dataSize()-1)])

    def replaceLastPoint(self, xPoint, yPoint):
        self.setData([self.x(i) for i in range(self.dataSize()-1)] + [xPoint], [self.y(i) for i in range(self.dataSize()-1)] + [yPoint])

    def getPointArray(self):
        return QPolygonF([QPointF(self.x(i), self.y(i)) for i in range(self.dataSize())] + [QPointF(self.x(0), self.y(0))])

    def getSelectedPoints(self, xData, yData, validData):
        pointArray = self.getPointArray()
        selected = numpy.zeros(len(xData))

        for i in range(len(xData)):
            if validData[i]:
                selected[i] = pointArray.containsPoint(QPointF(xData[i], yData[i]), Qt.OddEvenFill)
        return selected

    # is point defined at x,y inside a rectangle defined with this curve
    def isInside(self, x, y):
        return self.getPointArray().containsPoint(QPointF(x, y), Qt.OddEvenFill)

    def moveBy(self, dx, dy):
        xData = [self.x(i) + dx for i in range(self.dataSize())]
        yData = [self.y(i) + dy for i in range(self.dataSize())]
        self.setData(xData, yData)


    # test if the line going from before last and last point intersect any lines before
    # if yes, then add the intersection point and remove the outer points
    def closed(self):
        if self.dataSize() < 5: return 0
        x1 = self.x(self.dataSize()-3)
        x2 = self.x(self.dataSize()-2)
        y1 = self.y(self.dataSize()-3)
        y2 = self.y(self.dataSize()-2)
        for i in range(self.dataSize()-5, -1, -1):
            X1 = self.x(i)
            X2 = self.x(i+1)
            Y1 = self.y(i)
            Y2 = self.y(i+1)
            (intersect, xi, yi) = self.lineIntersection(x1, y1, x2, y2, X1, Y1, X2, Y2)
            if intersect:
                xData = [xi]; yData = [yi]
                for j in range(i+1, self.dataSize()-2): xData.append(self.x(j)); yData.append(self.y(j))
                xData.append(xi); yData.append(yi)
                self.setData(xData, yData)
                return 1
        return 0

    def lineIntersection(self, x1, y1, x2, y2, X1, Y1, X2, Y2):
        if min(x1,x2) > max(X1, X2) or max(x1,x2) < min(X1,X2): return (0, 0, 0)
        if min(y1,y2) > max(Y1, Y2) or max(y1,y2) < min(Y1,Y2): return (0, 0, 0)

        if x2-x1 != 0: k1 = (y2-y1)/(x2-x1)
        else:          k1 = 1e+12

        if X2-X1 != 0: k2 = (Y2-Y1)/(X2-X1)
        else:          k2 = 1e+12

        c1 = (y1-k1*x1)
        c2 = (Y1-k2*X1)

        if k1 == 1e+12:
            yTest = k2*x1 + c2
            if yTest > min(y1,y2) and yTest  < max(y1,y2): return (1, x1, yTest)
            else: return (0,0,0)

        if k2 == 1e+12:
            yTest = k1*X1 + c1
            if yTest > min(Y1,Y2) and yTest < max(Y1,Y2): return (1, X1, yTest)
            else: return (0,0,0)

        det_inv = 1/(k2 - k1)

        xi=((c1 - c2)*det_inv)
        yi=((k2*c1 - k1*c2)*det_inv)

        if xi >= min(x1, x2) and xi <= max(x1,x2) and xi >= min(X1, X2) and xi <= max(X1, X2) and yi >= min(y1,y2) and yi <= max(y1, y2) and yi >= min(Y1, Y2) and yi <= max(Y1, Y2):
            return (1, xi, yi)
        else:
            return (0, xi, yi)
        
    def isOnEdge(self, x, y):
        return 0

class RectangleSelectionCurve(SelectionCurve):
    def __init__(self, name = "", pen = Qt.SolidLine):
        SelectionCurve.__init__(self, name, pen)
        self.point1 = (0,0)
        self.point2 = (0,0)
        self.appropriateCursor = Qt.ArrowCursor
    
    def setPoints(self, x1, y1, x2, y2):
        self.point1, self.point2 = (x1,y1), (x2,y2)
        self.setData([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1])
        
    def approxEqual(self, p1, p2, axis = None):
        if type(p1) == tuple and type(p2) == tuple:
            x1, y1 = self.plot().transform(self.plot().xBottom, p1[0]), self.plot().transform(self.plot().yLeft, p1[1])
            x2, y2 = self.plot().transform(self.plot().xBottom, p2[0]), self.plot().transform(self.plot().yLeft, p2[1])
            return abs(x1-x2) + abs(y1-y2) <= 4
        else:
            v1, v2 = self.plot().transform(axis, p1), self.plot().transform(axis, p2)
            return abs(v1-v2) <= 2
    
    def between(self, val, v1, v2):
        return val >= min(v1, v2) and val <= max(v1, v2)
        
    # check if based on the mouse position (x,y) we should show a different cursor and enable resizing
    def isOnEdge(self, x, y):
        xData = [self.x(i) for i in range(self.dataSize())]
        yData = [self.y(i) for i in range(self.dataSize())]
        if len(xData) == 0: return 0
        x1, y1 = min(xData), min(yData)
        x2, y2 = max(xData), max(yData)
        
        if self.approxEqual((min(x1,x2), min(y1,y2)), (x,y)):
            self.point1, self.point2 = (max(x1,x2), max(y1,y2)), (min(x1,x2), min(y1,y2))
            self.appropriateCursor = Qt.SizeBDiagCursor
        elif self.approxEqual((max(x1,x2), max(y1,y2)), (x,y)):
            self.point1, self.point2 = (min(x1,x2), min(y1,y2)), (max(x1,x2), max(y1,y2))
            self.appropriateCursor = Qt.SizeBDiagCursor
        elif self.approxEqual((min(x1,x2), max(y1,y2)), (x,y)):
            self.point1, self.point2 = (max(x1,x2), min(y1,y2)), (min(x1,x2), max(y1,y2))
            self.appropriateCursor = Qt.SizeFDiagCursor
        elif self.approxEqual((max(x1,x2), min(y1,y2)), (x,y)):
            self.point1, self.point2 = (min(x1,x2), max(y1,y2)), (max(x1,x2), min(y1,y2))
            self.appropriateCursor = Qt.SizeFDiagCursor
        elif self.approxEqual(x1, x, self.plot().xBottom) and self.between(y, y1, y2):
            self.point1, self.point2 = (x2, y2), (x1,y1)
            self.appropriateCursor = Qt.SizeHorCursor
        elif self.approxEqual(x2, x, self.plot().xBottom) and self.between(y, y1, y2) :
            self.point1, self.point2 = (x1, y1), (x2,y2)
            self.appropriateCursor = Qt.SizeHorCursor
        elif self.approxEqual(y1, y, self.plot().yLeft) and self.between(x, x1, x2):
            self.point1, self.point2 = (x2, y2), (x1,y1)
            self.appropriateCursor = Qt.SizeVerCursor
        elif self.approxEqual(y2, y, self.plot().yLeft) and self.between(x, x1, x2):
            self.point1, self.point2 = (x1, y1), (x2,y2)
            self.appropriateCursor = Qt.SizeVerCursor
        else:
            self.appropriateCursor = Qt.ArrowCursor
        return self.appropriateCursor != Qt.ArrowCursor
    
    # update the curve with new x and y coordinates of one edge
    # here we assume that the isOnEdge was called first since it takes care of preparing the values in point1 and point2
    def updateCurve(self, x, y):
        if self.appropriateCursor in [Qt.SizeBDiagCursor, Qt.SizeFDiagCursor]:
            self.setPoints(self.point1[0], self.point1[1], x, y)
        elif self.appropriateCursor == Qt.SizeHorCursor:
            self.setPoints(self.point1[0], self.point1[1], x, self.point2[1])
        elif self.appropriateCursor == Qt.SizeVerCursor:
            self.setPoints(self.point1[0], self.point1[1], self.point2[0], y)

# a class that draws unconnected lines. first two points in the xData and yData are considered as the first line,
# the second two points as the second line, etc.
class UnconnectedLinesCurve(QwtPlotCurve):
    def __init__(self, name, pen = QPen(Qt.black), xData = None, yData = None):
        QwtPlotCurve.__init__(self, name)
        if pen.width() == 0:
            pen.setWidth(1)
        self.setPen(pen)
        self.Pen = pen
        self.setStyle(QwtPlotCurve.Lines)
        self.setItemAttribute(QwtPlotItem.Legend, 0)
        if xData != None and yData != None:
            self.setData(xData, yData)

    def drawCurve(self, painter, style, xMap, yMap, start, stop):
        start = max(start + start%2, 0)
        if stop == -1:
            stop = self.dataSize()
        for i in range(start, stop, 2):
            QwtPlotCurve.drawLines(self, painter, xMap, yMap, i, i+1)


class RectangleCurve(QwtPlotCurve):
    def __init__(self, pen = QPen(Qt.black), brush = QBrush(Qt.white), xData = None, yData = None):
        QwtPlotCurve.__init__(self)
        if pen:
            self.setPen(pen)
        if brush:
            self.setBrush(brush)
        self.Pen = pen
        self.Brush = brush
        self.setStyle(QwtPlotCurve.Lines)
        self.setItemAttribute(QwtPlotItem.Legend, 0)
        if xData != None and yData != None:
            self.setData(xData, yData)


    # To show a rectangle, we have to create a closed polygon.
    # Therefore we add to each rectangle the first point (each rect therefore contains 5 points in the xData and yData)
    def setData(self, xData, yData):
        startsX = xData[::4]
        startsY = yData[::4]
        for i in range(len(startsX))[::-1]:
            xData.insert(4+i*4, startsX[i])
            yData.insert(4+i*4, startsY[i])
        QwtPlotCurve.setData(self, xData, yData)

    def drawCurve(self, painter, style, xMap, yMap, start, stop):
        for i in range(start, stop, 5):
            QwtPlotCurve.drawLines(self, painter, xMap, yMap, i, i+4)


# ###########################################################
# a class that is able to draw arbitrary polygon curves.
# data points are specified by a standard call to graph.setCurveData(key, xArray, yArray)
# brush and pen can also be set by calls to setPen and setBrush functions
class PolygonCurve(QwtPlotCurve):
    def __init__(self, pen = QPen(Qt.black), brush = QBrush(Qt.white), xData = None, yData = None, tooltip = None):
        QwtPlotCurve.__init__(self)
        if pen:
            self.setPen(pen)
        if brush:
            self.setBrush(brush)
        self.Pen = pen
        self.Brush = brush
        self.setStyle(QwtPlotCurve.Lines)
        self.setItemAttribute(QwtPlotItem.Legend, 0)
        self.tooltip = tooltip
        if xData != None and yData != None:
            self.setData(xData, yData)


class errorBarQwtPlotCurve(QwtPlotCurve):
    def __init__(self, text = "", connectPoints = 0, tickXw = 0.1, tickYw = 0.1, showVerticalErrorBar = 1, showHorizontalErrorBar = 0):
        QwtPlotCurve.__init__(self, text)
        self.connectPoints = connectPoints
        self.tickXw = tickXw
        self.tickYw = tickYw
        self.showVerticalErrorBar = showVerticalErrorBar
        self.showHorizontalErrorBar = showHorizontalErrorBar
        self.setItemAttribute(QwtPlotItem.Legend, 0)

    def draw(self, p, xMap, yMap, f, t=-1):
        # save ex settings
        pen = p.pen()

        if type(f)==QRect:
            f = 0

        self.setPen( self.symbol().pen() )
        p.setPen( self.symbol().pen() )
        if self.style() == QwtPlotCurve.UserCurve:
            back = p.backgroundMode()

            p.setBackgroundMode(Qt.OpaqueMode)
            if t < 0: t = self.dataSize() - 1

            if divmod(f, 3)[1] != 0: f -= f % 3
            if divmod(t, 3)[1] == 0:  t += 1
            first = 1
            for i in range(f, t+1, 3):
                px = xMap.transform(self.x(i))
                py = yMap.transform(self.y(i))

                if self.showVerticalErrorBar:
                    vbxl = xMap.transform(self.x(i) - self.tickXw/2.0)
                    vbxr = xMap.transform(self.x(i) + self.tickXw/2.0)

                    vbyt = yMap.transform(self.y(i + 1))
                    vbyb = yMap.transform(self.y(i + 2))

                if self.showHorizontalErrorBar:
                    hbxl = xMap.transform(self.x(i + 1))
                    hbxr = xMap.transform(self.x(i + 2))

                    hbyt = yMap.transform(self.y(i) + self.tickYw/2.0)
                    hbyb = yMap.transform(self.y(i) - self.tickYw/2.0)

                if self.connectPoints:
                    if first:
                        first = 0
                    else:
                        p.drawLine(ppx, ppy, px, py)
                    ppx = px
                    ppy = py

                if self.showVerticalErrorBar:
                    p.drawLine(px,   vbyt, px,   vbyb)   ## |
                    p.drawLine(vbxl, vbyt, vbxr, vbyt) ## T
                    p.drawLine(vbxl, vbyb, vbxr, vbyb) ## _

                if self.showHorizontalErrorBar:
                    p.drawLine(hbxl, py,   hbxr, py)   ## -
                    p.drawLine(hbxl, hbyt, hbxl, hbyb) ## |-
                    p.drawLine(hbxr, hbyt, hbxr, hbyb) ## -|

                self.symbol().draw(p, px, py)

            p.setBackgroundMode(back)
        else:
            QwtPlotCurve.draw(self, p, xMap, yMap, f, t)

        # restore ex settings
        p.setPen(pen)


# ####################################################################
# create a marker in QwtPlot, that doesn't have a transparent background. Currently used in parallel coordinates widget.
class nonTransparentMarker(QwtPlotMarker):
    def __init__(self, backColor, *args):
        QwtPlotMarker.__init__(self, *args)
        self.backColor = backColor

    def draw(self, p, x, y, rect):
        p.setPen(self.labelPen())
        p.setFont(self.font())

        th = p.fontMetrics().height();
        tw = p.fontMetrics().width(self.label());
        r = QRect(x + 4, y - th/2 - 2, tw + 4, th + 4)
        p.fillRect(r, QBrush(self.backColor))
        p.drawText(r, Qt.AlignHCenter + Qt.AlignVCenter, self.label());



class RotatedMarker(QwtPlotMarker):
    def __init__(self, parent, label = "", x = 0.0, y = 0.0, rotation = 0):
        QwtPlotMarker.__init__(self, parent)
        self.rotation = rotation
        self.parent = parent
        self.x = x
        self.y = y
        self.setXValue(x)
        self.setYValue(y)
        self.parent = parent

        if rotation != 0: self.setLabel(label + "  ")
        else:             self.setLabel(label)

    def setRotation(self, rotation):
        self.rotation = rotation

    def draw(self, painter, x, y, rect):
        rot = math.radians(self.rotation)

        x2 = x * math.cos(rot) - y * math.sin(rot)
        y2 = x * math.sin(rot) + y * math.cos(rot)

        painter.rotate(-self.rotation)
        QwtPlotMarker.draw(self, painter, x2, y2, rect)
        painter.rotate(self.rotation)


# ####################################################################
# draw labels for discrete attributes
class DiscreteAxisScaleDraw(QwtScaleDraw):
    def __init__(self, labels):
        apply(QwtScaleDraw.__init__, (self,))
        self.labels = labels

    def label(self, value):
        index = int(round(value))
        if index != value: return QwtText("")    # if value not an integer value return ""
        if index >= len(self.labels) or index < 0: return QwtText("")
        return QwtText(str(self.labels[index]))

# ####################################################################
# use this class if you want to hide labels on the axis
class HiddenScaleDraw(QwtScaleDraw):
    def __init__(self, *args):
        QwtScaleDraw.__init__(self, *args)

    def label(self, value):
        return QwtText()

