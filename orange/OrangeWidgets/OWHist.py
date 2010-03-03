#
# OWHist.py
#
# the base for all histograms

import OWGUI
from OWWidget import *
from OWGraph import *

import numpy, math

class OWHist(OWGraph):
    def __init__(self, parent=None, type=0):
        OWGraph.__init__(self, parent, "Histogram")
        self.parent = parent
        self.type = type
        
        self.enableXaxis(1)
        self.enableYLaxis(1)
        
        self.xData = []
        self.yData = []
        
        self.minValue = 0
        self.maxValue = 0
        self.lowerBoundary = 0
        self.upperBoundary = 0
        self.lowerBoundaryKey = None
        self.upperBoundaryKey = None
        
        self.enableGridXB(False)
        self.enableGridYL(False)

        self.buttonCurrentlyPressed = None
        self.updatingState = False, None

    def setValues(self, values):
        nBins = 100
        if len(values) < 100:
            nBins = len(values)

        (self.yData, self.xData) = numpy.histogram(values, bins=100)
        #if numpy version greater than 1.3
        if len(self.xData) == len(self.yData) + 1:
            self.xData = [(self.xData[i] + self.xData[i+1]) / 2. for i in range(len(self.xData) - 1)]
        
        self.minx = min(self.xData)
        self.maxx = max(self.xData)
        self.miny = min(self.yData)
        self.maxy = max(self.yData)
        
        self.minValue = min(values)
        self.maxValue = max(values)
        
        self.updateData()
        self.replot()
        
    def setBoundary(self, lower, upper):
        self.lowerBoundary = lower
        self.upperBoundary = upper
        maxy = max(self.yData)
        
        self.lowerBoundaryKey.setData([float(self.lowerBoundary), float(self.lowerBoundary)], [0., float(maxy)])
        self.upperBoundaryKey.setData([float(self.upperBoundary), float(self.upperBoundary)], [0., float(maxy)])
#        self.updateData()
        self.replot()
            
    def updateData(self):
        self.removeDrawingCurves(removeLegendItems = 0, removeMarkers=1)
                    
        self.key = self.addCurve("histogramCurve", Qt.blue, Qt.blue, 6, symbol = QwtSymbol.NoSymbol, style = QwtPlotCurve.Steps, xData = self.xData, yData = self.yData)
        
        maxy = self.maxy
        self.lowerBoundaryKey = self.addCurve("lowerBoundaryCurve", Qt.red, Qt.red, 6, symbol = QwtSymbol.NoSymbol, style = QwtPlotCurve.Lines, xData = [float(self.lowerBoundary), float(self.lowerBoundary)], yData = [0., float(maxy)])
        self.upperBoundaryKey = self.addCurve("upperBoundaryCurve", Qt.red, Qt.red, 6, symbol = QwtSymbol.NoSymbol, style = QwtPlotCurve.Lines, xData = [float(self.upperBoundary), float(self.upperBoundary)], yData = [0., float(maxy)])

        minx = self.minx
        maxx = self.maxx
        miny = self.miny

        self.setAxisScale(QwtPlot.xBottom, minx - (0.05 * (maxx - minx)), maxx + (0.05 * (maxx - minx)))
        self.setAxisScale(QwtPlot.yLeft, miny - (0.05 * (maxy - miny)), maxy + (0.05 * (maxy - miny)))

class OWInteractiveHist(OWHist):
    shadeTypes = ["lowTail", "hiTail", "twoTail", "middle"]
    def updateData(self):
        OWHist.updateData(self)
        self.upperTailShadeKey = self.addCurve("upperTailShade", Qt.blue, Qt.blue, 6, symbol = QwtSymbol.NoSymbol, style = QwtPlotCurve.Steps)
        self.lowerTailShadeKey = self.addCurve("lowerTailShade", Qt.blue, Qt.blue, 6, symbol = QwtSymbol.NoSymbol, style = QwtPlotCurve.Steps)
        self.middleShadeKey = self.addCurve("middleShade", Qt.blue, Qt.blue, 6, symbol = QwtSymbol.NoSymbol, style = QwtPlotCurve.Steps)

        self.upperTailShadeKey.setBrush(QBrush(Qt.blue))
        self.lowerTailShadeKey.setBrush(QBrush(Qt.blue))
        self.middleShadeKey.setBrush(QBrush(Qt.blue))
##        self.setCurveBrush(self.upperTailShadeKey, QBrush(Qt.red))
##        self.setCurveBrush(self.lowerTailShadeKey, QBrush(Qt.red))
##        self.setCurveBrush(self.middleShadeKey, QBrush(Qt.red))

    def shadeTails(self):
        if self.type in ["hiTail", "twoTail"]:
            index = max(min(int(math.ceil(100*(self.upperBoundary-self.minx)/(self.maxx-self.minx))), 100), 0)
            x = [self.upperBoundary] + list(self.xData[index:])
            y = [self.yData[min(index, 99)]] + list(self.yData[index:])
            x = [float(a) for a  in x]
            y = [float(a) for a  in y]
            self.upperTailShadeKey.setData(x, y)
        if self.type in ["lowTail", "twoTail"]:
            index = max(min(int(math.ceil(100*(self.lowerBoundary-self.minx)/(self.maxx-self.minx))),100), 0)
            x = list(self.xData[:index]) + [self.lowerBoundary]
            y = list(self.yData[:index]) + [self.yData[min(index,99)]]
            x = [float(a) for a  in x]
            y = [float(a) for a  in y]
            self.lowerTailShadeKey.setData(x, y)
        if self.type in ["middle"]:
            indexLow = max(min(int(100*(self.lowerBoundary-self.minx)/(self.maxx-self.minx)),99), 0)
            indexHi = max(min(int(100*(self.upperBoundary-self.minx)/(self.maxx-self.minx)), 100)-1, 0)
            x = [self.lowerBoundary] + list(self.xData[indexLow: indexHi]) +[self.upperBoundary]
            y = [self.yData[max(index,0)]] + list(self.yData[indexLow: indexHi]) +[self.yData[max(indexHi, 99)]]
            x = [float(a) for a  in x]
            y = [float(a) for a  in y]
            self.middleShadeKey.setData(x, y)
        if self.type in ["hiTail", "middle"]:
            self.lowerTailShadeKey.setData([], [])
        if self.type in ["lowTail", "middle"]:
            self.upperTailShadeKey.setData([], [])
        if self.type in ["lowTail", "hiTail", "twoTail"]:
            self.middleShadeKey.setData([], [])
        
    def setBoundary(self, low, hi):
        OWHist.setBoundary(self, low, hi)
        self.shadeTails()
#        self.updateData()
        self.replot()
    
    def _setBoundary(self, boundary, cut):
        if self.type in ["twoTail", "middle"]:
            if boundary == "lower":
                low, hi = cut, self.upperBoundary
            else:
                low, hi = self.lowerBoundary, cut
            if low > hi:
                low, hi = hi, low
            self.setBoundary(low, hi)
        else:
            self.setBoundary(cut, cut)
        
    def mousePressEvent(self, e):
        if self.state == SELECT and self.getBoundaryAt(e.pos()) and e.button() == Qt.LeftButton:
            boundary = self.getBoundaryAt(e.pos())
            cut = self.invTransform(QwtPlot.xBottom, self.canvas().mapFrom(self, e.pos()).x())
##            self.mouseCurrentlyPressed = 1
##            self.buttonCurrentlyPressed = e.button()
            self.updatingState = True, boundary
            self._setBoundary(boundary, cut)
        else:
            return OWHist.mousePressEvent(self, e)
        
    def mouseMoveEvent(self, e):
        if self.state == SELECT:
            updating, boundary = self.updatingState
            if updating:
                cut = self.invTransform(QwtPlot.xBottom, self.canvas().mapFrom(self, e.pos()).x())
                self._setBoundary(boundary, cut)
            else:
                if self.getBoundaryAt(e.pos()):
                    self.canvas().setCursor(Qt.SizeHorCursor)
                else:
                    self.canvas().setCursor(self._cursor)
        else:
            return OWHist.mouseMoveEvent(self ,e)        

    def mouseReleaseEvent(self, e):
        updating, boundary = self.updatingState
        if self.state == SELECT and updating:
            cut = self.invTransform(QwtPlot.xBottom, self.canvas().mapFrom(self, e.pos()).x())
            self._setBoundary(boundary, cut)
            self.updatingState = False, None
        else:
            return OWHist.mouseReleaseEvent(self, e)

    def getBoundaryAt(self, pos):
        x = self.canvas().mapFrom(self, pos).x()
        def check (boundary):
            return abs(self.transform(QwtPlot.xBottom, boundary) - x) <= 3
        if self.type in ["hiTail", "twoTail", "middleTail"] and check(self.upperBoundary):
            return "upper"
        elif self.type in ["lowTail", "twoTail", "middleTail"] and check(self.lowerBoundary):
            return "lower"
        else:
            return None
            
