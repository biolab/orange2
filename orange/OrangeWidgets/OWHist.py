#
# OWHist.py
#
# the base for all histograms

import OWGUI
from OWWidget import *
from OWGraph import *

import numpy

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

    def setValues(self, values):
        nBins = 100
        if len(values) < 100:
            nBins = len(values)

        (self.yData, self.xData) = numpy.histogram(values, bins=100)
        
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
        
        self.lowerBoundaryKey.setData([self.lowerBoundary, self.lowerBoundary], [0, maxy])
        self.upperBoundaryKey.setData([self.upperBoundary, self.upperBoundary], [0, maxy])
        self.replot()
            
    def updateData(self):
        self.removeDrawingCurves(removeLegendItems = 0, removeMarkers=1)
                    
        self.key = self.addCurve("histogramCurve", Qt.blue, Qt.blue, 6, symbol = QwtSymbol.NoSymbol, style = QwtPlotCurve.Steps, xData = self.xData, yData = self.yData)
        
        maxy = self.maxy
        self.lowerBoundaryKey = self.addCurve("lowerBoundaryCurve", Qt.red, Qt.red, 6, symbol = QwtSymbol.NoSymbol, style = QwtPlotCurve.Lines, xData = [self.lowerBoundary, self.lowerBoundary], yData = [0, maxy])
        self.upperBoundaryKey = self.addCurve("upperBoundaryCurve", Qt.red, Qt.red, 6, symbol = QwtSymbol.NoSymbol, style = QwtPlotCurve.Lines, xData = [self.upperBoundary, self.upperBoundary], yData = [0, maxy])

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
            index = max(min(int(100*(self.upperBoundary-self.minx)/(self.maxx-self.minx)), 100), 0)
            x = [self.upperBoundary] + list(self.xData[index:])
            y = [self.yData[min(index, 99)]] + list(self.yData[index:])
            self.upperTailShadeKey.setData(x, y)
        if self.type in ["lowTail", "twoTail"]:
            index = max(min(int(100*(self.lowerBoundary-self.minx)/(self.maxx-self.minx)),100), 0)
            x = list(self.xData[:index]) + [self.lowerBoundary]
            y = list(self.yData[:index]) + [self.yData[min(index,99)]]
            self.lowerTailShadeKey.setData(x, y)
        if self.type in ["middle"]:
            indexLow = max(min(int(100*(self.lowerBoundary-self.minx)/(self.maxx-self.minx)),99), 0)
            indexHi = max(min(int(100*(self.upperBoundary-self.minx)/(self.maxx-self.minx)), 100)-1, 0)
            x = [self.lowerBoundary] + list(self.xData[indexLow: indexHi]) +[self.upperBoundary]
            y = [self.yData[max(index,0)]] + list(self.yData[indexLow: indexHi]) +[self.yData[max(indexHi, 99)]]
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
        self.replot()
    
    def _setBoundary(self, button, cut):
        if self.type in ["twoTail", "middle"]:
            if button == Qt.LeftButton:
                low, hi = cut, self.upperBoundary
            else:
                low, hi = self.lowerBoundary, cut
            if low > hi:
                low, hi = hi, low
            self.setBoundary(low, hi)
        else:
            self.setBoundary(cut, cut)
        
    def mousePressEvent(self, e):
        if self.state==SELECT:
            cut = self.invTransform(QwtPlot.xBottom, self.canvas().mapFrom(self, e.pos()).x())
            self.mouseCurrentlyPressed = 1
            self.buttonCurrentlyPressed = e.button()
            self._setBoundary(e.button(), cut)
        else:
            return OWHist.mousePressEvent(self, e)
        
    def mouseMoveEvent(self, e):
        if self.state==SELECT:
            if self.mouseCurrentlyPressed:
                cut = self.invTransform(QwtPlot.xBottom, self.canvas().mapFrom(self, e.pos()).x())
                self._setBoundary(self.buttonCurrentlyPressed, cut)
        else:
            return OWHist.mouseMoveEvent(self ,e)

    def mouseReleaseEvent(self, e):
        if self.state==SELECT:
            cut = self.invTransform(QwtPlot.xBottom, self.canvas().mapFrom(self, e.pos()).x())
            self._setBoundary(self.buttonCurrentlyPressed, cut)
            self.mouseCurrentlyPressed = 0
            self.buttonCurrentlyPressed = None
        else:
            return OWHist.mouseReleaseEvent(self, e)
