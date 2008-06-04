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

    def setValues(self, values):
        nBins = 100
        if len(values) < 100:
            nBins = len(values)

        (self.yData, self.xData) = numpy.histogram(values, bins=100)
        
        self.minx = min(self.xData)
        self.maxx = max(self.xData)
        self.miny = min(self.yData)
        self.maxy = max(self.yData)
        
        self.minValue = self.minx
        self.maxValue = self.maxx
        
        self.updateData()
        self.replot()
        
    def setBoundary(self, lower, upper):
        self.lowerBoundary = lower
        self.upperBoundary = upper
        maxy = max(self.yData)
        
        self.setCurveData(self.lowerBoundaryKey, [self.lowerBoundary, self.lowerBoundary], [0, maxy])
        self.setCurveData(self.upperBoundaryKey, [self.upperBoundary, self.upperBoundary], [0, maxy])
        self.replot()
            
    def updateData(self):
        self.removeDrawingCurves(removeLegendItems = 0)
                    
        self.key = self.addCurve("histogramCurve", Qt.blue, Qt.blue, 6, symbol = QwtSymbol.None, style = QwtPlotCurve.Steps, xData = list(self.xData), yData = list(self.yData))
        
        maxy = self.maxy
        self.lowerBoundaryKey = self.addCurve("lowerBoundaryCurve", Qt.red, Qt.red, 6, symbol = QwtSymbol.None, style = QwtPlotCurve.Lines, xData = [self.lowerBoundary, self.lowerBoundary], yData = [0, maxy])
        self.upperBoundaryKey = self.addCurve("upperBoundaryCurve", Qt.red, Qt.red, 6, symbol = QwtSymbol.None, style = QwtPlotCurve.Lines, xData = [self.upperBoundary, self.upperBoundary], yData = [0, maxy])

        minx = self.minx
        maxx = self.maxx
        miny = self.miny

        self.setAxisScale(QwtPlot.xBottom, minx - (0.05 * (maxx - minx)), maxx + (0.05 * (maxx - minx)))
        self.setAxisScale(QwtPlot.yLeft, miny - (0.05 * (maxy - miny)), maxy + (0.05 * (maxy - miny)))

class OWInteractiveHist(OWHist):
    def _setBoundary(self, button, cut):
        if self.type==1:
            if button == QMouseEvent.LeftButton:
                low, hi = cut, self.upperBoundary
            else:
                low, hi = self.lowerBoundary, cut
            if low > hi:
                low, hi = hi, low
            self.setBoundary(low, hi)
        else:
            self.setBoundary(cut, cut)
        
    def onMousePressed(self, e):
        cut = self.invTransform(QwtPlot.xBottom, e.x())
        self.mouseCurrentlyPressed = 1
        self._setBoundary(e.button(), cut)
        
    def onMouseMoved(self, e):
        if self.mouseCurrentlyPressed:
            cut = self.invTransform(QwtPlot.xBottom, e.x())
            self._setBoundary(e.state(), cut)

    def onMouseReleased(self, e):
        cut = self.invTransform(QwtPlot.xBottom, e.x())
        self.mouseCurrentlyPressed = 0
        self._setBoundary(e.button(), cut)
